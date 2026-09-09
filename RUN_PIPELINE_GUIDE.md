# `run_pipeline.py` usage guide

`run_pipeline.py` (repo root) runs Phase 0 (optional) &rarr; Phase 1 &rarr;
Phase 2 &rarr; Phase 3 (distant reading) &rarr; standalone report export
on **one new corpus**, in a single command. It calls the exact same
functions the notebooks call (`phase1/pipeline.py`, `phase2/pipeline.py`,
`phase2/condense.py`, `phase2/condensation_report.py`,
`common/standalone_report.py`), plus `phase3/pipeline.py` for the one
step with no notebook equivalent -- interactive prompts, outputs, and
decision logs are identical to running `phase0.ipynb` (optional) +
`phase1.ipynb` + `phase2.ipynb` by hand, with Phase 3 running
automatically in between Phase 2's report-building and its
regeneration loop. See the root [README.md](README.md) for the
pipeline's overall shape and the [web interface](README.md#web-interface)
if you'd rather drive this through a browser instead of a terminal.

## Prerequisites

- `pip install -r requirements.txt`
- `python -m spacy download en_core_web_sm`
- A GPU is recommended for Phase 1 (GlossBERT + embeddings); CPU works
  but is slow.
- For condensation: [Ollama](https://ollama.com) installed, a model
  pulled (`ollama pull <model>`), and `ollama serve` running.

See README.md's "Installation" section for the full setup.

## Flag reference

| Flag | Default | Meaning |
|---|---|---|
| `--corpus` (required unless `--list-corpora`) | -- | Corpus name; determines `data/<corpus>/`. For `--start-phase 2`/`3` this must be an existing corpus, not a new one |
| `--input` | -- | Path to the raw corpus file (`.txt`/`.md`/`.markdown`/`.pdf`). Required when `--start-phase` is 1 (the default); must be omitted for `--start-phase 2`/`3`, which reuse the corpus's already-saved raw text instead |
| `--clean` | off | Run Phase 0 cleaning (`phase0/clean_corpus.py`) before Phase 1. Only meaningful with `--start-phase 1` |
| `--start-phase` | `1` | Which phase to start at. `1`: a new corpus, needs `--input`, runs Phase 1 -> 2 -> 3. `2`: resume an existing `--corpus` at Phase 2, reusing its saved raw text and Phase 1 state (needs `data/<corpus>/raw/<corpus>.txt` and `<corpus>-phase1_state.json` already present) -- runs Phase 2 -> 3. `3`: resume an existing `--corpus` at Phase 3 only, reusing its saved raw text, Phase 1 state, and at least one already-generated condensed rate -- does not regenerate condensations, just re-runs the distant-reading report. Checked via `common/resume.py::check_prerequisites` before anything runs; a missing file aborts with a clear message listing exactly what's absent, instead of starting the desired phase |
| `--list-corpora` | off | List every corpus under `data/` and which `--start-phase` values it currently satisfies the prerequisites for (plus any already-generated condensed rates), then exit without running anything |
| `--frequency-percentile` | `0.50` | Percentile cutoff on the frequency-ranked stem vocabulary -- higher = fewer, more frequent stems (e.g. `0.75` keeps only the top 25% most-frequent stems) |
| `--max-stems` | `150` | Maximum number of stems to return |
| `--word-frequency-percentile` | `0.0` | Percentile cutoff on each stem's OWN derived-word frequency ranking (`0` keeps every derived word -- no filtering). Higher = fewer, more frequent derived words per stem (e.g. `0.75` keeps only the top 25% most-frequent words derived from each stem) -- trims how many distinct words (e.g. "observation"/"observes"/"observed" all under stem "observ") reach GlossBERT/flagged-term review. Always keeps at least one word per stem |
| `--sweep` | off | Before extracting stems, interactively sweep several candidate `--frequency-percentile` values (reporting each candidate's true stem count -- uncapped by `--max-stems` -- sentence coverage, and informative-sentence counts) and pick one -- overrides `--frequency-percentile`/`--max-stems` |
| `--max-sentences-per-stem` | `5` | Sentences collected per stem for WSD |
| `--max-synsets` | `5` | WordNet definitions considered per stem |
| `--merge-duplicate-word-occurrences` | off | Merge all sampled occurrences of the same word into one combined context and run GlossBERT once per word instead of once per occurrence -- avoids the same word being flagged for review more than once when its occurrences land on different senses |
| `--max-cluster-size` | `10` | Cluster size above which it gets re-split |
| `--min-clusters` | `5` | HDBSCAN `min_cluster_size` |
| `--min-cluster-len` | `3` | Minimum stems for a rerun/noise cluster to be valid |
| `--glossbert-model` | `jvomiranda/GlossBERT_Checkpoint` | Hugging Face Hub model ID |
| `--lang-model` | `en_core_web_sm` | spaCy language model |
| `--sentence-embedder` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `--top-n` | `10` | Informative sentences kept per Phase 1 cluster |
| `--density-percentile` | `75` | Lexical-density percentile threshold |
| `--rates` | *(prompted)* | Comma-separated condensation rate(s), e.g. `10,20` |
| `--ollama-model` | *(prompted)* | Ollama model name for condensation |
| `--adversarial-model` | `--ollama-model`'s value | Ollama model for the post-generation adversarial reviewer (audits/fixes the accepted condensation before it's saved) -- optional even when `--rates`/`--ollama-model`/`--max-trials` are all given |
| `--max-trials` | *(prompted)* | Max generation trials per rate before escalation |

`--rates`/`--ollama-model`/`--max-trials` are a set: give **all three**
to skip the interactive condensation-setup prompt entirely; omit any of
them and you'll be prompted for all three interactively (plus the
adversarial-reviewer model), exactly like `phase2.ipynb`'s
condensation-config cell. Either way the choice is logged to
`data/<corpus>/decisions/phase2_decisions.jsonl` (with `extra.source` set
to `"cli"` or the interactive prompt's own record, respectively).

## Walkthroughs

### Minimal run (fully interactive)

```bash
python run_pipeline.py --corpus Boisseau --input path/to/Boisseau.txt
```

Runs Phase 1 and Phase 2 with default tuning; you'll be prompted at every
review point (see "Interactive prompts" below), including for
condensation setup if you want one.

### Resuming an existing corpus at Phase 2 or 3

```bash
python run_pipeline.py --list-corpora
python run_pipeline.py --corpus Boisseau --start-phase 2   # redo Phase 2 (+3) with new condensation settings
python run_pipeline.py --corpus Boisseau --start-phase 3   # just rebuild the Phase 3 distant-reading report
```

No `--input`/`--clean` for either -- the corpus's already-saved raw text
and Phase 1 state are reused as-is. `--start-phase 2` still prompts for
Phase 2's own config (or reads `--top-n`/`--rates`/etc. if given) and then
runs Phase 2 through Phase 3 normally; `--start-phase 3` only rebuilds
the distant-reading report from whichever condensed rate(s) already exist
on disk -- it does not regenerate condensations. Both abort immediately,
before doing anything, if a required file from an earlier phase is
missing (`--list-corpora` shows exactly which phases each corpus already
satisfies, so you don't have to guess).

### With Phase 0 cleaning

```bash
python run_pipeline.py --corpus Boisseau --input path/to/raw_Boisseau.txt --clean
```

Runs the raw text through `phase0/clean_corpus.py::clean_text` first
(page numbers, headers/footers, footnote digits, front matter, end
sections removed), then proceeds as above using the cleaned text.

### Fully non-interactive condensation

```bash
python run_pipeline.py --corpus Boisseau --input path/to/Boisseau.txt \
  --rates 10,20 --ollama-model llama3 --max-trials 3
```

Skips the condensation-setup prompt (rates/model/trials are already
given). Phase 1's review prompts (flagged terms, clusters) still happen
interactively -- there's currently no flag to skip those; use the
[web interface](README.md#web-interface) if you want
those decisions in a form instead of a terminal, or the notebooks
directly if you want to script around them.

### Tuning Phase 1 for a larger or noisier corpus

```bash
python run_pipeline.py --corpus BigCorpus --input path/to/big.txt \
  --max-stems 300 --min-clusters 8 --max-cluster-size 15 \
  --merge-duplicate-word-occurrences
```

More stems considered, a higher HDBSCAN `min_cluster_size` (fewer,
larger initial clusters), a higher size threshold before a cluster gets
automatically re-split, and merging a word's occurrences before WSD
scoring so it can't be flagged for review more than once.

### Previewing stem-percentile candidates first

```bash
python run_pipeline.py --corpus Boisseau --input path/to/Boisseau.txt --sweep
```

Before extracting stems, prints a comparison table across several
candidate `--frequency-percentile` values (each candidate's true, uncapped
stem count, distinct derived-word count, sentence coverage, and
informative-sentence counts at a few density-percentile bands -- reusing
Phase 2's own scoring, not an approximation) and prompts for a pick or a
custom value. `--max-stems` does **not** cap the counts shown --
`frequency_percentile` is a statistical percentile cutoff on the
frequency-ranked stem vocabulary, so a *larger* value is a *stricter* bar
and stem counts *fall* as it rises (e.g. 0.75 keeps only the top 25%
most-frequent stems); the table shows each candidate's true size first,
and `max_stems` is applied afterward, only to whichever value you pick.
If `--word-frequency-percentile` is also set, the distinct-word and
sentence-coverage counts already reflect that derived-word filter, not
the unfiltered corpus. Cheap: no GlossBERT/clustering runs during the
sweep itself.

## Interactive prompts

In order, when they appear:

1. **Parameter sweep** (Phase 1, only if `--sweep` was given) -- a table
   of candidate `--frequency-percentile` values (each one's true, uncapped
   stem count -- larger `frequency_percentile` = fewer stems, since it's a
   percentile cutoff on the ranked vocabulary, matching the everyday sense
   of "90th percentile" meaning "top 10%" -- distinct derived-word count
   (already reflecting `--word-frequency-percentile` if set), sentences
   containing a top-stem word, and informative-sentence counts at a few
   lexical-density percentile bands); pick a candidate number or type a
   custom `frequency_percentile`/`max_stems`.
2. **Flagged term review** (Phase 1, after GlossBERT analysis) --
   accept all predicted definitions, review one by one, select specific
   terms, or delete a word from all following steps entirely. Only
   appears if GlossBERT flagged any sense mismatches.
3. **Cluster review** (Phase 1, after clustering + n-gram mining) --
   per cluster: accept as-is, or rename/remove stems/remove n-grams.
4. **Condensation setup** (Phase 2) -- only if `--rates`/`--ollama-model`/
   `--max-trials` weren't all given on the command line: rate(s), Ollama
   model, adversarial-reviewer model (ENTER to reuse the generation
   model), max trials.
5. **Escalation** (Phase 2, per rate, only if trials miss the target) --
   retry with the full source text included in the prompt?
6. **Borderline classification review** (Phase 2, only if any F/T spans
   look denser than their own definition allows) -- keep the
   classification or reassign it (F/T/R/C).
7. **Collocation-pair selection** (Phase 3, only if at least one
   condensation rate was generated and a real source collocation was
   found) -- pick one or more ranked, confound-flagged term pairs to
   compare between the source and each summary. Press ENTER for the
   top-ranked pair.
8. **Regenerate a condensation** (Phase 2, after every requested rate has
   been generated, its reports built, and Phase 3 has run) -- would you
   like to regenerate one? If yes: which rate (an existing one is
   overwritten in place, a new one is added alongside the rest), whether
   to adjust the target rate, and whether to add information for the LLM
   by editing the default prompt (opens it in a scratch file in your
   editor). Loops back to "regenerate another?" until you decline. Note:
   this does not re-run Phase 3, so the distant reading report reflects
   the rates approved at the time it ran, not any later regeneration.

Between generation and verification, every rate's accepted condensation
also passes automatically (no prompt) through an **adversarial review**:
a second LLM pass that checks for an abrupt ending, garbled text, or
drift outside the informative-sentence pool, and fixes it if needed --
see README.md's "Phase 2 condensation & injection review" section.

Every choice at every one of these points is logged to
`data/<corpus>/decisions/phase1_decisions.jsonl`, `phase2_decisions.jsonl`,
or `phase3_decisions.jsonl` -- see README.md's "Interactive review &
decision log" section for the record schema.

## Output

Same layout as running the notebooks (plus Phase 3, which has no
notebook) -- see README.md's "Data directory contract". In short,
everything lands under `data/<corpus>/`: `phase1/` (state JSON, cluster
HTML, top stems), `phase2/` (informative sentences JSON, and
`condensation/` with the condensed text, injection report, HTML
fragment/preview, human report, plain summary, and standalone report --
per rate), `phase3/` (the single distant-reading report HTML), and
`decisions/` (the three JSONL decision logs).

## Troubleshooting

- **"Ollama does not appear to be running"** -- install
  [Ollama](https://ollama.com), run `ollama pull <model>`, and make sure
  `ollama serve` is running (or that the Ollama desktop app is open)
  before answering the condensation-setup prompt.
- **No GPU / CUDA available** -- Phase 1 falls back to CPU automatically
  (prints `Using device: cpu`), just slower. GlossBERT WSD and HDBSCAN
  reclustering are the steps most affected.
- **Condensation generation times out** -- large "thinking"-capable
  local models can take well over Ollama's default timeout to respond.
  `phase2/condense.py::attempt_condensation_trials` accepts a `timeout`
  parameter (seconds, default 900) if you're scripting around
  `run_pipeline.py` directly; from the CLI, a smaller/faster model is
  the simplest fix.
- **GlossBERT download is slow/fails** -- it's fetched from the Hugging
  Face Hub (`jvomiranda/GlossBERT_Checkpoint`, ~420MB) on first run and
  cached afterward; make sure you have network access and enough disk
  space for the cache.
- **Adversarial review reports "reviewer call failed" / never fixes
  anything** -- it fails safe by design: an unreachable Ollama, a
  response that doesn't follow the expected `VERDICT: OK`/`VERDICT:
  FIXED` format, or a fixed text that's empty/implausibly short all just
  keep the original condensation and log why (`phase2_decisions.jsonl`,
  `decision_type="adversarial_review_result"`) rather than blocking the
  run or substituting bad text.
