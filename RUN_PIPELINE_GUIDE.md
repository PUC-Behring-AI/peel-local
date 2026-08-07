# `run_pipeline.py` usage guide

`run_pipeline.py` (repo root) runs Phase 0 (optional) &rarr; Phase 1 &rarr;
Phase 2 &rarr; Voyant + standalone report export on **one new corpus**, in
a single command. It calls the exact same functions the notebooks call
(`phase1/pipeline.py`, `phase2/pipeline.py`, `phase2/condense.py`,
`phase2/condensation_report.py`, `common/voyant_notebook.py`,
`common/standalone_report.py`) -- interactive prompts, outputs, and
decision logs are identical to running `phase0.ipynb` (optional) +
`phase1.ipynb` + `phase2.ipynb` by hand. See the root
[README.md](README.md) for the pipeline's overall shape and the
[web interface](README.md#web-interface) if you'd rather drive this
through a browser instead of a terminal.

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
| `--corpus` (required) | -- | Corpus name; determines `data/<corpus>/` |
| `--input` (required) | -- | Path to the raw `.txt` file |
| `--clean` | off | Run Phase 0 cleaning (`phase0/clean_corpus.py`) before Phase 1 |
| `--top-percentile` | `0.50` | Fraction of most-frequent stems considered |
| `--max-stems` | `150` | Maximum number of stems to return |
| `--max-sentences-per-stem` | `5` | Sentences collected per stem for WSD |
| `--max-synsets` | `5` | WordNet definitions considered per stem |
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
| `--max-trials` | *(prompted)* | Max generation trials per rate before escalation |

`--rates`/`--ollama-model`/`--max-trials` are a set: give **all three**
to skip the interactive condensation-setup prompt entirely; omit any of
them and you'll be prompted for all three interactively, exactly like
`phase2.ipynb`'s condensation-config cell. Either way the choice is
logged to `data/<corpus>/decisions/phase2_decisions.jsonl` (with
`extra.source` set to `"cli"` or the interactive prompt's own record,
respectively).

## Walkthroughs

### Minimal run (fully interactive)

```bash
python run_pipeline.py --corpus Boisseau --input path/to/Boisseau.txt
```

Runs Phase 1 and Phase 2 with default tuning; you'll be prompted at every
review point (see "Interactive prompts" below), including for
condensation setup if you want one.

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
given). Phase 1's review prompts (flagged terms, clusters, Voyant
settings) still happen interactively -- there's currently no flag to
skip those; use the [web interface](README.md#web-interface) if you want
those decisions in a form instead of a terminal, or the notebooks
directly if you want to script around them.

### Tuning Phase 1 for a larger or noisier corpus

```bash
python run_pipeline.py --corpus BigCorpus --input path/to/big.txt \
  --max-stems 300 --min-clusters 8 --max-cluster-size 15
```

More stems considered, a higher HDBSCAN `min_cluster_size` (fewer,
larger initial clusters), and a higher size threshold before a cluster
gets automatically re-split.

## Interactive prompts

In order, when they appear:

1. **Flagged term review** (Phase 1, after GlossBERT analysis) --
   accept all predicted definitions, review one by one, or select
   specific terms. Only appears if GlossBERT flagged any sense
   mismatches.
2. **Cluster review** (Phase 1, after clustering + n-gram mining) --
   per cluster: accept as-is, or rename/remove stems/remove
   n-grams/exclude removed stems globally.
3. **Voyant settings** (Phase 1, before saving) -- a Voyant corpus ID
   (can be left blank) and whether to apply Voyant's `en_smart`
   stopword list.
4. **Condensation setup** (Phase 2) -- only if `--rates`/`--ollama-model`/
   `--max-trials` weren't all given on the command line: rate(s), Ollama
   model, max trials.
5. **Escalation** (Phase 2, per rate, only if trials miss the target) --
   retry with the full source text included in the prompt?
6. **Borderline classification review** (Phase 2, only if any F/T spans
   look denser than their own definition allows) -- keep the
   classification or reassign it (F/T/R/C).
7. **Regenerate a condensation** (Phase 2, after every requested rate has
   been generated and its reports built) -- would you like to regenerate
   one? If yes: which rate (an existing one is overwritten in place, a
   new one is added alongside the rest), whether to adjust the target
   rate, and whether to add information for the LLM by editing the
   default prompt (opens it in a scratch file in your editor). Loops
   back to "regenerate another?" until you decline.

Every choice at every one of these points is logged to
`data/<corpus>/decisions/phase1_decisions.jsonl` or
`phase2_decisions.jsonl` -- see README.md's "Interactive review &
decision log" section for the record schema.

## Output

Same layout as running the notebooks -- see README.md's "Data directory
contract". In short, everything lands under `data/<corpus>/`:
`phase1/` (state JSON, cluster HTML, top stems), `phase2/`
(informative sentences JSON, and `condensation/` with the condensed
text, injection report, HTML fragment/preview, human report, plain
summary, Voyant notebook, and standalone report -- per rate), and
`decisions/` (the two JSONL decision logs).

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
