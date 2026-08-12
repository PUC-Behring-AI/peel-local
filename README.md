# peel-local

PEEL-Local is a computationally augmented interpretive inquiry framework for
running phases derived from *Protocols for Epistemically Engaged Literacy in
AI (PEEL)* locally, with complementary AI and data analysis tools.

PEEL-Local, stemming from PEEL, aims at not automating interpretation.
Instead, it provides computational support for:

- semantic inspection
- lexical organization
- clustering
- structured interpretive analysis

Interpretation remains researcher-driven.

For a function-by-function map of the codebase (what's implemented where,
and what calls what), see [FUNCTIONS.md](FUNCTIONS.md).

---

## Pipeline overview

The pipeline runs in four phases, each corpus-specific and each reading
its input from the previous phase's output (see
[Data directory contract](#data-directory-contract) below):

1. **Phase 0 -- Cleaning** (`phase0/phase0.ipynb`): removes page numbers,
   running headers/footers, footnote-call digits, broken hyphenation,
   front matter (mastheads, copyright notices, abstracts), and end
   sections (notes/references) from a raw `.txt` corpus.
2. **Phase 1 -- Semantic analysis** (`phase1/phase1.ipynb`): extracts
   frequent word stems, maps each occurrence to its word, stem, part of
   speech, and sentence, retrieves WordNet synsets, predicts the most
   likely sense with GlossBERT (Word Sense Disambiguation), builds
   Sentence-BERT embeddings from the accepted definitions, and clusters
   them with HDBSCAN into semantic groups. Flagged sense mismatches and
   the resulting clusters go through optional manual review, and the
   outcome -- plus every decision made along the way -- is exported to
   JSON and HTML.
3. **Phase 2 -- Informative sentence selection & condensation**
   (`phase2/phase2.ipynb`): for each Phase 1 cluster, scores every
   sentence in the corpus by lexical density and how many of the
   cluster's stems/n-grams it contains, keeping the top-scoring
   sentences as representative examples. Optionally, it then prompts a
   locally running LLM (via [Ollama](https://ollama.com)) to condense the
   corpus to a chosen target rate, and verifies that condensation --
   classifying every non-verbatim span by injection risk, scanning for
   verbatim overlap with the source, and checking per-cluster term
   coverage -- before exporting a standalone HTML report. See
   [Phase 2 condensation & injection review](#phase-2-condensation--injection-review)
   below.
4. **Phase 3 -- Distant reading & source-vs-summary comparison**
   (`phase3/pipeline.py`, run automatically as part of `run_pipeline.py`
   or the web interface -- no notebook of its own): a self-contained,
   Python-native reimplementation of the "distant reading" analyses a
   separate, Voyant-based PEEL system once relied on an external hosted
   tool for. Builds a word cloud, a per-cluster prevalence chart, a
   recurring-phrase table, and a term frequency/distribution-shape table
   over the source alone, plus -- for every approved Phase 2 condensation
   rate -- a document-profile table, cluster-coverage comparison, word
   clouds, and keyword-in-context concordances comparing the source
   against each summary for a researcher-confirmed collocation pair
   (found by actually scanning the source for terms that co-occur, never
   guessed). Everything is written to one standalone HTML file; nothing
   is uploaded anywhere. See
   [Phase 3 distant reading](#phase-3-distant-reading) below.

Some stems may not exist in WordNet, and clusters HDBSCAN considers noise
are automatically re-clustered before being discarded. GPU execution is
recommended for Phase 1 (GlossBERT + embeddings); Phase 0, Phase 2, and
Phase 3 run fine on CPU.

---

## Technologies

### NLP
- [spaCy](https://spacy.io)
- [NLTK](https://www.nltk.org)
- [WordNet](https://wordnet.princeton.edu)

### Word Sense Disambiguation
- [GlossBERT](https://github.com/HSLCY/GlossBERT)

### Semantic Embeddings
- [Sentence-Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

### Clustering
- [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/)

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

which covers: `torch`, `transformers`, `sentence-transformers` (Phase 1
WSD/embeddings), `spacy`, `nltk` (core NLP), `hdbscan` (clustering),
`scipy`, `wordcloud` (Phase 3 distant reading), `numpy`, `tqdm`,
`requests` (general), and `flask` (the [web interface](#web-interface)).

Phase 2's condensation step also requires [Ollama](https://ollama.com)
installed separately (not a pip package) -- see
[Phase 2 condensation & injection review](#phase-2-condensation--injection-review).

Download the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

Download WordNet resources:

```python
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")
```

`huggingface_hub` (used to fetch GlossBERT, see below) ships transitively
with `transformers`, so no separate install step is needed for it.

---

## GlossBERT model

GlossBERT is fetched automatically from the Hugging Face Hub
(`jvomiranda/GlossBERT_Checkpoint`) the first time `phase1.ipynb` runs, and
cached locally afterwards -- no manual download or local placement is
required. Expect a one-time ~420MB download on first run; a network
connection is required for it.

If you previously kept a local `GlossBERT_Checkpoint/` folder for offline
use, it's no longer read by the pipeline and can be deleted to reclaim
disk space.

---

## CUDA

CUDA is required for efficient GPU processing during Phase 1. See:
[CUDA INSTALLATION](https://developer.nvidia.com/cuda-downloads)

`requirements.txt` installs a CUDA 13.0-enabled `torch` build by default
(via `--extra-index-url https://download.pytorch.org/whl/cu130`; every
other package still resolves from PyPI as usual). If your GPU/driver
needs a different CUDA version, edit that line to match the build listed
at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/),
or delete it and run plain `pip install torch` for CPU-only -- Phase 1
falls back to CPU automatically either way, just slower.

---

## Data directory contract

Every notebook derives all of its input/output paths from a single
`CORPUS_NAME` variable set in its config cell, via `common/paths.py`'s
`CorpusPaths`. This is the only thing you edit per notebook to switch
corpus -- there is no manual copying of files between phase folders.

```
data/<CORPUS_NAME>/
├── raw/<CORPUS_NAME>.txt          # input: raw or phase0-cleaned corpus text
├── phase1/
│   ├── top_stems.txt
│   ├── glossbert_accepted_terms.txt
│   ├── <CORPUS_NAME>-phase1_state.json
│   └── <CORPUS_NAME>-Phase1-clusters.html
├── phase2/
│   ├── informative_sentences.json
│   └── condensation/               # per requested rate, see below
│       ├── <CORPUS_NAME>-condensed-<rate>pct.txt
│       ├── <CORPUS_NAME>-condensed-<rate>pct-injection-report.json
│       ├── <CORPUS_NAME>-condensed-<rate>pct-fragment.html
│       ├── <CORPUS_NAME>-condensed-<rate>pct-preview.html
│       ├── <CORPUS_NAME>-condensed-<rate>pct-report.txt
│       ├── <CORPUS_NAME>-condensed-<rate>pct-summary.txt
│       └── <CORPUS_NAME>-standalone-report-<rate>pct.html
├── phase3/
│   └── <CORPUS_NAME>-distant-reading-report.html
└── decisions/
    ├── phase1_decisions.jsonl      # see "Interactive review & decision log" below
    ├── phase2_decisions.jsonl
    └── phase3_decisions.jsonl
```

`data/` is git-ignored (generated/corpus-specific content shouldn't pile
up in version control), with one deliberate exception: `data/Boisseau/phase1/*`
is committed as a worked example, carried over from this repo's earlier
`phase1/results/` folder.

---

## Running the pipeline

### Notebook by notebook

Example using the `Boisseau` corpus already included under
`data/Boisseau/raw/Boisseau.txt`:

1. **Phase 0** (optional cleaning step): run as a script (also importable
   as `phase0.clean_corpus`, used by `phase0.ipynb` and `run_pipeline.py`):
   ```bash
   python phase0/clean_corpus.py raw_corpus.txt -o cleaned_corpus.txt
   ```
   Place the cleaned output at `data/<CORPUS_NAME>/raw/<CORPUS_NAME>.txt`
   if you want Phase 1 to analyze the cleaned version.
2. **Phase 1**: open `phase1/phase1.ipynb`, set `CORPUS_NAME = "Boisseau"`
   in the config cell, and run all cells top to bottom. You'll be
   prompted to review GlossBERT's flagged sense mismatches and the
   resulting clusters along the way.
3. **Phase 2**: open `phase2/phase2.ipynb`, set the same
   `CORPUS_NAME = "Boisseau"`, and run all cells. It reads Phase 1's
   state JSON directly from `data/Boisseau/phase1/`. The first four cells
   select informative sentences per cluster; the remaining cells (optional)
   generate and verify an LLM condensation -- see the next section.

To analyze a new corpus, place its raw text at
`data/<CORPUS_NAME>/raw/<CORPUS_NAME>.txt` and set `CORPUS_NAME`
accordingly in both notebooks. Phase 3 has no notebook of its own -- it
only runs as part of `run_pipeline.py` or the web interface, immediately
after Phase 2's initial condensation rates (if any) are built. See
[Phase 3 distant reading](#phase-3-distant-reading) below.

### All at once: `run_pipeline.py`

For a new corpus, `run_pipeline.py` at the repo root runs Phase 0
(optional) -> Phase 1 -> Phase 2 -> Phase 3 -> standalone report export
in a single command, with the exact same interactive prompts, outputs,
and decision logs as running the two notebooks by hand plus Phase 3's
own automatic step (it calls the same
`pipeline.py`/`condense.py`/`condensation_report.py`/`standalone_report.py`
functions -- nothing is reimplemented):

```bash
python run_pipeline.py --corpus Boisseau --input path/to/raw.txt
python run_pipeline.py --corpus Boisseau --input path/to/raw.txt --clean
python run_pipeline.py --corpus Boisseau --input path/to/raw.txt --rates 10,20 --ollama-model llama3 --max-trials 3
```

`--clean` runs Phase 0 first. Every Phase 1/Phase 2 config constant
(percentiles, stem/cluster limits, model names, ...) is available as a
flag with the same default as the notebooks -- run `--help` to see all of
them. `--rates`/`--ollama-model`/`--max-trials` are optional: omit any of
them to be prompted interactively for condensation setup, same as
`phase2.ipynb`. See [RUN_PIPELINE_GUIDE.md](RUN_PIPELINE_GUIDE.md) for a
full flag reference, walkthroughs, and troubleshooting.

### In a browser: the web interface

If you'd rather not use a terminal at all, the [web interface](#web-interface)
below wraps this same flow -- file upload, every config field, and every
interactive decision point as a form.

---

## Phase 2 condensation & injection review

After selecting informative sentences, Phase 2 can optionally prompt a
locally running Ollama model to condense the corpus to a target word
count, then verify that condensation. This requires
[Ollama](https://ollama.com) installed, a model pulled
(`ollama pull <model>`, e.g. `llama3`), and `ollama serve` running --
the config cell checks for it and lists installed models.

**Generation**: you're prompted for one or more target rates (5-30% of
the source word count), an Ollama model name, and a maximum number of
generation trials. Generation always runs with `repeat_penalty: 1.0`
(Ollama's neutral value), so no implicit anti-repetition bias from the
model's own Modelfile suppresses legitimate reuse of source/cluster key
terms. Each trial is checked against both the target word count (±5%
tolerance) **and** a set of generation sanity checks (below); if every
trial misses tolerance, you're asked whether to retry with the full
source text included in the prompt (not just the informative sentences
and cluster terms).

**Generation sanity checks.** Three cheap, deterministic checks run on
every trial before it's accepted, catching visible generation
degeneration without needing another LLM call to judge it: (1) a run of
consecutive words, at least as long as the condensation's own mean
words-per-sentence, that are absent from the informative sentences
actually used in the prompt and aren't ordinary connectives -- topic
drift/hallucinated vocabulary; (2) a single word over 30 characters --
runaway token concatenation; (3) an exact 6-word phrase repeated two or
more times anywhere in the text -- a repetition loop. A trial only counts
as an outright success if it clears tolerance *and* all three checks; if
every trial fails the checks, the closest-by-word-count trial is still
used (never silently discarded) but clearly flagged as unreviewed in the
console output, decision log, and human-readable report.

**Verification**, per rate:
- **Injection taxonomy** -- every non-verbatim sentence in the
  condensation is classified by how it relates to the source:

  | Code | Name | Meaning | Risk |
  |---|---|---|---|
  | F | Framing | Whole sentence otherwise matches a source sentence; the only difference is <=4 inserted/changed tokens, all purely connective (spaCy POS/dependency-based) -- or, if nothing in the source matches at all, a freestanding connective fragment on its own | Low |
  | T | Transition | Metalinguistic sentence about the text's own argument/structure, or a whole sentence that otherwise matches a source sentence with <=6 inserted/changed tokens that include genuine (non-connective) content | Medium |
  | R | Reformulation | Paraphrases a single identifiable source sentence (only reached once F/T are ruled out) | Medium-high |
  | C | Compression | Content drawn from/merged across multiple source sentences -- every source sentence sharing >=15% content-word overlap is cited (not just the closest one), shown 5 at a time with a "Show N more" toggle for the rest | High |

  Classification here is **heuristic, not LLM-judged**. For every
  non-verbatim sentence, the pipeline first finds its best-matching source
  sentence (content-word overlap), then runs an actual sequence diff
  (`difflib`) between the two -- F and T are decided from what that diff
  actually inserted or changed, not from the sentence's raw length: all
  connective (spaCy POS/dependency tags -- coordinating/subordinating
  conjunctions plus adverbial-modifier discourse connectives like
  "however"/"therefore") means F; a small amount of genuine lexical
  content (nouns, verbs, adjectives, pronouns, adverbs, numbers, wh-words)
  means T. R/C still come from the same word-count and keyword-based
  overlap rule as before, not a semantic reading, and are only reached
  once F/T are ruled out. It can misclassify ambiguous spans. A
  borderline-flag pass surfaces F/T spans that look denser than their own
  definition allows, and you're asked whether to keep or reclassify each
  one.
- **Verbatim-overlap scan** -- an independent, mechanical token-alignment
  check against the source, reported alongside (not instead of) the
  word share outside any classified span, since the two measure
  different things.
- **Cluster coverage** -- per Phase 1 cluster, what share of all
  cluster-vocabulary token occurrences belong to it in the source (the
  target -- the source's own relative emphasis) vs. in the condensation
  (actual), with OK (within +/-5pp of the source's own share), WARN
  (beyond that), or DARK (a cluster with real presence in the source gets
  zero mentions in the condensation) status.

**Source title/author(s)/date.** Before generation, you're asked for the
source document's title, author(s), and publication date, for the report
header -- either typed in directly (leave a field blank for "Unclear"),
or left to the model: a small retrieval step embeds the source's
paragraph-level chunks, retrieves the ones most likely to contain this
front-matter information, and asks the model to extract it in a fixed
format, falling back to "Unclear" per field rather than guessing. This
replaced an earlier version that tried to guess the title/author from the
condensation's own first two lines -- unreliable whenever the
condensation didn't happen to open with a literal one-line title (the
usual case, since the condensation prompt never asks for one). Collected
once per corpus run and reused for every rate and any later
regeneration.

**Output**, per rate, under `data/<CORPUS_NAME>/phase2/condensation/`: the
condensed text, an injection-report JSON, an HTML fragment (inline styles
only, no `<style>`/classes), a standalone browser-preview HTML, a
plain-text verification report, a markup-free plain-text summary, and a
**standalone report** (`<CORPUS_NAME>-standalone-report-<rate>pct.html`)
showing the same information -- condensation, Phase 1 cluster results,
and a run summary -- as its own cleanly styled, self-contained page (see
`common/standalone_report.py`).

Every choice made in this flow (rates, model, trial outcomes, escalation,
reclassifications) is logged to `data/<CORPUS_NAME>/decisions/phase2_decisions.jsonl`,
same schema as Phase 1's decision log below.

**Regenerating a condensation.** After all requested rates are generated
and their reports built, both `run_pipeline.py` and the web interface
offer to regenerate one -- as many times as you like, before you'd move
on to a new run. You can optionally target a different rate (redoing an
existing one overwrites its outputs; a new one is added alongside the
rest, nothing else is touched or deleted) and/or edit the exact prompt
sent to the LLM: the full rendered default prompt (informative sentences,
cluster terms, and instructions) is shown, and whatever you submit --
edited or not -- is what gets sent, verbatim, for every trial. In the CLI
this opens the rendered prompt in a scratch file for you to edit in your
own editor; in the web UI it's an inline, editable text box. This is also
how you can give the model additional context or instructions the
default prompt doesn't include.

---

## Phase 3 distant reading

Runs automatically right after Phase 2's initial condensation rates (if
any) are built and their reports saved -- a Python-native, self-contained
reimplementation of the "distant reading" analyses a separate,
Voyant-based PEEL system this repo doesn't use once relied on an
external hosted tool for. No upload, no manual ID round-trips, no
notebook to open: everything lands in one standalone HTML file,
`data/<CORPUS_NAME>/phase3/<CORPUS_NAME>-distant-reading-report.html`.

**Stopwords.** Before anything else, a comprehensive stopword set is
built for the word clouds and frequency tables below: spaCy's built-in
English stopword list, plus numerals (including citation-year variants
like `2024a`), common citation abbreviations (`et`, `al`, `ibid`, ...),
and any name spaCy tags as a person mentioned 3 or more times (an
automatic proxy for "probably a cited author," disclosed in the
report's Provenance section rather than treated as a silent default --
unlike a researcher-confirmed judgment call, this can have false
positives).

**Distant reading (source only):**
- **Word cloud** -- corpus term frequency, stopword-filtered.
- **Cluster prevalence chart** -- each Phase 1 cluster's stem/n-gram hit
  count across 5 sequential segments of the source, as a line chart.
- **Recurring phrases** -- 2-5 word n-grams, frequency-ranked, with
  shorter grams dropped when they're fully contained in a more frequent
  longer one.
- **Term frequency & distribution shape** -- for every cluster-significant
  term: raw/relative frequency, plus *peakedness* (kurtosis) and
  *skewness* (skew) of its per-segment frequency vector -- how
  concentrated vs. evenly spread, and front-loaded vs. back-loaded, each
  term's usage is across the source.
- **Full text, highlighted** -- every occurrence of a cluster stem/n-gram
  marked and colour-coded by cluster (the same token-level stem/lemma
  matching used elsewhere in this pipeline, not a regex approximation).

**Collocation-pair selection (interactive, only if at least one
condensation rate was approved).** The source text is scanned for pairs
of significant terms, drawn from different clusters, that actually
co-occur within 5 token positions of each other -- ranked by real hit
count, and flagged when one term's own frequency is disproportionate (a
possible base-rate confound, not a real relationship: a term that
appears in nearly every sentence will co-occur with almost anything).
You pick one or more pairs from this ranked, flagged list -- never an
automatic top-score pick, since the confound flag is a heuristic, not a
verdict. If no real collocation is found among the cluster-significant
terms, this step -- and the Contexts/Collocates comparison below -- is
skipped.

**Source vs. Summary (only if at least one condensation rate was
approved):**
- **Document profile** -- word count, unique-word count, and lexical
  density for the source and every approved summary, side by side.
- **Cluster coverage** -- reuses `condense.py`'s own
  `compute_cluster_coverage()` (the same OK/WARN/DARK comparison Phase 2's
  own condensation report already shows per rate), so the source-vs-summary
  question isn't answered twice by two different mechanisms.
- **Word clouds** -- one per document (source, then each summary).
- **Contexts** -- a keyword-in-context concordance for each selected
  collocation pair, run separately per document, so you can see whether
  and how a real source collocation survives into each summary.
- **Collocates** -- a co-occurrence table (term, count, anchor-term
  occurrences) for the first selected pair's first term, per document.

**A disclosed limitation, matching the same choice `run_pipeline.py`'s
condensation regeneration already makes for its own reports:** Phase 3
runs once, using whatever rates are approved at that point. A later
post-completion condensation regeneration does not retroactively rebuild
this report.

Every choice (the collocation-pair selection) is logged to
`data/<CORPUS_NAME>/decisions/phase3_decisions.jsonl`, same schema as
Phase 1/Phase 2's decision logs.

---

## Web interface

`webapp/` wraps `run_pipeline.py`'s exact same flow -- Phase 0 (optional)
-> Phase 1 -> Phase 2 -> Phase 3 -> standalone report export -- behind a
browser UI: a file-upload form with every config field from the CLI
(percentiles, stem/cluster limits, model names, ...), and every
interactive decision point (flagged-term review, cluster review,
condensation setup, escalation, borderline-classification review,
collocation-pair selection) as a form instead of a terminal prompt --
dropdowns, checkboxes, and Yes/No buttons. It calls the exact same
`phase1/pipeline.py`/`phase2/pipeline.py`/`phase2/condense.py`/
`phase2/condensation_report.py`/`phase3/pipeline.py`/
`common/standalone_report.py` functions the notebooks and
`run_pipeline.py` call -- nothing about the pipeline itself is
reimplemented for the web UI.

### Starting it

```bash
pip install -r requirements.txt   # includes flask
python webapp/app.py
```

Then open **http://localhost:5000**.

This is a local, single-researcher tool, not a multi-user server: it
runs **one pipeline run at a time** -- starting a new run from the setup
screen replaces whatever the previous one was doing.

### How it works

Phase 1 (GlossBERT, embeddings, HDBSCAN reclustering) and Phase 2 (Ollama
generation) are slow -- the same operations `run_pipeline.py` runs
synchronously in a terminal can take minutes. The web UI runs each
computational chunk in a background thread and the page polls for
updates every ~1.5s, showing a **live log panel** (the step's actual
`print()` output -- progress bars, "Loading GlossBERT checkpoint...",
trial results, etc.) rather than a bare spinner. When a chunk finishes
and needs a decision, the page shows the corresponding form; submitting
it resumes the pipeline in the background.

### Screens, in order

1. **Setup** -- corpus name, `.txt` file upload, optional "clean with
   Phase 0" checkbox, and a collapsible advanced-settings panel with
   every Phase 1 config field (same defaults as the CLI).
2. **Progress console** -- live log panel + current step name.
3. **Flagged-term review** -- one row per flagged term, with the
   sentence, default vs. predicted definition, and a choice control
   (keep default / use predicted / pick a candidate / manual entry);
   plus a "use predicted for all" quick action. All rows submitted together.
4. **Cluster review** -- one card per cluster: editable name, and a
   checkbox per stem and n-gram (uncheck to remove).
5. **Phase 2 setup** -- `top_n`/density-percentile fields, a "generate a
   condensation?" Yes/No toggle, and (if yes) rate(s), an Ollama model
   dropdown (populated live from Ollama, with an availability check),
   max trials, and the source's title/author(s)/date -- three optional
   text fields, or a checkbox to have the model determine them instead.
6. **Escalation** (only if triggered) -- per rate that missed its target
   word count: the trial results and a Yes/No "retry with full source
   text?" button.
7. **Borderline classification review** (only if any exist) -- per
   flagged span: the text, the reason it was flagged, and
   `[Keep] [F] [T] [R] [C]` buttons.
8. **Collocation-pair selection** (Phase 3, only if a condensation was
   generated and a real source collocation was found) -- a table of
   ranked, confound-flagged term pairs; check one or more, then submit.
9. **Complete** -- links to every generated file (Phase 1 outputs, both
   decision logs, the Phase 3 distant reading report, and per-rate
   condensation/standalone-report files), served straight from the
   browser, plus a **"Regenerate a
   condensation"** panel: a rate field (existing rate = overwrite in
   place, new rate = add alongside the rest) and a "Preview / edit
   default prompt" button that reveals the full rendered prompt in an
   editable text box before you submit. Regenerating reuses the same
   progress-console/polling flow as any other step, then returns here
   with the manifest updated. "Start another run" resets the server-side
   session before reloading, so it correctly returns to the setup screen
   instead of bouncing back to this one.

Every decision submitted through the web UI is logged via the same
`common/decisions.py` `DecisionLog` as the notebooks and
`run_pipeline.py` use, tagged `extra.source: "web"` (batched one record
per screen submission, rather than one per individual choice the way the
CLI logs them) -- see "Interactive review & decision log" below for the
record schema.

### Scope

Matches `run_pipeline.py`: a **new corpus, end to end**. There's no
"resume an existing corpus" mode -- to re-run just Phase 2 on a corpus
that already has a `phase1_state.json`, use `phase2/phase2.ipynb` or
`run_pipeline.py` directly instead.

---

## Interactive review & decision log

Phase 1 has two points where you make interpretive calls:

1. **Flagged term review** -- for each stem/word whose GlossBERT-predicted
   sense disagrees with WordNet's default sense, accept the prediction,
   keep the default, enter a manual definition, or review all of them at
   once. Words are matched case-insensitively (a word appearing once in
   an all-caps heading and once in normal prose no longer produces two
   separate review items for the same underlying mismatch), and the
   candidate list respects your configured `max_synsets` (previously
   capped at 3 regardless of that setting).
2. **Cluster review** -- accept, rename, or edit each semantic cluster:
   remove stems or n-grams from it.

The final `<CORPUS_NAME>-phase1_state.json` only records the *outcome* of
these steps. Every individual choice is also appended, as it happens, to
`data/<CORPUS_NAME>/decisions/phase1_decisions.jsonl` via
`common/decisions.py`'s `DecisionLog`, so the reasoning behind a cluster
or definition can be consulted later even if the final state doesn't
show it. Each line is a JSON object:

| field | meaning |
|---|---|
| `timestamp` | ISO-8601 UTC time the decision was recorded |
| `corpus`, `phase` | which corpus/phase this belongs to |
| `step` | review point (`flagged_term_review`, `cluster_review`; Phase 2/3 keep their own logs with the same schema, e.g. `condensation_setup`, `injection_review`, `collocation_review`) |
| `decision_type` | specific sub-decision (e.g. `per_term_choice`, `stem_removal`) |
| `prompt` | the exact prompt text shown |
| `options` | the menu/candidates presented, if any |
| `choice` | the raw input given |
| `extra` | free-form context (word/stem, cluster name, resulting definition, removed items, ...) |

The log is append-only and local (`data/` is git-ignored), so it doesn't
affect the interactive flow and isn't shared unless you choose to.

---

## Repository layout

```
peel-local/
├── README.md
├── FUNCTIONS.md       # function-by-function map of the codebase
├── RUN_PIPELINE_GUIDE.md  # detailed run_pipeline.py CLI guide
├── run_pipeline.py    # Phase 0 (optional) -> 1 -> 2 -> 3 -> standalone report export, one command
├── requirements.txt
├── .gitignore
├── common/            # shared modules: data-directory contract, decision log,
│                       # Ollama client, standalone report
├── phase0/            # phase0.ipynb + clean_corpus.py -- corpus cleaning
├── phase1/            # phase1.ipynb + pipeline.py -- stems, WSD, clustering
├── phase2/            # phase2.ipynb + pipeline.py (sentence selection) +
│                       # condense.py + condensation_report.py (condensation)
├── phase3/            # pipeline.py (orchestration) + distant_reading.py +
│                       # comparison.py + collocations.py + stopwords.py +
│                       # terms.py + report.py -- distant reading, no notebook
├── webapp/            # Flask web interface -- app.py, pipeline_session.py, static/
└── data/              # per-corpus inputs/outputs (git-ignored, see contract above)
```
