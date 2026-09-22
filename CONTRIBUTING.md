# Contributing to PEEL-Local

## The one rule that is not negotiable: the behaviour freeze

The tag `v1.0.0` marks the commit whose outputs the accompanying paper
reports. Everything under `data/` at that tag *is* the paper's evidence.

**A change that moves a byte under `data/` is a failed change, not a new
baseline.** `tests/golden/data.sha256` holds a hash of all 51 committed
artifacts and `tests/test_tier0_golden.py` checks every one of them. If that
test fails, revert — do not regenerate the manifest.

This is stricter than ordinary regression testing, and it expires: once the
paper is published, the entries in [LIMITATIONS.md](LIMITATIONS.md) become
fixable. Until then they are documented, pinned by tests, and left alone. The
tests exist so that those fixes can later be judged by the exact diff they
produce in the reported numbers.

Work that *is* welcome before publication: tests, documentation, packaging,
performance work that provably changes no output, and anything under
`L11` in LIMITATIONS.md (the only known defect whose fix touches no pipeline
output).

## Setting up

Python **3.10 or newer** is required — the code uses PEP 604 unions
(`str | None`) in evaluated annotations.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Phase 2's condensation additionally needs [Ollama](https://ollama.com)
installed and running, with at least one model pulled. Nothing in the test
suite requires it.

**If `pip install` fails, it is almost certainly `hdbscan`.** The standalone
`hdbscan` package needs a C extension built against your NumPy, and has a
history of breaking across NumPy major versions. It is the single most
fragile dependency here. Note that scikit-learn now ships its own
`sklearn.cluster.HDBSCAN`, but switching to it would change clustering output,
so it is not an option under the freeze.

## Running the tests

```bash
make check          # what CI runs, and what to run before opening a PR
pytest              # same as `pytest -m "not slow"`, per pytest.ini
pytest -m slow      # the tier that needs a GPU, model downloads, or Ollama
pytest -m ""        # everything
```

### The tiers, and what a failure in each one means

Tests are tiered by what they need installed, so the cheap ones can run
everywhere and first.

| Tier | Needs | A failure means |
|---|---|---|
| **0** | stdlib only | a real regression, always |
| **1** | `+ nltk` (PorterStemmer, WordNet) | a real regression, always |
| **2** | `+ spaCy` and `en_core_web_sm` | usually a real regression — but see below |
| **3** | `+ GPU / model downloads / Ollama` | marked `slow`, deselected by default |

**Tier 2 has one important caveat.** Its golden tests assert things that
depend on spaCy's tokeniser, lemmatiser and NER: the 443-item stopword set,
the 12 auto-excluded author names, the F/T/R/C span distribution, the
regenerated `informative_sentences.json`. Upgrade spaCy or its model and these
can move *without any code change*.

That is a feature, not a nuisance. It is the reproducibility signal the paper
argues for, made mechanical: it tells you the environment drifted. When a
Tier 2 test fails after a dependency bump, the question to ask is "what
changed in the environment", not "what did I break in the code" — and the
answer belongs in the commit message, because it means the pinned versions in
`requirements.txt` are load-bearing for reproducing the paper's numbers.

**But check the other explanation first, because it caught us.** The committed
injection reports are *post-review* artifacts: they carry the researcher's
borderline reclassifications, which `run_injection_review` applies by setting
a span's `type` in place and leaving everything else — `basis` included —
untouched. So a span reclassified from T to F still records
`basis: "phrase_match"`, a combination the classifier itself never produces.

Comparing `classify_condensation`'s output straight against a committed report
therefore "fails" on five spans across the two qwen9b configurations, and it
looks exactly like environment drift. It is not. The Phase 2 decision log
records all five (`borderline_reclassification_batch`), and
`test_committed_spans_are_the_heuristic_plus_the_logged_decisions` replays
them and gets byte-equal types — which is the architecture's central claim,
tested. Before concluding that spaCy moved, replay the log.

### Golden tests, and where their expected values come from

Golden tests do not hard-code numbers someone typed in. They regenerate an
artifact by calling the real function and compare it against the committed
one — so the committed artifact is the specification. A few also assert the
literal figures printed in the paper (41.6% / 35.1% / 54.7% verbatim overlap;
the `expectation` cluster at 2.9% → 0.0%), on purpose: a change that moved the
code and the artifact together in step would pass the first kind of test and
fail the second.

## Style

Match the surrounding code. Two conventions here are stronger than usual and
worth stating:

- **Comments explain *why*, and cite the case that motivated them.** This
  codebase is unusually good at this — see `phase1/pipeline.py`'s notes on
  GlossBERT window truncation, or `rename_clusters`'s record of the six
  clusters a name collision used to lose. When you fix something subtle, say
  what it did before and on what input you observed it.
- **Never silently discard a result.** Where a check cannot confirm something,
  the pipeline keeps the output and flags it — `soft_accept`, `sanity_failed`,
  `find_oversized_clusters`, `describe_wordnet_excluded_stems` are all
  instances. Follow that: return a note the caller prints, rather than
  swallowing the case.

Run `ruff check --select=F,E9 .` — `make check` includes it.

## Opening a pull request

1. Run `make check` locally first. CI is a confirmation, not a discovery tool.
2. Reference the LIMITATIONS.md entry (`L1`, `L2`, …) or the issue your change
   addresses.
3. If your change touches anything that could affect an output, say in the PR
   description which golden tests cover it and paste the passing run.
