# Provenance — `Boisseau-qwen2b-25cond-20informativesents`

## The source text is a third-party work

`raw/Boisseau-qwen2b-25cond-20informativesents.txt` is the full text of:

> Boisseau, É. (2026). Expertise, opacity, and trust in AI systems.
> *Synthese* **207**(3):104. <https://doi.org/10.1007/s11229-026-05484-2>

| | |
|---|---|
| **Author** | Éloïse Boisseau |
| **Publisher** | Springer Nature (*Synthese*) |
| **DOI** | [10.1007/s11229-026-05484-2](https://doi.org/10.1007/s11229-026-05484-2) |
| **Source licence** | Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) |
| **Retrieved** | Not recorded at the time of the run. The pipeline logged no acquisition date; this file states that rather than reconstructing one. |
| **Redistribution permission** | Requested from the author and publisher; not yet granted. Tracked in [#1](https://github.com/PUC-Behring-AI/peel-local/issues/1). |

**Neither of this repository's own licences covers this text.** The Apache-2.0
licence in [`../../LICENSE`](../../LICENSE) covers the source code; the CC-BY-4.0
licence in [`../../LICENSE-DOCS`](../../LICENSE-DOCS) covers this repository's own
documentation, figures, and generated reports. The passages of Boisseau (2026)
reproduced in this directory remain under CC BY-NC-ND 4.0 and are attributed
above. If you reuse anything from this directory, attribute the source work and
observe its terms, not this repository's.

## What Phase 0 removed, and why the copyright line is absent

The committed text is **not** the publisher's PDF as extracted. It was processed by
`phase0/clean_corpus.py`, which removed page numbers, running headers and footers,
footnote-call digits, and front matter. Its `FRONT_MATTER_PATTERNS` match `^©`,
`under exclusive licence`, and `Springer Nature`, so the article's own copyright and
licence lines were stripped along with the masthead. A search of this directory for
`springer`, `©`, `copyright`, `licence`, or `doi.org` returns nothing.

That is why this file exists. It restores, as metadata, the attribution the cleaning
step removed from the text — and it is the reason the CC BY-NC-ND **ND** term is at
issue: the committed text is a modified version of the published work, and the
condensations below are derivative works of it.

## How much of the source is in this directory

Deleting `raw/*.txt` alone would not remove the source text, because the pipeline
embeds source passages in its derived artifacts by design — that is what makes the
condensation auditable:

| Artifact | Source text it contains |
|---|---|
| `raw/…txt` | the whole work: 83,164 characters / 13,826 words |
| `phase3/…-distant-reading-report.html` | 20,155 characters, in the *Full text (highlighted)* section (capped at 20,000 by `build_reader_html`) |
| `phase2/informative_sentences.json` | 154 source sentences / 39,228 characters |
| `phase2/condensation/…-fragment.html` and `…-preview.html` | the source sentences cited by 31 classified R/C spans, in the "Show source" toggles |
| `phase2/condensation/…-injection-report.json` | the same span attributions, by sentence index |

Roughly 59,000 of the work's 83,164 characters persist outside `raw/`.

## What is this repository's own work

The condensation, all verification statistics, the cluster definitions, the decision
logs, and every generated report are PEEL-Local's output, licensed CC-BY-4.0 — with
the caveat that they quote the source work as tabulated above.

This directory is one of three Phase 2 configurations reported in the paper's worked
example (§3.2). All three share one Phase 0/Phase 1 run and differ only in the
condensation model and the number of informative sentences retained per cluster:

| Configuration | Ollama model | Informative sentences per cluster | Condensation kept |
|---|---|---|---|
| **this one** | `qwen3.5:2b` | 20 | 747 words (5.4% of source) |
| `Boisseau-qwen9b-25cond-10informativesents` | `qwen3.5:9b` | 10 | 3,062 words |
| `Boisseau-qwen9b-25cond-20informativesents` | `qwen3.5:9b` | 20 | 3,079 words |

## Reproducing this without the committed text

Obtain the article from its DOI, then:

```bash
python run_pipeline.py --corpus MyBoisseau --input path/to/boisseau-2026.pdf --clean
```

Phase 0 will produce a cleaned text from your own copy. Note that it will not be
byte-identical to the committed one unless your PDF extraction matches — see the
reproducibility note in [`../../README.md`](../../README.md), and the environment
caveat: the committed artifacts predate this repository's dependency pinning.
