# Known limitations

What is known to be wrong with PEEL-Local, measured rather than estimated, so
a reader does not have to rediscover it.

Every entry below is a defect that **changes an output**. That is why none of
them is fixed yet: the artifacts under `data/` are the evidence the paper
reports, and the `v1.0.0-paper` tag freezes them. Fixing any of these moves a
number the paper cites, so each waits until the paper is published, and each
has a characterisation test pinning today's behaviour first (see
[CONTRIBUTING.md](CONTRIBUTING.md)) so its fix lands as a visible diff rather
than a guess.

This file is written to be consumed. An entry leaves when its fix lands.
Entries keep stable IDs (`L1`, `L2`, …) so tracker issues and code comments
can cite them.

The paper's §4.3 states the same limitations at the level of the argument;
this is the implementation-level register of them, plus the ones found by a
post-submission audit of the code against the paper.

---

## L1 — Cluster n-grams are counted by raw substring search

**Where** `phase2/condense.py::cluster_hit_counts`

**What happens** A cluster's n-grams are matched with `text.count(gram)`. But
Phase 1 mines n-grams from a *lemmatised, stopword-filtered* token stream, so
a real phrase generally has no literal occurrence in the text:
`"expertise opacity trust"` was mined from "Expertise, opacity, and trust",
with a comma and an "and" in between.

**Measured** On the committed example corpus, **22 of 33** cluster n-grams
count exactly zero, and **46%** of all n-gram occurrences in the source are
lost. The clusters `action`, `agent` and `trust` lose **100%** of their n-gram
evidence.

**What it affects** Phase 2's cluster-coverage table per rate; the same table
reused for the source-vs-summary comparison (`phase3/comparison.py`); and the
cluster-prevalence chart (`phase3/distant_reading.py::bin_cluster_frequencies`).
The term-frequency table in that *same* Phase 3 report uses the correct
matching (`phase3/terms.py`), so the two disagree inside one page.

**How much it moves the published numbers** Recomputed with correct matching:
the largest shift is **+0.9pp** on one cluster's target (`expertise`), and
**no OK/WARN/DARK status changes** — the single DARK flag (`expectation`,
2.9% → 0.0%) survives. So the paper's Table 2 stands; the defect is
nonetheless real and the reported shares are computed over a vocabulary
missing two thirds of its n-grams.

**The fix** Call the already-correct `phase3/terms.py::term_positions` path.
Pinned by `tests/test_tier1_golden.py`.

---

## L2 — The adversarial reviewer's fix threshold is measured against the target, not the text

**Where** `phase2/condense.py::run_adversarial_review`

**What happens** A proposed fix is accepted only if
`count_words(fixed) >= 0.5 * target_words` — half the *target* word count,
not half the length of the text actually under review.

**Measured** In the paper's own qwen2b run the accepted condensation was 747
words against a 3,456-word target. Any faithful minimal fix of it would be
about 747 words, and 747 < 1,728 — so **no fix could ever have been applied
to that text**, whatever the reviewer found. The decision log shows the
reviewer did flag an abrupt, mid-thought ending and did propose a fix, and
records `fixed text rejected as degenerate (empty or far too short); kept
original`.

**What it affects** Every condensation accepted by the soft-accept or
sanity-failed fallback below half the target — which is precisely the class of
output that most needs review.

**The fix** Compare against the length of the text being revised.

---

## L3 — Phase 0 can silently truncate the corpus, and logs nothing

**Where** `phase0/clean_corpus.py::remove_end_sections`, and the absence of any
`phase0_decisions.jsonl`

**What happens** Everything from the first line matching `^\s*notes\s*$`,
`^\s*references\s*$` or `^\s*bibliography\s*$` to the end of the file is
deleted. A mid-document "Notes" heading — ordinary in humanities articles with
per-section endnotes, and in book chapters — therefore deletes the rest of the
corpus. No count is reported and nothing is written to any decision log.

**Why it matters here specifically** Phase 0 is the only phase with no
decision log at all, inside a framework whose central claim is that every
consequential step leaves a record. And the README recommends `--clean` for
any PDF-sourced corpus.

**The fix** Have `clean_text` return per-rule removal counts, and write them
to a `phase0_decisions.jsonl` through the existing `DecisionLog`. Also
preserve the pre-cleaning text as `raw/<corpus>.original.txt`: today it exists
only in memory during the run, which is why resuming at Phase 2 has to fall
back to the cleaned text for source-metadata detection.

---

## L4 — Front-matter removal is bounded by a line offset, contrary to its own docstring

**Where** `phase0/clean_corpus.py::remove_front_matter` and `scan_front_matter`

**What happens** `scan_front_matter` scans the whole document and returns
`hits`; `remove_front_matter` discards those `hits` and removes only within
`lines[:body_start]`, where `body_start` is the position of the first
"Introduction" heading. The docstring on `scan_front_matter` documents the
whole-document behaviour as a fix applied after a real failure; the caller
does not use it. The paper's §2.3 describes the documented behaviour.

**The second failure mode** With no recognisable "Introduction" heading,
`body_start` falls back to the end of the document and the whole text is
scanned — where the `journal_masthead` pattern
`^(ORIGINAL RESEARCH|[A-Z][a-z]+\s+\(\d{4}\)|https://doi\.org/)` matches any
line that opens with an author-year citation. In such a document, every line
beginning `Hardwig (1985)` is deleted.

---

## L5 — An oversized-cluster re-split can place the same stem in two clusters

**Where** `phase1/pipeline.py::_split_oversized_clusters_once`

**What happens** The stems of *all* oversized clusters are pooled into one
re-clustering run, so a resulting subcluster can contain stems from two
different parents. The function then maps that subcluster to each parent and
materialises it once per parent — the same stems end up in two clusters. If
one parent contributed only that subcluster, it is also left in place with its
original stems intact.

**Measured** In the committed run, **11 stems belong to 2–3 clusters each**
(`action`, `artifici`, `causal`, `epistem`, `expertis`, `human`, `knowledg`,
`ration`, `sens`, `system`, `use`). The Phase 3 report's own *Cross-cluster
stems* section lists them, so the pipeline discloses the symptom; the paper's
§2.4, which describes re-splitting "considering only the stems belonging to the
oversized cluster", describes the intent rather than the code.

**What it affects** `compute_cluster_coverage` sums per-cluster hits to get its
denominator, so a stem counted twice shifts *every* cluster's share, not only
the duplicated one.

---

## L6 — Renaming two clusters to the same name silently drops one

**Where** `phase1/pipeline.py::run_cluster_review` (CLI) and
`webapp/pipeline_session.py::apply_cluster_review` (web)

**What happens** Both write `final_clusters[new_name] = …` with no
disambiguation. Rename two clusters to "trust" and the second overwrites the
first, with all of its stems, silently.

**Why it is worth flagging** `rename_clusters` already solves exactly this
class of bug one layer up, with a numeric suffix, and its docstring records the
measured case that motivated it ("6 of 20 raw noise clusters collided this
way, all lost, before this fix"). The researcher-facing rename step
reintroduces it — with free-text input, which makes a collision more likely,
not less.

---

## L7 — A flagged term nobody reviews disappears without record

**Where** `webapp/pipeline_session.py::apply_flagged_term_review`, and
`phase1/pipeline.py::run_flagged_term_review` in its "select from all terms"
mode

**What happens** A flagged word absent from the submitted decisions never
reaches `accepted_definitions`, so it drops out of embeddings, clustering and
every downstream phase — and appears in no exclusion list anywhere. Compare
`describe_wordnet_excluded_stems`, which exists precisely to surface the
analogous silent exclusion.

---

## L8 — `glossbert_accepted_terms.txt` is neither parseable nor reproducible

**Where** `phase1/pipeline.py::save_glossbert_output`

**What happens** The file has a TSV header but its columns are `repr()` of a
Python `set` and a `list` of dicts:
`expert\t{'expert'}\t169\t{'NOUN'}\t[{'word': 'expert', …}]`. It cannot be
read without `eval`, and because set iteration order for strings varies
between processes, **the file differs between byte-identical runs**.

**Why it is deferred furthest** Changing the format rewrites a committed
artifact outright, so this one waits for publication rather than for a test.

---

## L9 — After escalation, legitimate source vocabulary is reported as hallucinated

**Where** `phase2/condense.py::_check_invented_token_run`

**What happens** The check compares the condensation's vocabulary against the
informative sentences only — deliberately, since comparing against the full
source would match almost everything. But when the researcher escalates, the
full source *is* injected into the prompt, and words legitimately drawn from it
are then flagged as topic drift. The check is systematically biased against
exactly the trials escalation produces.

---

## L10 — A typo in the CLI's condensation setup discards Phase 1's work

**Where** `phase2/condense.py::run_condensation_setup`

**What happens** `rates = [int(r) …]` and `max_trials = int(input(…))` are
unguarded, and the announced 5–30% range is never validated — `0` and `500`
are accepted. An unparseable entry raises `ValueError` *after* Phase 1's whole
GPU pass and all of its manual review. The web interface handles the same
input correctly (`webapp/app.py` returns a 400); only the CLI does not.

---

## L11 — The web interface has no CSRF protection

**Where** `webapp/app.py`

**What happens** `request.get_json(force=True)` accepts a body sent with
`Content-Type: text/plain`, which a browser will send cross-origin without a
preflight. Any page the researcher happens to visit while a run is in progress
can therefore `POST /api/reset {"force": true}` and discard it, or
`POST /api/start` and overwrite `data/<corpus>`. Nothing can be read back
(cross-origin responses are opaque), so this destroys work rather than leaking
it. `MAX_CONTENT_LENGTH` is also unset, so an upload is unbounded in memory.

The interface is a local, single-researcher tool bound to `127.0.0.1`, which
is why this is a limitation rather than an emergency. Unlike every other entry
here, its fix changes no pipeline output — it could land before publication if
that is preferred.

---

## L12 — Smaller things, grouped

- `common/ollama_client.py::is_available` returns `True` for any HTTP
  response, including a 500.
- `phase3/collocations.py` takes `counts[len(counts) // 2]` as the median,
  which is the upper-middle element for an even-length list.
- The Phase 3 *Provenance* section reports "N real source collocation(s)
  found", where N has already been capped at `top_n_candidates=20`. It would
  read 20 whether 20 or 200 were found.
- `phase2/condense.py::classify_span` recomputes `_content_words` for every
  source sentence for every classified span — O(spans × sentences) with a
  regex per word.

---

## Not defects, but worth knowing before you run it

These are properties of the design, stated because the paper's claims are
easier to read against them.

- **LLM condensation is not reproducible run to run.** Generation passes
  `num_ctx` and `repeat_penalty` to Ollama, and nothing else — no `seed`, no
  `temperature`. Two identical runs produce different condensations. The
  *decisions* are auditable; the generated text is not reproducible.
- **The committed artifacts predate dependency pinning.** No library or model
  version was recorded when they were produced, and the models are loaded from
  Hugging Face `main` with no `revision=`. A rerun today may cluster
  differently. See the reproducibility note in [README.md](README.md).
- **Corpora are limited to about 2,000,000 characters** by
  `common/limits.py::SPACY_MAX_LENGTH`, and well before that by memory: Phase 2
  holds one parsed `Doc` per sentence for the whole corpus. The worked example
  is 83,164 characters. Book-length input will not work as shipped.
- **"This term matches this token" has three different answers** in the
  pipeline: bidirectional prefix matching in Phase 1's n-gram mining, exact
  stem plus filtered-lemma matching in Phase 3, and exact stem plus raw
  substring in cluster coverage (L1). Unifying them is L1's fix; naming them
  is done.
- **Phases 0–1 of the worked example ran once, under the corpus name
  `Boisseau`**, and their artifacts were copied into the three configuration
  directories. That is why every committed Phase 1 HTML header reads
  "Boisseau" rather than its own directory name.
