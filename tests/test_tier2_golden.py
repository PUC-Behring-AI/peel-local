"""Tier 2 golden tests: everything that needs spaCy and `en_core_web_sm`.

These lock the rest of the numbers the paper reports -- the F/T/R/C span
distribution, the source term frequencies and distribution shapes, the
stopword set and its auto-excluded author names, and the collocation
candidates.

**They are environment-sensitive on purpose.** Every assertion here depends on
spaCy's tokeniser, lemmatiser or NER. Upgrade spaCy or its model and these
move with no code change. That is the reproducibility signal the paper argues
for, made mechanical: a failure here after a dependency bump means the
environment drifted, not that the code broke. See CONTRIBUTING.md.

Verified against spaCy 3.8.11 / en_core_web_sm 3.8.0.
"""

import json

import pytest

from conftest import DATA_DIR, RATE

from phase2 import condense
from phase2 import pipeline as phase2_pipeline
from phase3 import collocations, distant_reading as dr, stopwords as sw, terms as pterms


# ------------------------------------------------------------
# Term matching -- strategy 2, the correct one (common/matching.py)
# ------------------------------------------------------------

def test_source_term_frequencies_match_the_published_table(source_doc, stemmer, corpus):
    """The Phase 3 term-frequency table's own raw counts, as the paper's §3.3
    reports them: `expert` 169, `system` 124, `trust` 105, `human` 91.

    Note the paper calls `human` the third-highest; it is fourth -- `trust`
    at 105 sits between them. Asserted here as the ranking actually is.
    """
    stems, lemmas = pterms.token_forms(source_doc(corpus), stemmer)
    counts = {
        term: len(pterms.term_positions(stems, lemmas, term))
        for term in ("expert", "system", "trust", "human")
    }
    assert counts == {"expert": 169, "system": 124, "trust": 105, "human": 91}
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    assert [term for term, _ in ranked] == ["expert", "system", "trust", "human"]


def test_ngrams_are_found_by_the_filtered_stream_not_literally(source_doc, stemmer, corpus):
    """The other side of L1: the n-grams that score zero under substring
    matching are found by `phase3/terms.py`, because it matches the same
    lemmatised, stopword-filtered stream Phase 1 mined them from."""
    doc = source_doc(corpus)
    stems, lemmas = pterms.token_forms(doc, stemmer)
    source_lower = doc.text.lower()

    for gram in ("expertise opacity trust", "expert competence", "trust agent"):
        assert source_lower.count(gram) == 0, f"{gram!r} should have no literal occurrence"
        assert len(pterms.term_positions(stems, lemmas, gram)) > 0, (
            f"{gram!r} should be found by the filtered-stream matcher"
        )


# ------------------------------------------------------------
# The injection taxonomy
# ------------------------------------------------------------

def _logged_reclassifications(corpus: str, rate: int = RATE) -> dict[str, str]:
    """Every researcher reclassification the Phase 2 decision log records for
    this corpus's borderline review, as `{span_id: new_type}`.

    Read from the log rather than hard-coded, because that is the point: the
    committed artifact is the heuristic's output *plus* these decisions, and
    the test below is what proves the log accounts for the difference.
    """
    path = DATA_DIR / corpus / "decisions" / "phase2_decisions.jsonl"
    changes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["decision_type"] != "borderline_reclassification_batch":
            continue
        per_rate = (entry.get("extra") or {}).get("decisions") or {}
        for span_id, change in per_rate.get(str(rate), {}).items():
            changes[span_id] = change["new_type"]
    return changes


def test_committed_spans_are_the_heuristic_plus_the_logged_decisions(
    corpus, nlp, condensed_text, source_text, injection_report
):
    """The architecture's central claim, tested end to end: the committed
    artifact equals what the deterministic classifier produces, plus exactly
    the researcher decisions the audit log records -- and nothing else.

    This is why the test is shaped this way rather than comparing the
    classifier's output directly against the artifact. Five spans across the
    two qwen9b configurations differ from the heuristic, and all five are
    accounted for by a `borderline_reclassification_batch` entry: `s9` and
    `s11` moved T->F and `s110` moved T->C in the 10-sentence run; `s100`
    moved T->C and `s130` moved T->F in the 20-sentence run. The qwen2b run
    has no such entry, and reproduces with no replay at all.

    If this test ever fails, exactly one of two things is true: the classifier
    changed, or the artifact contains a change no decision explains. Both are
    worth stopping for, and the failure message distinguishes them.
    """
    spans, _source_sentences = condense.classify_condensation(
        condensed_text(corpus), source_text(corpus), nlp
    )
    committed = injection_report(corpus)["spans"]

    assert len(spans) == len(committed)
    assert [s["span_id"] for s in spans] == [s["span_id"] for s in committed]
    assert [s["text"] for s in spans] == [s["text"] for s in committed]
    assert [s["source_refs"] for s in spans] == [s["source_refs"] for s in committed]

    # Replay the logged decisions, the same way run_injection_review does:
    # set the type in place and leave everything else, `basis` included,
    # untouched -- which is why a reclassified span still carries the basis
    # the heuristic assigned it.
    replayed = {s["span_id"]: s["type"] for s in spans}
    for span_id, new_type in _logged_reclassifications(corpus).items():
        assert span_id in replayed, f"log reclassifies {span_id}, which the classifier did not produce"
        replayed[span_id] = new_type

    unexplained = {
        s["span_id"]: (replayed[s["span_id"]], s["type"])
        for s in committed
        if replayed[s["span_id"]] != s["type"]
    }
    assert not unexplained, (
        "committed span types differ from the classifier's output in ways no logged "
        f"decision explains (span_id: (replayed, committed)): {unexplained}"
    )


def test_pre_review_span_distribution_per_configuration(corpus, nlp, condensed_text, source_text):
    """The classifier's own distribution, *before* any researcher decision --
    which is the number a change to `classify_span` would move.

    These differ from the published Table-2-adjacent counts by exactly the
    reclassifications above: the qwen9b/20 run is published as F=3 T=19 R=18
    C=71 and classifies as F=2 T=21 R=18 C=70.
    """
    expected = {
        "Boisseau-qwen2b-25cond-20informativesents": {"F": 0, "T": 8, "R": 6, "C": 17},
        "Boisseau-qwen9b-25cond-10informativesents": {"F": 1, "T": 23, "R": 16, "C": 85},
        "Boisseau-qwen9b-25cond-20informativesents": {"F": 2, "T": 21, "R": 18, "C": 70},
    }[corpus]
    spans, _ = condense.classify_condensation(
        condensed_text(corpus), source_text(corpus), nlp
    )
    assert {t: sum(1 for s in spans if s["type"] == t) for t in "FTRC"} == expected


def test_published_distribution_is_the_pre_review_one_plus_the_log(corpus, injection_report):
    """And the published counts, asserted as literals, so the two tests above
    cannot both drift in step without being noticed."""
    published = {
        "Boisseau-qwen2b-25cond-20informativesents": {"F": 0, "T": 8, "R": 6, "C": 17},
        "Boisseau-qwen9b-25cond-10informativesents": {"F": 3, "T": 20, "R": 16, "C": 86},
        "Boisseau-qwen9b-25cond-20informativesents": {"F": 3, "T": 19, "R": 18, "C": 71},
    }[corpus]
    spans = injection_report(corpus)["spans"]
    assert {t: sum(1 for s in spans if s["type"] == t) for t in "FTRC"} == published


def test_borderline_flags_reproduce_the_committed_report(
    corpus, nlp, condensed_text, source_text, injection_report
):
    """The borderline cross-check -- the word-list heuristic disagreeing with
    the POS/dependency classifier -- must flag the same spans it flagged when
    the committed report was produced."""
    spans, _ = condense.classify_condensation(
        condensed_text(corpus), source_text(corpus), nlp
    )
    flags = condense.flag_borderline_classifications(spans)
    committed = injection_report(corpus)["borderline_flags"]
    assert [f["span_id"] for f in flags] == [f["span_id"] for f in committed]


# ------------------------------------------------------------
# Phase 3 stopwords and collocations
# ------------------------------------------------------------

def test_stopword_set_and_auto_excluded_authors(nlp, source_doc, corpus):
    """The Phase 3 report's Provenance section: 443 stopwords in effect, and
    12 names auto-excluded as probable cited authors.

    The author list is a spaCy NER heuristic with a real false-positive rate,
    which is why the report discloses it by name rather than applying it
    silently -- and why it is pinned here rather than trusted.
    """
    stopword_set, auto_authors, _candidates = sw.build_stopwords(
        nlp.Defaults.stop_words, source_doc(corpus)
    )
    assert len(stopword_set) == 443
    assert auto_authors == [
        "Ross", "Freiman", "Ryan", "Rolin", "Hardwig", "Zerilli",
        "Burrell", "Berens", "Hacker", "Jones", "Watson", "Lowe",
    ]


def test_collocation_candidates_are_capped_at_twenty_and_ranked_by_hits(
    source_doc, phase1_state, stemmer, corpus
):
    """20 candidates, ranked by real co-occurrence count, drawn from pairs in
    different clusters.

    The count is also the cap (`top_n_candidates=20`), which is why the Phase 3
    report's "20 real source collocation(s) found" would read 20 whether 20 or
    200 existed -- recorded as L12 in LIMITATIONS.md.
    """
    candidates = collocations.find_source_collocations(
        source_doc(corpus), phase1_state(corpus)["clusterDefs"], stemmer
    )
    assert len(candidates) == 20
    assert [c["hits"] for c in candidates] == sorted(
        (c["hits"] for c in candidates), reverse=True
    )
    assert all(c["cluster_a"] != c["cluster_b"] for c in candidates)
    assert candidates[0]["term_a"] == "expert" and candidates[0]["term_b"] == "human"


# ------------------------------------------------------------
# Distribution shape
# ------------------------------------------------------------

def test_term_distribution_shapes_match_the_paper(source_doc, phase1_state, stemmer, corpus):
    """The five skewness figures the paper's §3.3 quotes, over eight segments:
    expert -0.61, describ 0.67, point 0.0, moral 1.03, action 1.09."""
    html = dr.build_term_stats_table(
        source_doc(corpus), phase1_state(corpus)["clusterDefs"], stemmer
    )
    import re

    rows = re.findall(
        r"<tr><td[^>]*>([^<]+)</td><td[^>]*>(\d+)</td><td[^>]*>[\d.]+%</td>"
        r"<td[^>]*>(-?[\d.]+)</td><td[^>]*>(-?[\d.]+)</td>",
        html,
    )
    skew_by_term = {r[0]: float(r[3]) for r in rows}
    for term, expected in (
        ("expert", -0.61), ("describ", 0.67), ("point", 0.0),
        ("moral", 1.03), ("action", 1.09),
    ):
        assert skew_by_term[term] == expected, f"{term}: {skew_by_term.get(term)} != {expected}"


def test_cross_cluster_stems_are_disclosed_in_the_report(phase1_state, corpus):
    """`build_cross_cluster_html` is how the pipeline discloses L5's symptom.
    Pinned so the disclosure cannot quietly disappear."""
    from phase3 import report as phase3_report

    html = phase3_report.build_cross_cluster_html(phase1_state(corpus)["clusterDefs"])
    for stem in ("epistem", "system", "action"):
        assert f"<code>{stem}</code>" in html
    assert html.count("<li>") == 11


# ------------------------------------------------------------
# Phase 2 sentence selection -- the whole of Phase 2's first half
# ------------------------------------------------------------

@pytest.mark.slow
def test_informative_sentences_regenerate_identically(corpus, nlp, stemmer, source_text):
    """Regenerating `informative_sentences.json` must reproduce the committed
    file exactly -- which locks the density percentile, the scoring function,
    stem and n-gram matching, and the highlighting, all at once.

    Marked slow because `segment_corpus` parses the corpus once and then
    re-parses every sentence individually, so this is minutes per corpus
    rather than seconds.
    """
    committed_path = DATA_DIR / corpus / "phase2" / "informative_sentences.json"
    committed = json.loads(committed_path.read_text(encoding="utf-8"))

    top_n = 10 if "10informativesents" in corpus else 20
    state = {"excludedNgrams": committed["excludedNgrams"],
             "clusterDefs": [{k: v for k, v in c.items() if k != "most_insightful_sentences"}
                             for c in committed["clusterDefs"]]}

    regenerated = phase2_pipeline.enrich_with_informative_sentences(
        state, source_text(corpus), nlp, stemmer, top_n=top_n, density_percentile=75,
    )
    assert regenerated == committed
