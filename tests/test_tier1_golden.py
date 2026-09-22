"""Tier 1 golden tests: the cluster-coverage table the paper's Table 2 reports.

Needs nltk (PorterStemmer) but no spaCy model and no network, which is what
makes this affordable to run on every push -- and it locks the single number
the paper's §4.2 argument turns on: the one DARK cluster.

Why this matters beyond regression-catching. `condense.cluster_hit_counts`
counts a cluster's n-grams by raw substring search, but Phase 1 mines n-grams
from a lemmatised, stopword-filtered stream, so most of them do not occur
literally in the text: measured on the committed corpus, 22 of 33 n-grams
count zero and 46% of all n-gram occurrences in the source are lost. Fixing
that is tracked as an issue, not done here, because it moves an output. These
tests are what make that fix safe to land afterwards -- they say exactly which
numbers currently exist, so the fix's effect is a diff rather than a guess.
(Measured in advance: the correct matching shifts one cluster's target by
+0.9pp and flips no status.)
"""

import pytest

from conftest import DATA_DIR, RATE

from phase2 import condense
from phase3 import distant_reading as dr


def test_coverage_table_reproduces_committed_report(
    corpus, phase1_state, condensed_text, source_text, stemmer, injection_report
):
    """`compute_cluster_coverage` must reproduce the committed coverage table
    exactly -- all 11 clusters, target/actual/delta/status, for every one of
    the three configurations. This is Table 2's cluster-coverage column."""
    recomputed = condense.compute_cluster_coverage(
        phase1_state(corpus), condensed_text(corpus), source_text(corpus), stemmer
    )
    assert recomputed == injection_report(corpus)["coverage"]


def test_the_dark_cluster_is_expectation_at_2_9_percent(
    phase1_state, condensed_text, source_text, stemmer
):
    """The paper's §4.2 headline: the smaller model's run is the only one of
    the three to lose a semantic cluster outright. Asserted against the
    literal published figures rather than against the committed JSON, so a
    change that moved code and artifact together would still be caught."""
    corpus = "Boisseau-qwen2b-25cond-20informativesents"
    coverage = condense.compute_cluster_coverage(
        phase1_state(corpus), condensed_text(corpus), source_text(corpus), stemmer
    )
    assert coverage["expectation"] == {
        "target": 2.9, "actual": 0.0, "delta": -2.9, "status": "DARK",
    }
    assert [name for name, c in coverage.items() if c["status"] == "DARK"] == ["expectation"]


def test_coverage_is_a_share_of_cluster_vocabulary_only(
    corpus, phase1_state, condensed_text, source_text, stemmer
):
    """The paper's §2.5 claim about what the percentages are a share *of*:
    cluster vocabulary, not document length. So the targets sum to ~100%.

    They sum to 100% of a total that double-counts, though: 11 stems belong to
    more than one cluster in this run (the Phase 3 report's own
    "Cross-cluster stems" section lists them), and a shared stem's occurrences
    are counted once per cluster. Asserted here so the property is recorded
    rather than assumed.
    """
    coverage = condense.compute_cluster_coverage(
        phase1_state(corpus), condensed_text(corpus), source_text(corpus), stemmer
    )
    assert sum(c["target"] for c in coverage.values()) == pytest.approx(100.0, abs=0.5)


def test_stems_shared_across_clusters_are_counted_once_per_cluster(corpus, phase1_state):
    """The double-counting above, stated as a fact about the data: exactly 11
    stems appear in more than one cluster. If a future change partitions the
    vocabulary instead, this test fails and the coverage denominator changes
    with it -- which is the point of pinning it."""
    owners = {}
    for cluster in phase1_state(corpus)["clusterDefs"]:
        for stem in cluster.get("stems", []):
            owners.setdefault(stem, []).append(cluster["name"])
    shared = {s: names for s, names in owners.items() if len(names) > 1}
    assert sorted(shared) == [
        "action", "artifici", "causal", "epistem", "expertis",
        "human", "knowledg", "ration", "sens", "system", "use",
    ]


def test_ngram_substring_counting_loses_most_ngram_occurrences(corpus, phase1_state, source_text):
    """Pins the defect itself, so the fix that removes it is visible as a diff.

    `cluster_hit_counts` counts an n-gram with `text.count(gram)`. Phase 1
    mined those n-grams from a lemmatised, stopword-filtered stream, so a real
    phrase like "expertise opacity trust" (from "Expertise, opacity, and
    trust") has no literal occurrence at all. 22 of the 33 n-grams in this
    run's clusters therefore contribute exactly zero, on both sides of the
    comparison.
    """
    source = source_text(corpus).lower()
    ngrams = [g for c in phase1_state(corpus)["clusterDefs"] for g in c.get("ngrams", [])]
    zero = [g for g in ngrams if source.count(g.lower()) == 0]

    assert len(ngrams) == 33
    assert len(zero) == 22
    # A sample, to show these are real phrases rather than mining artifacts.
    for gram in ("expertise opacity trust", "expert competence", "trust agent"):
        assert gram in zero


def test_cluster_prevalence_chart_uses_the_same_counting_as_coverage(
    corpus, phase1_state, source_text, stemmer
):
    """`bin_cluster_frequencies` reuses `cluster_hit_counts`, so the Phase 3
    trend chart inherits the n-gram undercount above -- while the term-stats
    table in the same report uses the correct `phase3/terms.py` matching.

    That disagreement inside one report is tracked as an issue. Pinned here as
    the total across bins, so the fix shows up as a changed number rather than
    silently.
    """
    freqs = dr.bin_cluster_frequencies(
        source_text(corpus), phase1_state(corpus)["clusterDefs"], stemmer, n_bins=5
    )
    assert sorted(freqs) == [
        "action", "agent", "character", "epistemic", "epistemic AI", "expectation",
        "expertise", "human", "moral", "systems", "trust",
    ]
    assert all(len(v) == 5 for v in freqs.values())
    # Summed over bins, this must equal the whole-source hit count the
    # coverage table's denominator is built from.
    per_cluster = condense.cluster_hit_counts(
        source_text(corpus), phase1_state(corpus)["clusterDefs"], stemmer
    )
    assert sum(sum(v) for v in freqs.values()) == pytest.approx(
        sum(per_cluster.values()), rel=0.02
    )


def test_public_alias_for_cluster_hit_counts(corpus, phase1_state, source_text, stemmer):
    """`phase3/distant_reading.py` imported `condense._cluster_hit_counts` --
    a private name -- across a module boundary. The public name is the same
    function; the private alias stays so nothing that referenced it breaks."""
    args = (source_text(corpus), phase1_state(corpus)["clusterDefs"], stemmer)
    assert condense.cluster_hit_counts(*args) == condense._cluster_hit_counts(*args)


def test_condensed_text_path_matches_the_paths_contract(corpus, paths_for):
    """The path contract resolves to the artifacts this suite reads, so a
    change to `CorpusPaths` cannot silently point the tests elsewhere."""
    paths = paths_for(corpus)
    expected = DATA_DIR / corpus / "phase2" / "condensation" / f"{corpus}-condensed-{RATE}pct.txt"
    assert paths.condensation_paths(RATE)["condensed_text"] == expected
    assert paths.raw_txt().exists()
    assert paths.phase1_state_json().exists()
