"""Tier 0 golden tests: regenerate a committed artifact and demand it match.

Stdlib only -- no spaCy, no nltk, no network. These are the cheapest tests
that lock a number the paper reports, so they run everywhere and first.

A failure here is not a new baseline. It means a change moved an output that
the paper's Table 2 reports, which under the `v1.0.0` behaviour freeze
makes it a failed change. See CONTRIBUTING.md.
"""

import subprocess

import pytest

from conftest import DATA_DIR, RATE, REPO_ROOT

from phase1 import pipeline as phase1_pipeline
from phase2 import condense


# ------------------------------------------------------------
# The whole of data/ is byte-frozen
# ------------------------------------------------------------

def test_committed_artifacts_are_unchanged():
    """Every pipeline artifact under data/ must still hash to what it hashed
    at the v1.0.0 tag.

    This is the backstop for the entire behaviour freeze: it catches an
    output moving even when no individual golden test below covers that
    particular file. PROVENANCE.md files are excluded from the manifest by
    construction -- they are documentation added after the tag, not pipeline
    output.
    """
    manifest = REPO_ROOT / "tests" / "golden" / "data.sha256"
    assert manifest.exists(), f"missing hash manifest: {manifest}"

    result = subprocess.run(
        ["shasum", "-a", "256", "-c", str(manifest)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        failed = [line for line in result.stdout.splitlines() if not line.endswith("OK")]
        pytest.fail(
            "committed artifacts under data/ have changed -- under the behaviour "
            "freeze this is a failed change, not a new baseline:\n  "
            + "\n  ".join(failed or [result.stderr.strip()])
        )

    checked = sum(1 for line in result.stdout.splitlines() if line.endswith("OK"))
    assert checked == 51, f"expected 51 frozen artifacts, verified {checked}"


def test_no_artifact_is_missing_from_the_manifest():
    """A new artifact added under data/ without being hashed would slip past
    the test above entirely, so the manifest's own coverage is asserted."""
    manifest_paths = set()
    for line in (REPO_ROOT / "tests" / "golden" / "data.sha256").read_text().splitlines():
        if line.strip():
            manifest_paths.add(line.split("  ", 1)[1])

    on_disk = {
        str(p.relative_to(REPO_ROOT))
        for p in DATA_DIR.rglob("*")
        if p.is_file() and p.name != "PROVENANCE.md"
    }
    assert on_disk == manifest_paths, (
        "data/ and tests/golden/data.sha256 disagree.\n"
        f"  on disk but unhashed: {sorted(on_disk - manifest_paths)}\n"
        f"  hashed but missing:   {sorted(manifest_paths - on_disk)}"
    )


# ------------------------------------------------------------
# Phase 1's HTML export
# ------------------------------------------------------------

# The corpus name Phase 1 actually ran under. Not the directory name: the
# paper's §3.2 says Phases 0-1 "were run once" and their output was "carried
# into three independent Phase 2 configurations", and this is the mechanism --
# the run was named `Boisseau`, and its Phase 1 artifacts were copied into the
# three `Boisseau-<model>-25cond-<n>informativesents` directories afterwards.
# So every committed Phase 1 HTML header reads "Boisseau", not its own
# directory's name. Asserted directly below rather than left as a surprise.
PHASE1_RUN_NAME = "Boisseau"


def test_phase1_artifacts_were_produced_under_one_shared_run_name(corpus):
    """The three configurations share one Phase 1 run, named `Boisseau`."""
    committed = (DATA_DIR / corpus / "phase1" / f"{corpus}-Phase1-clusters.html").read_text(
        encoding="utf-8"
    )
    assert f"\n  {PHASE1_RUN_NAME} &mdash;\n" in committed
    assert corpus not in committed


def test_cluster_html_reproduces_committed_export(corpus, final_clusters_from_state):
    """`build_cluster_html` fed the committed Phase 1 state must reproduce the
    committed HTML byte for byte.

    This is the test that makes escaping the Phase 1 export safe to add:
    `phase1/pipeline.py` was the only HTML-producing module with no
    `html.escape`, and no cluster name, stem or n-gram in any committed state
    contains `& < > " '`, so applying it cannot change these bytes. If a future
    corpus does contain one, this test will fail on that corpus -- correctly,
    because then the escaping does change the output.
    """
    committed = (DATA_DIR / corpus / "phase1" / f"{corpus}-Phase1-clusters.html").read_text(
        encoding="utf-8"
    )
    regenerated = phase1_pipeline.build_cluster_html(
        final_clusters_from_state(corpus), PHASE1_RUN_NAME
    )
    assert regenerated == committed


def test_no_committed_cluster_vocabulary_needs_escaping(corpus, phase1_state):
    """The premise the test above rests on, asserted directly rather than
    left implicit: if this fails, escaping is no longer a no-op."""
    needs_escaping = []
    for cluster in phase1_state(corpus)["clusterDefs"]:
        values = (
            [cluster["name"]]
            + cluster.get("stems", [])
            + cluster.get("ngrams", [])
            + cluster.get("excluded_stems", [])
            + cluster.get("excluded_ngrams", [])
        )
        needs_escaping += [v for v in values if any(c in v for c in "&<>\"'")]
    assert needs_escaping == []


# ------------------------------------------------------------
# Table 2's verification statistics
# ------------------------------------------------------------

def test_injection_stats_reproduce_table_2(corpus, injection_report, condensed_text, source_text):
    """`compute_injection_stats`, fed the committed spans, must reproduce the
    `non_injected_pct` and `verbatim_overlap_pct` the paper's Table 2 reports
    (5.8/41.6, 8.2/35.1, 22.0/54.7 across the three configurations).

    Both are pure string work -- no model, no stemmer -- so this locks two of
    Table 2's five columns at Tier 0.
    """
    report = injection_report(corpus)
    recomputed = condense.compute_injection_stats(
        report["spans"], condensed_text(corpus), source_text(corpus)
    )
    assert recomputed == report["stats"]


@pytest.mark.parametrize(
    "corpus_name, expected_verbatim_pct",
    [
        ("Boisseau-qwen2b-25cond-20informativesents", 41.6),
        ("Boisseau-qwen9b-25cond-10informativesents", 35.1),
        ("Boisseau-qwen9b-25cond-20informativesents", 54.7),
    ],
)
def test_verbatim_overlap_matches_the_published_percentage(
    corpus_name, expected_verbatim_pct, condensed_text, source_text
):
    """The verbatim-overlap scan, asserted against the literal percentages in
    the paper rather than only against the committed JSON -- so a change that
    moved both the code and the artifact in step would still be caught."""
    matched, total, _runs = condense.scan_verbatim_overlap(
        condensed_text(corpus_name), source_text(corpus_name)
    )
    assert round(100 * matched / total, 1) == expected_verbatim_pct


@pytest.mark.parametrize(
    "corpus_name, expected_words",
    [
        ("Boisseau-qwen2b-25cond-20informativesents", 747),
        ("Boisseau-qwen9b-25cond-10informativesents", 3062),
        ("Boisseau-qwen9b-25cond-20informativesents", 3079),
    ],
)
def test_condensation_word_counts_match_table_2(corpus_name, expected_words, condensed_text):
    assert condense.count_words(condensed_text(corpus_name)) == expected_words


def test_source_word_count_and_target(source_text):
    """The source is 13,826 words by the pipeline's own `count_words`, and the
    25% target is 3,456 -- the number every trial in Table 2 was measured
    against. (The paper rounds the source to 13,824; the derived target is the
    same either way.)"""
    text = source_text("Boisseau-qwen9b-25cond-20informativesents")
    assert condense.count_words(text) == 13826
    assert condense.target_word_count(text, RATE) == 3456


def test_span_type_counts_match_committed_reports(corpus, injection_report):
    """F/T/R/C counts per configuration, read from the committed reports. The
    classifier that produced them is Tier 2; this asserts the artifacts
    themselves still carry the distribution the paper's §3.2 discussion rests
    on."""
    expected = {
        "Boisseau-qwen2b-25cond-20informativesents": {"F": 0, "T": 8, "R": 6, "C": 17},
        "Boisseau-qwen9b-25cond-10informativesents": {"F": 3, "T": 20, "R": 16, "C": 86},
        "Boisseau-qwen9b-25cond-20informativesents": {"F": 3, "T": 19, "R": 18, "C": 71},
    }[corpus]
    spans = injection_report(corpus)["spans"]
    actual = {t: sum(1 for s in spans if s["type"] == t) for t in "FTRC"}
    assert actual == expected


def test_only_one_cluster_is_dark_and_it_is_expectation(corpus, injection_report):
    """Table 2's cluster-coverage column: exactly one DARK flag across all
    three runs (`expectation`, 2.9% -> 0.0%, in the qwen2b configuration), and
    no WARN anywhere -- the caption's claim that "every other decrease is
    still flagged OK"."""
    coverage = injection_report(corpus)["coverage"]
    dark = {name: c for name, c in coverage.items() if c["status"] == "DARK"}
    warn = {name: c for name, c in coverage.items() if c["status"] == "WARN"}

    assert warn == {}
    if corpus == "Boisseau-qwen2b-25cond-20informativesents":
        assert dark == {
            "expectation": {"target": 2.9, "actual": 0.0, "delta": -2.9, "status": "DARK"}
        }
    else:
        assert dark == {}


def test_all_three_corpora_share_one_phase1_run(phase1_state):
    """The paper's §3.2 premise: Phases 0-1 ran once and the three Phase 2
    configurations share that corpus contract. If the states ever diverge, the
    comparison in Table 2 stops isolating what it claims to isolate."""
    states = [phase1_state(name) for name in
              ("Boisseau-qwen2b-25cond-20informativesents",
               "Boisseau-qwen9b-25cond-10informativesents",
               "Boisseau-qwen9b-25cond-20informativesents")]
    assert states[0] == states[1] == states[2]
    assert [c["name"] for c in states[0]["clusterDefs"]] == [
        "action", "agent", "character", "epistemic", "epistemic AI", "expectation",
        "expertise", "human", "moral", "systems", "trust",
    ]
