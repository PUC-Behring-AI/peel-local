"""Shared fixtures for PEEL-Local's characterisation and golden tests.

These tests assert what the pipeline does *today*, quirks included. That is
deliberate: the artifacts under `data/` are the evidence the paper reports, so
release work has to be provably byte-identical to the `v1.0.0-paper` tag. A
golden test failing means an output moved -- which is a failed change, not a
new baseline. See CONTRIBUTING.md for the tier system and for what to do when
a Tier 2 test fails after a dependency bump (it usually means the environment
drifted, not that the code broke).

Tiers, by what they need installed:

  Tier 0  stdlib only                      -- always runs
  Tier 1  + nltk (PorterStemmer, WordNet)  -- skipped if nltk is missing
  Tier 2  + spaCy and en_core_web_sm       -- skipped if the model is missing
  Tier 3  + GPU / Ollama                   -- marked `slow`, deselected by default

Expensive objects (the spaCy pipeline, the parsed source document) are
session-scoped: parsing the 83k-character example corpus takes seconds, and
Tier 2 reuses it across every test.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"

# The three Phase 2 configurations the paper's worked example reports (§3.2).
# Kept as an explicit tuple rather than globbed, so a test can never silently
# pass by finding zero corpora.
CORPORA = (
    "Boisseau-qwen2b-25cond-20informativesents",
    "Boisseau-qwen9b-25cond-10informativesents",
    "Boisseau-qwen9b-25cond-20informativesents",
)

# The single condensation rate every committed corpus was generated at.
RATE = 25


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: needs a GPU, a downloaded model, or a running Ollama; deselected by default"
    )


# ------------------------------------------------------------
# Tier gates
# ------------------------------------------------------------

def _require(module_name, reason):
    return pytest.importorskip(module_name, reason=reason)


@pytest.fixture(scope="session")
def stemmer():
    """Tier 1+. The same PorterStemmer every phase constructs."""
    nltk_stem = _require("nltk.stem", "Tier 1 needs nltk (pip install -r requirements.txt)")
    return nltk_stem.PorterStemmer()


@pytest.fixture(scope="session")
def nlp():
    """Tier 2+. The default spaCy pipeline, loaded once per session.

    `max_length` is raised the same way the pipeline itself raises it, so a
    test never fails for a reason the real run would not hit.
    """
    spacy = _require("spacy", "Tier 2 needs spaCy (pip install -r requirements.txt)")
    try:
        loaded = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip(
            "Tier 2 needs the spaCy model: python -m spacy download en_core_web_sm"
        )
    from common.limits import SPACY_MAX_LENGTH

    loaded.max_length = SPACY_MAX_LENGTH
    return loaded


# ------------------------------------------------------------
# Corpus access
# ------------------------------------------------------------

@pytest.fixture(scope="session", params=CORPORA)
def corpus(request):
    """Parametrised over all three committed corpora, so every golden
    assertion is made three times rather than on whichever one is handy."""
    return request.param


@pytest.fixture(scope="session")
def paths_for():
    """`paths_for(corpus_name)` -> CorpusPaths, the pipeline's own path
    contract rather than strings rebuilt here."""
    from common.paths import CorpusPaths

    return CorpusPaths


@pytest.fixture(scope="session")
def source_text():
    """`source_text(corpus)` -> the raw corpus text, cached per corpus."""
    cache = {}

    def _get(name):
        if name not in cache:
            cache[name] = (DATA_DIR / name / "raw" / f"{name}.txt").read_text(encoding="utf-8")
        return cache[name]

    return _get


@pytest.fixture(scope="session")
def condensed_text():
    """`condensed_text(corpus)` -> the accepted 25% condensation."""
    cache = {}

    def _get(name):
        if name not in cache:
            path = DATA_DIR / name / "phase2" / "condensation" / f"{name}-condensed-{RATE}pct.txt"
            cache[name] = path.read_text(encoding="utf-8")
        return cache[name]

    return _get


@pytest.fixture(scope="session")
def phase1_state():
    """`phase1_state(corpus)` -> the committed Phase 1 state JSON."""
    cache = {}

    def _get(name):
        if name not in cache:
            path = DATA_DIR / name / "phase1" / f"{name}-phase1_state.json"
            cache[name] = json.loads(path.read_text(encoding="utf-8"))
        return cache[name]

    return _get


@pytest.fixture(scope="session")
def injection_report():
    """`injection_report(corpus)` -> the committed injection report JSON,
    which carries `spans`, `coverage` and `stats` -- the numbers Table 2 of
    the paper reports."""
    cache = {}

    def _get(name):
        if name not in cache:
            path = (
                DATA_DIR / name / "phase2" / "condensation"
                / f"{name}-condensed-{RATE}pct-injection-report.json"
            )
            cache[name] = json.loads(path.read_text(encoding="utf-8"))
        return cache[name]

    return _get


@pytest.fixture(scope="session")
def source_doc(nlp, source_text):
    """`source_doc(corpus)` -> the parsed source. Session-scoped and cached:
    all three corpora share the same source text, so this parses once."""
    cache = {}

    def _get(name):
        text = source_text(name)
        if text not in cache:
            cache[text] = nlp(text)
        return cache[text]

    return _get


@pytest.fixture(scope="session")
def final_clusters_from_state(phase1_state):
    """Rebuilds the `final_clusters` dict `build_cluster_html` expects from a
    committed `clusterDefs` list, so the golden HTML test can call the real
    function with the real run's data rather than a hand-made fixture."""

    def _get(name):
        return {
            cluster["name"]: {
                "stems": cluster.get("stems", []),
                "ngrams": cluster.get("ngrams", []),
                "excluded_stems": cluster.get("excluded_stems", []),
                "excluded_ngrams": cluster.get("excluded_ngrams", []),
                "reembedded_stems": cluster.get("reembedded_stems", []),
                "reembedded_source": cluster.get("reembedded_source"),
            }
            for cluster in phase1_state(name)["clusterDefs"]
        }

    return _get
