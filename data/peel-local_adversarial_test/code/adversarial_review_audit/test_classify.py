"""Pins the audit taxonomy to the pipeline's real behaviour.

For each canned reviewer response, the real
`phase2/condense.py::run_adversarial_review` is run with Ollama replaced by
a stub, and the classifier must agree with what it decided. No Ollama, no
network. Run from the repository root:

    python -m pytest experiments/adversarial_review_audit/test_classify.py
"""

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("PEEL_REPO", HERE.parents[1])).resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from common import ollama_client  # noqa: E402
from phase2 import condense  # noqa: E402

from classify import GROUP_OF, classify_call, word_similarity  # noqa: E402

TARGET = 100
REVIEWED = " ".join(f"w{i}" for i in range(120)) + "."
SHORT = " ".join(f"w{i}" for i in range(30))            # < 0.5 * TARGET
EDITED = REVIEWED.replace("w7 ", "seven ")
EXPANDED = REVIEWED + " " + " ".join(f"x{i}" for i in range(40))   # 161 > 1.05 * 120


class _Log:
    def __init__(self):
        self.records = []

    def record(self, **kw):
        self.records.append(kw)


CASES = [
    ("call_failed", None, "boom"),
    ("empty_response", "   \n", None),
    ("unparseable", "I think the text is fine.", None),
    ("ok", "VERDICT: OK", None),
    ("fixed_no_text", "VERDICT: FIXED\nISSUES: abrupt ending", None),
    ("fixed_no_text", "VERDICT: FIXED\nISSUES: abrupt ending\nTEXT:\n   ", None),
    ("fixed_rejected_short", f"VERDICT: FIXED\nISSUES: x\nTEXT:\n{SHORT}", None),
    ("fixed_identical", f"VERDICT: FIXED\nISSUES: repeated\nTEXT:\n{REVIEWED}", None),
    ("fixed_identical", f"VERDICT: FIXED\nISSUES: repeated\nTEXT:\n  {REVIEWED.replace(' ', '  ')}\n", None),
    ("fixed_changed", f"VERDICT: FIXED\nISSUES: typo\nTEXT:\n{EDITED}", None),
    ("fixed_expanded", f"VERDICT: FIXED\nISSUES: drift\nTEXT:\n{EXPANDED}", None),
]


@pytest.mark.parametrize("expected, response, error", CASES)
def test_classifier_matches_pipeline(monkeypatch, expected, response, error):
    def fake_generate(*a, **kw):
        if error is not None:
            raise ollama_client.OllamaError(error)
        return response

    monkeypatch.setattr(condense.ollama_client, "generate", fake_generate)
    log = _Log()
    final, was_fixed, _orig, _issues = condense.run_adversarial_review(
        REVIEWED, [{"sentence": "s"}], TARGET, "c", "m", log, 25)

    got = classify_call(None if error else response, error, REVIEWED, TARGET)
    assert got["category"] == expected
    assert got["group"] == GROUP_OF[expected]
    assert got["pipeline_choice"] == log.records[-1]["choice"]
    assert was_fixed == (got["pipeline_choice"] == "fixed")
    if expected == "fixed_identical":
        # The defect itself: logged as fixed, text unchanged up to whitespace.
        assert " ".join(final.split()) == " ".join(REVIEWED.split())


def test_similarity_separates_copy_cut_short_from_edit():
    ratio, containment = word_similarity(REVIEWED, SHORT)
    assert ratio < 0.5 and containment == 1.0
    ratio, containment = word_similarity(REVIEWED, EDITED)
    assert 0.99 <= ratio < 1.0
