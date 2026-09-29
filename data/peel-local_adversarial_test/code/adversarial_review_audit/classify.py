"""Outcome taxonomy for one adversarial-review call.

Stdlib only. The parse step is the pipeline's own
`phase2/condense.py::parse_adversarial_response`, and the acceptance rule
below is the one in `phase2/condense.py::run_adversarial_review`, so a call
is classified exactly as the pipeline would have treated it. That
equivalence is pinned by `test_classify.py`, which runs the real
`run_adversarial_review` on the same canned responses.

Nine mutually exclusive categories, in the order the pipeline meets them:

    call_failed           the HTTP call raised (unreachable, non-200, timeout)
    empty_response        the model returned only whitespace
    unparseable           non-empty, but no `VERDICT: OK|FIXED` anywhere
    ok                    `VERDICT: OK`
    fixed_no_text         `VERDICT: FIXED` with no `TEXT:` block, or an empty one
    fixed_rejected_short  FIXED with text, but under 0.5 x target words (L2)
    fixed_identical       FIXED, accepted, and equal to the reviewed text
                          after whitespace normalisation
    fixed_expanded        FIXED, accepted, different, and longer than
                          1.05 x max(reviewed words, target words)
    fixed_changed         FIXED, accepted, and different, within that bound

and four user-facing groups:

    no_output   the researcher receives nothing from the reviewer: every
                category above `fixed_identical` except `ok`. The pipeline
                logs `unchanged`.
    same_text   the pipeline logs `fixed`, but the text is the reviewed one.
    ok          the reviewer found no defect.
    expanded    the reviewer returned a text longer than both what it was
                given and the target, beyond the pipeline's own 5% length
                tolerance -- the opposite of the "smallest possible edit" the
                prompt asks for. No community standard exists for this; the
                5% bound is the length tolerance PEEL-Local inherits from PEEL
                (attempt_condensation_trials, tolerance=0.05).
    changed     the reviewer returned an edited text within that bound.

`fixed_changed` is further graded by word-level similarity, because "the
same text" also has a near-identical reading (a full stop added, one word
swapped); `analyze.py` reports that at several thresholds.
"""

from __future__ import annotations

import difflib

from phase2.condense import count_words, parse_adversarial_response

CATEGORIES = (
    "call_failed",
    "empty_response",
    "unparseable",
    "ok",
    "fixed_no_text",
    "fixed_rejected_short",
    "fixed_identical",
    "fixed_expanded",
    "fixed_changed",
)

GROUP_OF = {
    "call_failed": "no_output",
    "empty_response": "no_output",
    "unparseable": "no_output",
    "fixed_no_text": "no_output",
    "fixed_rejected_short": "no_output",
    "fixed_identical": "same_text",
    "fixed_expanded": "expanded",
    "ok": "ok",
    "fixed_changed": "changed",
}

GROUPS = ("no_output", "same_text", "expanded", "ok", "changed")

# The pipeline's acceptance rule, run_adversarial_review: a FIXED text is
# applied only if it is non-empty and count_words(fixed) >= 0.5 * target_words.
FIX_MIN_FRACTION = 0.5

# Expansion bound: the pipeline's own length tolerance (5%), applied to the
# larger of the reviewed text and the target, so a fix that merely restores
# a short text toward its target is not counted as expansion.
EXPANSION_TOLERANCE = 0.05


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def word_similarity(original: str, candidate: str) -> tuple[float, float]:
    """(ratio, containment) over whitespace tokens.

    ratio        difflib's 2*M/T -- 1.0 means identical token sequences.
    containment  share of the candidate's tokens that sit inside blocks it
                 shares with the original -- near 1.0 with a low ratio means
                 the candidate is a *piece* of the original (a copy cut short).
    """
    a, b = original.split(), candidate.split()
    if not a and not b:
        return 1.0, 1.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    return sm.ratio(), (matched / len(b) if b else 0.0)


def classify_call(response: str | None, error: str | None, reviewed_text: str,
                  target_words: int) -> dict:
    """Classifies one reviewer call. `response` is the raw model output
    (Ollama's `response` field); `error` is set when the call raised."""
    out = {
        "category": None, "group": None, "pipeline_choice": "unchanged",
        "verdict": None, "fixed_words": None, "reviewed_words": count_words(reviewed_text),
        "similarity_ratio": None, "containment": None,
    }

    if error is not None:
        out["category"] = "call_failed"
    elif response is None or not response.strip():
        out["category"] = "empty_response"
    else:
        verdict, _issues, fixed_text = parse_adversarial_response(response)
        out["verdict"] = verdict
        if verdict == "UNPARSEABLE":
            out["category"] = "unparseable"
        elif verdict == "OK":
            out["category"] = "ok"
        elif not fixed_text:
            out["category"] = "fixed_no_text"
        else:
            out["fixed_words"] = count_words(fixed_text)
            ratio, containment = word_similarity(reviewed_text, fixed_text)
            out["similarity_ratio"], out["containment"] = ratio, containment
            if count_words(fixed_text) < FIX_MIN_FRACTION * target_words:
                out["category"] = "fixed_rejected_short"
            else:
                out["pipeline_choice"] = "fixed"
                bound = (1 + EXPANSION_TOLERANCE) * max(out["reviewed_words"], target_words)
                if normalize_ws(fixed_text) == normalize_ws(reviewed_text):
                    out["category"] = "fixed_identical"
                elif out["fixed_words"] > bound:
                    out["category"] = "fixed_expanded"
                else:
                    out["category"] = "fixed_changed"

    out["group"] = GROUP_OF[out["category"]]
    return out
