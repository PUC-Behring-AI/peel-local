"""Persists interactive review decisions as consultable artifacts.

The final phase JSON (e.g. phase1_state.json) only records outcomes.
DecisionLog records *what alternatives existed, what was chosen, and when*
for every interactive review step, appended as JSON Lines so the history
can be replayed or audited later without altering interactive behavior.
"""

import json
from datetime import datetime, timezone

from common.paths import CorpusPaths


class DecisionLog:
    def __init__(self, corpus_name: str, phase: str):
        self.corpus_name = corpus_name
        self.phase = phase
        self.path = CorpusPaths(corpus_name).decisions_log(phase)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, step: str, decision_type: str, prompt: str,
               options=None, choice=None, extra: dict | None = None) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "corpus": self.corpus_name,
            "phase": self.phase,
            "step": step,
            "decision_type": decision_type,
            "prompt": prompt,
            "options": options,
            "choice": choice,
            "extra": extra or {},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry
