"""Lets a researcher resume an existing corpus at Phase 2 or Phase 3
instead of always starting a brand-new corpus at Phase 1 -- shared by the
CLI (--start-phase) and the webapp (the "resume an existing corpus" setup
option) so both use identical corpus-discovery and prerequisite-checking
logic, never two independently-drifting copies of it.

Phase 3 lives inside run_phase2's own tail (see run_pipeline.py) rather
than being invoked from main() directly -- it needs Phase 2's
condensed_texts dict, not anything separately persisted -- so "start from
Phase 3" means loading whichever condensed-text files already exist on
disk and reconstructing that dict, not literally re-entering run_phase2.
"""

import json
import re

from common.paths import REPO_ROOT, CorpusPaths

START_PHASES = (1, 2, 3)


def find_last_decision_choice(corpus_name: str, phase: str, decision_type: str) -> str | None:
    """Scans an existing phase decision log for the most recent recorded
    choice of the given decision_type -- used when resuming a corpus at
    Phase 2/3 to recover config (e.g. which spaCy lang_model Phase 1 was
    actually run with) that isn't otherwise persisted anywhere except the
    decision log itself. Returns None if the log doesn't exist or has no
    such entry (e.g. a run from before phase1_pipeline.log_phase1_config
    started logging it) -- callers fall back to a CLI/webapp default."""
    path = CorpusPaths(corpus_name).decisions_log(phase)
    if not path.exists():
        return None
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("decision_type") == decision_type:
                last = entry.get("choice")
    return last


def list_corpora() -> list[str]:
    """Every corpus directory under data/, sorted -- for a "pick an
    existing dataset" UI/CLI listing. Returns [] if data/ doesn't exist
    yet (a fresh checkout before any run)."""
    data_dir = REPO_ROOT / "data"
    if not data_dir.exists():
        return []
    return sorted(p.name for p in data_dir.iterdir() if p.is_dir())


def available_condensation_rates(paths: CorpusPaths) -> list[int]:
    """Rates that already have a saved condensed-text file, discovered
    directly from disk -- not from any config or decision log, since a
    resume point may be well after whoever ran Phase 2 last remembers
    which rates they generated."""
    cond_dir = paths.condensation_dir()
    if not cond_dir.exists():
        return []
    pattern = re.compile(rf"^{re.escape(paths.corpus_name)}-condensed-(\d+)pct\.txt$")
    rates = []
    for f in cond_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            rates.append(int(m.group(1)))
    return sorted(rates)


def check_prerequisites(paths: CorpusPaths, start_phase: int) -> tuple[bool, list[str], list[int]]:
    """Returns (ok, missing, available_rates).

    `missing` is a human-readable description of every absent required
    file/artifact for starting at `start_phase` -- empty iff `ok`.
    `available_rates` is only ever populated when start_phase == 3 (the
    rates discovered as ready for it); callers use it to know which rates
    Phase 3 will actually run against.

    start_phase == 1 has no prerequisites here -- that path creates
    data/<corpus>/raw/<corpus>.txt itself from --input, same as today."""
    if start_phase not in START_PHASES:
        raise ValueError(f"start_phase must be one of {START_PHASES}, got {start_phase}")

    missing = []
    available_rates = []

    if start_phase >= 2:
        if not paths.raw_txt().exists():
            missing.append(f"raw corpus text ({paths.raw_txt()})")
        if not paths.phase1_state_json().exists():
            missing.append(f"Phase 1 state ({paths.phase1_state_json()})")

    if start_phase >= 3:
        available_rates = available_condensation_rates(paths)
        if not available_rates:
            missing.append(
                f"at least one condensed rate under {paths.condensation_dir()} "
                "(Phase 2 must have generated at least one condensation first)"
            )

    return (len(missing) == 0, missing, available_rates)


def corpus_phase_summary(corpus_name: str) -> dict:
    """{corpus_name, available_start_phases: [...], rates: [...]} -- the
    highest start_phase a corpus could resume at is implied by which
    numbers appear in available_start_phases; used by both the CLI's
    --list-corpora and the webapp's corpus-picker endpoint so a researcher
    can see what's actually resumable before picking a phase that will
    just fail prerequisite-checking."""
    paths = CorpusPaths(corpus_name)
    available_start_phases = []
    rates: list[int] = []
    for phase in START_PHASES:
        ok, _, phase_rates = check_prerequisites(paths, phase)
        if ok:
            available_start_phases.append(phase)
        if phase == 3:
            rates = phase_rates
    return {
        "corpus_name": corpus_name,
        "available_start_phases": available_start_phases,
        "rates": rates,
    }
