"""Shared data-directory contract for all PEEL-Local phases.

Every notebook derives its I/O paths from a single CORPUS_NAME via
CorpusPaths, so phases agree on where to read/write without any manual
copying of files between phase folders.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class CorpusPaths:
    def __init__(self, corpus_name: str):
        self.corpus_name = corpus_name
        self.root = REPO_ROOT / "data" / corpus_name
        self.raw_dir = self.root / "raw"
        self.phase1_dir = self.root / "phase1"
        self.phase2_dir = self.root / "phase2"
        self.decisions_dir = self.root / "decisions"

    def ensure_dirs(self):
        for d in (self.raw_dir, self.phase1_dir, self.phase2_dir,
                  self.decisions_dir, self.condensation_dir()):
            d.mkdir(parents=True, exist_ok=True)

    def raw_txt(self) -> Path:
        return self.raw_dir / f"{self.corpus_name}.txt"

    def top_stems(self) -> Path:
        return self.phase1_dir / "top_stems.txt"

    def glossbert_output(self) -> Path:
        return self.phase1_dir / "glossbert_accepted_terms.txt"

    def phase1_state_json(self) -> Path:
        return self.phase1_dir / f"{self.corpus_name}-phase1_state.json"

    def phase1_html(self) -> Path:
        return self.phase1_dir / f"{self.corpus_name}-Phase1-clusters.html"

    def phase2_output_json(self) -> Path:
        return self.phase2_dir / "informative_sentences.json"

    def decisions_log(self, phase: str) -> Path:
        return self.decisions_dir / f"{phase}_decisions.jsonl"

    def condensation_dir(self) -> Path:
        return self.phase2_dir / "condensation"

    def condensation_paths(self, rate: int) -> dict:
        base = self.condensation_dir()
        stem = f"{self.corpus_name}-condensed-{rate}pct"
        return {
            "condensed_text": base / f"{stem}.txt",
            "injection_report": base / f"{stem}-injection-report.json",
            "html_fragment": base / f"{stem}-fragment.html",
            "html_preview": base / f"{stem}-preview.html",
            "human_report": base / f"{stem}-report.txt",
            "plain_summary": base / f"{stem}-summary.txt",
        }

    def voyant_notebook_path(self, rate: int) -> Path:
        return self.condensation_dir() / f"{self.corpus_name}-voyant-notebook-{rate}pct.html"

    def standalone_report_path(self, rate: int) -> Path:
        return self.condensation_dir() / f"{self.corpus_name}-standalone-report-{rate}pct.html"


def resources_dir() -> Path:
    return REPO_ROOT / "resources"
