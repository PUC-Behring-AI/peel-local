#!/usr/bin/env python3
"""Checks the two documentation claims this repository cannot afford to get
wrong, because the paper cites them as reproducibility artifacts (§2.8).

1. Every `file.py::symbol` reference in docs/ resolves to a symbol actually
   defined in that file.

   These used to be `file.py:LINE` references, and all twelve of them were
   wrong -- off by anywhere from 1 to 85 lines, because a line number rots on
   the next edit while nothing checks it. `glossbert_predict` was cited at
   `pipeline.py:111`, which is a constant assignment. A reader following the
   reference to "determine exactly what a model was asked to do", which is
   what the catalogue promises, landed somewhere else.

2. Every public function in the codebase appears in FUNCTIONS.md, which the
   paper describes as "a complete function-by-function call graph". It was
   missing 25 of them, including the entire webapp/ layer -- 1,762 lines.

3. Every LIMITATIONS.md entry names its issue, and no document promises a
   "tracking issue" without a number.

   Before #2, README.md, LIMITATIONS.md and all three PROVENANCE.md files
   pointed readers at tracking issues while the repository had none. A
   reviewer checks that claim in one click. The check is structural (it
   does not call the GitHub API), so it runs offline and in CI.

Run via `make check`.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories whose functions FUNCTIONS.md is expected to document.
DOCUMENTED_PACKAGES = ("common", "phase0", "phase1", "phase2", "phase3", "webapp")

# Not part of the documented surface: the test suite documents itself, and
# scripts/ are gate tooling rather than pipeline API. Matched against paths
# RELATIVE to the repo root -- matching absolute parts silently skipped every
# file when the repo was checked out under a directory with one of these
# names (a git worktree under .claude/worktrees/ did exactly that).
SKIP_DIRS = {"tests", "scripts", "docgraph"}

SYMBOL_REF = re.compile(r"`([a-z0-9_]+(?:/[a-z0-9_]+)*\.py)::([A-Za-z_][A-Za-z0-9_]*)`")


def _defined_symbols(py_path: Path) -> set[str]:
    """Every top-level and class-level def/class name in a file."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def check_symbol_references() -> list[str]:
    problems = []
    for doc in sorted(REPO_ROOT.glob("docs/*.md")):
        text = doc.read_text(encoding="utf-8")
        for rel_path, symbol in SYMBOL_REF.findall(text):
            target = REPO_ROOT / rel_path
            if not target.exists():
                problems.append(f"{doc.name}: references {rel_path}, which does not exist")
                continue
            if symbol not in _defined_symbols(target):
                problems.append(
                    f"{doc.name}: references {rel_path}::{symbol}, "
                    f"but {rel_path} defines no such symbol"
                )
    return problems


def _is_documented(name: str, documented: str) -> bool:
    """Whether FUNCTIONS.md mentions `name` as a whole symbol.

    A plain substring test is too loose to be useful here: `start` would match
    inside `start_regeneration`, and `status` inside `status_dict`, so a
    genuinely undocumented function could pass by being a prefix of a
    documented one. Requires the name to be followed by a non-identifier
    character.
    """
    return re.search(rf"`\.?{re.escape(name)}(?![A-Za-z0-9_])", documented) is not None


def check_functions_md_coverage() -> list[str]:
    documented = (REPO_ROOT / "FUNCTIONS.md").read_text(encoding="utf-8")

    undocumented = []
    for package in DOCUMENTED_PACKAGES:
        for py_path in sorted((REPO_ROOT / package).rglob("*.py")):
            if any(part in SKIP_DIRS for part in py_path.relative_to(REPO_ROOT).parts):
                continue
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name.startswith("_"):
                    continue
                if not _is_documented(node.name, documented):
                    rel = py_path.relative_to(REPO_ROOT)
                    undocumented.append(f"{rel}::{node.name}")

    for py_path in sorted(REPO_ROOT.glob("*.py")):
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                if not _is_documented(node.name, documented):
                    undocumented.append(f"{py_path.name}::{node.name}")

    if not undocumented:
        return []
    return [
        f"FUNCTIONS.md does not mention {len(undocumented)} public function(s): "
        + ", ".join(undocumented)
    ]


LIMITATION_HEADING = re.compile(r"^## (L\d+) — ", re.M)
ISSUE_LINE = re.compile(r"^\*\*Issue\*\* \[#\d+\]\(https://github\.com/[^)]+/issues/\d+\)$", re.M)

# Documents a reader of the paper lands on. A dangling "tracking issue" in
# any of them is the claim this check exists to keep honest.
READER_DOCS = ("README.md", "LIMITATIONS.md", "CONTRIBUTING.md", "NOTICE")


def check_issue_references() -> list[str]:
    problems = []

    text = (REPO_ROOT / "LIMITATIONS.md").read_text(encoding="utf-8")
    headings = list(LIMITATION_HEADING.finditer(text))
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        if not ISSUE_LINE.search(text, heading.end(), end):
            problems.append(f"LIMITATIONS.md {heading.group(1)} has no **Issue** [#NN](...) line")

    docs = [REPO_ROOT / name for name in READER_DOCS]
    docs += sorted((REPO_ROOT / "data").glob("*/PROVENANCE.md"))
    for path in docs:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "tracking issue" in line.lower() and not re.search(r"#\d+", line):
                rel = path.relative_to(REPO_ROOT)
                problems.append(f"{rel}:{lineno} promises a tracking issue without naming it")
    return problems


def main() -> int:
    problems = check_symbol_references() + check_functions_md_coverage() + check_issue_references()
    if problems:
        print("Documentation check failed:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nFix the documentation, not this check: the paper's §2.8 presents "
            "these two files as reproducibility artifacts.",
            file=sys.stderr,
        )
        return 1
    print("Documentation check passed: every docs/ symbol reference resolves, "
          "FUNCTIONS.md covers every public function, and every issue reference "
          "names its issue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
