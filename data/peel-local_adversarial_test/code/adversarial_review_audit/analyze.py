#!/usr/bin/env python3
"""Classifies every audit call and reports outcome rates with 95% Wilson
intervals, per condition and design. Stdlib only.

    python experiments/adversarial_review_audit/analyze.py
    python experiments/adversarial_review_audit/analyze.py results/fixed_input.jsonl results/pipeline.jsonl

The repository root defaults to two levels up; set PEEL_REPO to override.

Writes, next to the inputs:
    summary_categories.csv   one row per (design, condition, category)
    summary_groups.csv       one row per (design, condition, group)
    summary_similarity.csv   near-identical shares at several thresholds
    summary.md               the same, readable, plus run diagnostics

Every classified call is cross-checked against the decision the real
pipeline recorded for it (`pipeline_decision`); any disagreement is reported
and makes the script exit non-zero, because it would mean the taxonomy does
not describe what the pipeline did.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audit  # noqa: E402  (CONDITIONS, load_condition, DEFAULT_REPO)

Z95 = 1.959963984540054
SIMILARITY_THRESHOLDS = (0.999, 0.99, 0.95)


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def load(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    paths = [Path(a) for a in argv] or sorted((HERE / "results").glob("*.jsonl"))
    if not paths:
        raise SystemExit("no results/*.jsonl found")
    repo = Path(os.environ.get("PEEL_REPO", audit.DEFAULT_REPO)).resolve()
    sys.path.insert(0, str(repo))
    from phase2 import condense  # noqa: E402
    from classify import CATEGORIES, GROUPS, classify_call  # noqa: E402

    reviewed_by_cond = {}
    rows = load(paths)
    classified = defaultdict(list)       # (mode, condition) -> [classification]
    no_condensation = Counter()
    diag = defaultdict(lambda: defaultdict(list))
    mismatches = []

    for r in rows:
        key = (r["mode"], r["condition"])
        if r.get("no_condensation"):
            no_condensation[key] += 1
            continue
        text = r.get("reviewed_text")
        if text is None:
            if r["condition"] not in reviewed_by_cond:
                reviewed_by_cond[r["condition"]] = audit.load_condition(repo, r["condition"], condense)
            text = reviewed_by_cond[r["condition"]]["paper_reviewed_text"]
        c = classify_call(r.get("raw_response"), r.get("call_error"), text, r["target_words"])
        if c["pipeline_choice"] != r.get("pipeline_decision"):
            mismatches.append((key, r.get("replicate"), c["category"], r.get("pipeline_decision")))
        c["tag"] = r.get("tag") or "local"
        o, q = r.get("ollama") or {}, r.get("request") or {}
        c["ctx_used"] = (o.get("prompt_eval_count") or 0) + (o.get("eval_count") or 0)
        c["done_reason"] = o.get("done_reason")
        c["elapsed_s"] = q.get("elapsed_s")
        classified[key].append(c)
        diag[key]["done_reason"].append(o.get("done_reason"))
        for f in ("prompt_eval_count", "eval_count"):
            if o.get(f) is not None:
                diag[key][f].append(o[f])
        if q.get("elapsed_s") is not None:
            diag[key]["elapsed_s"].append(q["elapsed_s"])
        for m in r.get("loaded_models") or []:
            if isinstance(m, dict) and m.get("name") == r["model"] and m.get("context_length"):
                diag[key]["context_length"].append(m["context_length"])

    out_dir = paths[0].parent
    cat_rows, grp_rows, sim_rows = [], [], []
    for key in sorted(classified):
        cs = classified[key]
        n = len(cs)
        cats, grps = Counter(c["category"] for c in cs), Counter(c["group"] for c in cs)
        for name in CATEGORIES:
            lo, hi = wilson(cats[name], n)
            cat_rows.append({"design": key[0], "condition": key[1], "category": name, "k": cats[name],
                             "n": n, "share": cats[name] / n, "ci95_lo": lo, "ci95_hi": hi})
        for name in GROUPS:
            lo, hi = wilson(grps[name], n)
            grp_rows.append({"design": key[0], "condition": key[1], "group": name, "k": grps[name],
                             "n": n, "share": grps[name] / n, "ci95_lo": lo, "ci95_hi": hi})
        # "Same text" under a looser reading: accepted FIXED texts at or
        # above a word-similarity threshold, identical ones included.
        accepted = [c for c in cs if c["pipeline_choice"] == "fixed"]
        for t in SIMILARITY_THRESHOLDS:
            k = sum(1 for c in accepted if c["similarity_ratio"] is not None and c["similarity_ratio"] >= t)
            lo, hi = wilson(k, n)
            sim_rows.append({"design": key[0], "condition": key[1], "threshold": t, "k": k, "n": n,
                             "share_of_calls": k / n, "ci95_lo": lo, "ci95_hi": hi,
                             "n_accepted_fixed": len(accepted)})

    def write_csv(name, rows_):
        if rows_:
            with (out_dir / name).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_[0]))
                w.writeheader()
                w.writerows(rows_)

    # Length of every returned FIXED text relative to what was reviewed.
    len_rows, tag_rows, integ_rows = [], [], []
    for key in sorted(classified):
        cs = classified[key]
        ratios = sorted(c["fixed_words"] / c["reviewed_words"] for c in cs
                        if c["fixed_words"] is not None and c["reviewed_words"])
        if ratios:
            q = statistics.quantiles(ratios, n=20, method="inclusive") if len(ratios) > 1 else ratios * 19
            len_rows.append({"design": key[0], "condition": key[1], "n_fixed_with_text": len(ratios),
                             "min": ratios[0], "p05": q[0], "p25": q[4], "median": q[9], "p75": q[14],
                             "p95": q[18], "max": ratios[-1],
                             "share_above_1.05": sum(x > 1.05 for x in ratios) / len(ratios),
                             "share_below_0.95": sum(x < 0.95 for x in ratios) / len(ratios)})
        for tag in sorted({c["tag"] for c in cs}):
            sub = [c for c in cs if c["tag"] == tag]
            g = Counter(c["group"] for c in sub)
            tag_rows.append({"design": key[0], "condition": key[1], "tag": tag, "n": len(sub),
                             **{f"k_{name}": g[name] for name in GROUPS}})
        integ_rows.append({"design": key[0], "condition": key[1], "n": len(cs),
                           "max_ctx_used": max(c["ctx_used"] for c in cs),
                           "n_done_reason_not_stop": sum(1 for c in cs if c["category"] != "call_failed"
                                                         and c["done_reason"] != "stop"),
                           "n_ctx_used_ge_32768": sum(1 for c in cs if c["ctx_used"] >= 32768)})
    write_csv("summary_length.csv", len_rows)
    write_csv("summary_by_tag.csv", tag_rows)
    write_csv("summary_integrity.csv", integ_rows)

    write_csv("summary_categories.csv", cat_rows)
    write_csv("summary_groups.csv", grp_rows)
    write_csv("summary_similarity.csv", sim_rows)

    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    md = ["# Adversarial-review audit -- summary", "",
          f"Inputs: {', '.join(p.name for p in paths)}. Intervals are 95% Wilson, per cell.", ""]
    if mismatches:
        md += [f"**{len(mismatches)} classification/pipeline mismatches** -- see stderr.", ""]
    for key in sorted(classified):
        n = len(classified[key])
        md += [f"## {key[0]} / {key[1]}  (n = {n}"
               + (f"; {no_condensation[key]} replicate(s) produced no condensation" if no_condensation[key] else "")
               + ")", "", "| group | k | share | 95% CI |", "|---|---:|---:|---|"]
        md += [f"| {g['group']} | {g['k']} | {pct(g['share'])} | {pct(g['ci95_lo'])}–{pct(g['ci95_hi'])} |"
               for g in grp_rows if (g["design"], g["condition"]) == key]
        md += ["", "| category | k | share | 95% CI |", "|---|---:|---:|---|"]
        md += [f"| {c['category']} | {c['k']} | {pct(c['share'])} | {pct(c['ci95_lo'])}–{pct(c['ci95_hi'])} |"
               for c in cat_rows if (c["design"], c["condition"]) == key and c["k"]]
        md += ["", "| accepted FIXED with word similarity ≥ | k | share of calls | 95% CI |", "|---|---:|---:|---|"]
        md += [f"| {s['threshold']} | {s['k']} | {pct(s['share_of_calls'])} | "
               f"{pct(s['ci95_lo'])}–{pct(s['ci95_hi'])} |"
               for s in sim_rows if (s["design"], s["condition"]) == key]
        d = diag[key]
        med = lambda v: f"{statistics.median(v):.0f}" if v else "n/a"  # noqa: E731
        md += ["", f"Diagnostics: done_reason {dict(Counter(d['done_reason']))}; "
               f"median prompt_eval_count {med(d['prompt_eval_count'])}; median eval_count {med(d['eval_count'])}; "
               f"median call time {med(d['elapsed_s'])} s; loaded context_length "
               f"{sorted(set(d['context_length'])) or 'n/a'}.", ""]
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote summary_*.csv and summary.md to {out_dir}")

    if mismatches:
        for m in mismatches:
            print("MISMATCH", m, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
