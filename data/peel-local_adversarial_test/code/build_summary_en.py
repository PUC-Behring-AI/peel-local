"""Builds RESULTS.en.md (English) from the audit's CSV/JSONL outputs.

English translation of build_summary.py, which builds the Portuguese
RESULTS.md. No number in the report is typed by hand: every figure is read
from the analysis tables written by code/adversarial_review_audit/analyze.py.

    python build_summary_en.py
"""
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

from scipy.stats import chi2_contingency

HERE = Path(__file__).resolve().parent
# Worktree used to run analyze.py's original SageMaker jobs; not portable
# as-is, set PEEL_REPO_WORKTREE to your own path to re-run this script.
WT = Path(os.environ.get("PEEL_REPO_WORKTREE", "/path/to/peel-local/.claude/worktrees/release-hardening"))
PRIMARY = HERE / "results" / "aws"
MAC = HERE / "results" / "mac_pilot"
PRICE = {"ml.g4dn.xlarge": 1.252, "ml.g4dn.2xlarge": 1.598, "ml.g5.xlarge": 2.1375,
         "ml.g5.2xlarge": 2.5752, "ml.g6.xlarge": 1.915, "ml.g6.2xlarge": 2.077}
COND_LABEL = {"qwen2b-20": "qwen2b · 20 sent.", "qwen9b-10": "qwen9b · 10 sent.", "qwen9b-20": "qwen9b · 20 sent."}
GROUP_LABEL = {"no_output": "no output", "same_text": "same text (exact)", "expanded": "expanded",
               "ok": "OK", "changed": "changed"}
CAT_LABEL = {"call_failed": "call failure/timeout", "empty_response": "empty response",
             "unparseable": "no `VERDICT`", "fixed_no_text": "FIXED with no `TEXT`",
             "fixed_rejected_short": "FIXED rejected by L2", "ok": "OK",
             "fixed_identical": "FIXED identical", "fixed_expanded": "FIXED expanded",
             "fixed_changed": "FIXED changed"}


def pct(x):
    return f"{100 * float(x):.1f}%"


def num(x, d=2):
    return f"{float(x):.{d}f}"


def ci(r):
    return f"{pct(r['ci95_lo'])}–{pct(r['ci95_hi'])}"


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_analysis(out_dir, jsonls):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PEEL_REPO=str(WT))
    subprocess.run([sys.executable, str(HERE / "adversarial_review_audit/analyze.py"), *map(str, jsonls)],
                   check=True, env=env)


def main():
    # ---- assemble inputs -------------------------------------------------
    PRIMARY.mkdir(parents=True, exist_ok=True)
    MAC.mkdir(parents=True, exist_ok=True)
    jobs = json.load(open(HERE / "aws_jobs" / "handoff_jobs.json"))["jobs"]
    merged = PRIMARY / "fixed_input_aws.jsonl"
    with merged.open("w", encoding="utf-8") as out:
        for j in jobs:
            for f in sorted(glob.glob(str(HERE / "aws_results" / j["name"] / "*.jsonl"))):
                out.write(Path(f).read_text(encoding="utf-8"))
    (MAC / "fixed_input_mac.jsonl").write_text((HERE / "results" / "fixed_input.jsonl").read_text(encoding="utf-8"),
                                               encoding="utf-8")
    run_analysis(PRIMARY, [merged])
    run_analysis(MAC, [MAC / "fixed_input_mac.jsonl"])

    groups, cats = read(PRIMARY / "summary_groups.csv"), read(PRIMARY / "summary_categories.csv")
    sims, lens = read(PRIMARY / "summary_similarity.csv"), read(PRIMARY / "summary_length.csv")
    integ, bytag = read(PRIMARY / "summary_integrity.csv"), read(PRIMARY / "summary_by_tag.csv")
    mac_groups = read(MAC / "summary_groups.csv")
    conds = sorted({r["condition"] for r in groups})

    def g(rows, cond, name, field="group"):
        return next(r for r in rows if r["condition"] == cond and r[field] == name)

    # ---- environment / cost ---------------------------------------------
    env_files = sorted(glob.glob(str(HERE / "aws_results" / "*" / "env_*.json")))
    env0 = json.load(open(env_files[0]))
    digests = {m["name"]: m["digest"] for m in env0["tags"]["models"] if m["name"].startswith("qwen3.5")}
    gpus = sorted({json.load(open(f)).get("gpu") or "?" for f in env_files})
    sys.path.append(str(HERE / "aws_jobs"))
    import sm_monitor as m  # noqa: E402
    cost_rows, total_cost, total_bill = [], 0.0, 0
    for j in jobs:
        d = m.sm.describe_training_job(TrainingJobName=j["name"])
        bill = d.get("BillableTimeInSeconds") or 0
        c = bill / 3600 * PRICE[j["type"]]
        total_cost += c
        total_bill += bill
        cost_rows.append((j["stage"], j["type"], d["TrainingJobStatus"], bill, c))

    # ---- hardware homogeneity (chi-square on groups, per condition) -------
    # All machines: many cells hold n = 2 (pilot), so the chi-square
    # approximation is invalid; use a permutation p-value for the same statistic.
    import numpy as np
    rng = np.random.default_rng(20260928)
    hw_tests = []
    for cond in conds:
        rows = [r for r in bytag if r["condition"] == cond]
        labels, tags = [], []
        for ti, r in enumerate(rows):
            for gi, name in enumerate(GROUP_LABEL):
                labels += [gi] * int(r[f"k_{name}"])
                tags += [ti] * int(r[f"k_{name}"])
        labels, tags = np.array(labels), np.array(tags)

        def stat(lab):
            t = np.zeros((len(rows), len(GROUP_LABEL)))
            np.add.at(t, (tags, lab), 1)
            t = t[:, t.sum(0) > 0]
            e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
            return ((t - e) ** 2 / e).sum()
        obs = stat(labels)
        perm = np.array([stat(rng.permutation(labels)) for _ in range(10000)])
        hw_tests.append((cond, len(rows), len(labels), obs, float((1 + (perm >= obs).sum()) / (1 + len(perm)))))
    main_tags = [r for r in bytag if r["tag"].startswith("main-")]
    main_tests = []
    for cond in conds:
        rows = [r for r in main_tags if r["condition"] == cond]
        table = [[int(r[f"k_{name}"]) for name in GROUP_LABEL] for r in rows]
        cols = [i for i in range(len(GROUP_LABEL)) if sum(t[i] for t in table) > 0]
        table = [[t[i] for i in cols] for t in table]
        if len(table) == 2 and len(cols) > 1:
            chi2, p, dof, _ = chi2_contingency(table)
            main_tests.append((cond, chi2, dof, p))

    # ---- write -------------------------------------------------------------
    L = []
    w = L.append
    w("# Adversarial-reviewer audit (Phase 2) — results summary\n")
    w("*(English translation of [`RESULTS.md`](../RESULTS.md); the original Portuguese is the file of record.)*\n")
    w("Working document for deciding the text of the Limitations section and §4.3. No number here was typed by hand: "
      "everything comes from `results/aws/summary_*.csv` and `results/mac_pilot/summary_*.csv`, generated by "
      "`code/adversarial_review_audit/analyze.py`.\n")

    w("## 1. Question\n")
    w("How often does the adversarial reviewer (a) deliver nothing usable to the researcher, or (b) return the exact "
      "same text it was given, across the paper's three configurations? The design measures the **reviewer in "
      "isolation**: each call receives exactly the condensation the paper's run actually sent it (read from the "
      "decision log), and only the model's own sampling varies.\n")

    w("## 2. Protocol\n")
    n_by = {c: int(g(groups, c, "no_output")["n"]) for c in conds}
    w("| Item | Value |\n|---|---|")
    w(f"| Calls analyzed (AWS) | {sum(n_by.values())} ({', '.join(f'{COND_LABEL[c]}: {n_by[c]}' for c in conds)}) |")
    w(f"| Models | {', '.join(f'`{k}` digest `{v[:12]}`' for k, v in sorted(digests.items()))} — verified on every job |")
    w(f"| Ollama | {env0['ollama_version'].get('version')} |")
    w(f"| GPUs | {'; '.join(gpus)} |")
    w("| Context | `OLLAMA_CONTEXT_LENGTH=32768` (see §7) |")
    w("| Call | the pipeline's own: real `run_adversarial_review`, no `temperature`/`seed`, `think=False`, 600 s timeout |")
    w("| Sampling | Modelfile parameters: temperature 1, top_k 20, top_p 0.95, presence_penalty 1.5 |")
    w("| Intervals | Wilson 95%, per cell |\n")
    w("**Weight identity.** The models were downloaded on 2026-09-28, the day of the pilot. The paper's own runs "
      "(2026-09-09) recorded no digest, so **it cannot be stated that the weights measured here are the same ones "
      "used in the paper**. The rates below are for the digests above.\n")

    w("## 3. Main result\n")
    w("| Configuration | n | no output | same text (exact) | expanded | OK | changed |\n|---|---:|---|---|---|---|---|")
    for c in conds:
        cells = [f"{pct(g(groups, c, k)['share'])} ({ci(g(groups, c, k))})" for k in GROUP_LABEL]
        w(f"| {COND_LABEL[c]} | {n_by[c]} | " + " | ".join(cells) + " |")
    w("\n*No output* = the researcher gets nothing from the reviewer and the log records `unchanged` (the sum of the "
      "categories in §4). *Same text* = the log records **`fixed`**, but the text is what the reviewer was given. "
      "*Expanded* = an accepted correction longer than 1.05 × max(reviewed text, target), PEEL's own length "
      "tolerance.\n")

    w("## 4. Breaking down \"no output\"\n")
    w("| Configuration | " + " | ".join(CAT_LABEL[k] for k in CAT_LABEL if k in
      ("call_failed", "empty_response", "unparseable", "fixed_no_text", "fixed_rejected_short")) + " |")
    w("|---|" + "---|" * 5)
    for c in conds:
        cells = [f"{g(cats, c, k, 'category')['k']} · {pct(g(cats, c, k, 'category')['share'])} ({ci(g(cats, c, k, 'category'))})"
                 for k in ("call_failed", "empty_response", "unparseable", "fixed_no_text", "fixed_rejected_short")]
        w(f"| {COND_LABEL[c]} | " + " | ".join(cells) + " |")
    w("\n`FIXED rejected by L2` is a **code** defect (the model did produce text; the pipeline discarded it for "
      "falling under the 0.5 × target threshold). The other four are **model** or **call** defects. "
      "`failure/timeout` depends on the hardware (see §7).\n")
    w("Every call failure was a **600 s timeout** (the pipeline's own limit), on the main run on A10G, with the 2b "
      "model sharing the GPU with the 9b model. The share likely depends on machine speed (not tested separately).\n")

    w("## 5. \"Same text\" under every reading\n")
    w("Share of calls whose **accepted** correction has word similarity to the reviewed text ≥ threshold (includes "
      "identical ones).\n")
    w("| Configuration | exact | ≥ 0.999 | ≥ 0.99 | ≥ 0.95 |\n|---|---|---|---|---|")
    for c in conds:
        ex = g(groups, c, "same_text")
        cells = [f"{pct(ex['share'])} ({ci(ex)})"]
        for t in ("0.999", "0.99", "0.95"):
            r = next(r for r in sims if r["condition"] == c and r["threshold"] == t)
            cells.append(f"{pct(r['share_of_calls'])} ({ci(r)})")
        w(f"| {COND_LABEL[c]} | " + " | ".join(cells) + " |")
    w("")

    w("## 6. Length of the corrections returned\n")
    w("Ratio of correction word count to reviewed-text word count, for every FIXED response that included text "
      "(accepted or rejected).\n")
    w("| Configuration | n | min | p05 | median | p95 | max | > 1.05 | < 0.95 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in lens:
        w(f"| {COND_LABEL[r['condition']]} | {r['n_fixed_with_text']} | {num(r['min'])} | {num(r['p05'])} | "
          f"{num(r['median'])} | {num(r['p95'])} | {num(r['max'])} | {pct(r['share_above_1.05'])} | "
          f"{pct(r['share_below_0.95'])} |")
    w("")

    w("## 7. Integrity and robustness\n")
    for r in integ:
        w(f"- {COND_LABEL[r['condition']]}: largest context used {r['max_ctx_used']} tokens (limit 32,768); "
          f"{r['n_ctx_used_ge_32768']} calls at the limit; {r['n_done_reason_not_stop']} with `done_reason` ≠ `stop`.")
    w("- Classification × the pipeline's own recorded decision: `analyze.py` exits with an error on any disagreement; "
      "it exited cleanly.")
    if main_tests:
        w("- Same GPU (A10G), different machines on the main run — chi-squared test of homogeneity across groups:")
        for cond, chi2, dof, p in main_tests:
            w(f"  - {COND_LABEL[cond]}: χ² = {num(chi2)}, df = {dof}, p = {num(p, 3)}")
    if hw_tests:
        w("- All machines (pilot across 6 types + main run):")
        for cond, k, n, chi2, p in hw_tests:
            w(f"  - {COND_LABEL[cond]}: {k} machines, n = {n}, χ² = {num(chi2)}, permutation p = {num(p, 3)} "
              "(10,000 permutations; the pilot has n = 2 per machine, so the test has little power for those machines)")
    w("- Mac pilot (Apple Metal, context 131,072, n = 2 per configuration), for qualitative comparison only:")
    for c in conds:
        cells = ", ".join(f"{GROUP_LABEL[k]} {g(mac_groups, c, k)['k']}" for k in GROUP_LABEL if int(g(mac_groups, c, k)["k"]))
        w(f"  - {COND_LABEL[c]}: {cells}")
    w("")

    w("## 8. What this measurement does not cover\n")
    w("- **Three input texts.** The fixed-input design measures the reviewer on the paper's three condensations; "
      "the rate on other texts was not estimated. The full-pipeline test (30 replicates) did not fit the US$30 "
      "budget and was not run.")
    w("- **Semantic correctness.** `OK` and `changed` are formal classifications. Whether an OK verdict was actually "
      "right, or whether a change actually fixed anything, was not evaluated.")
    w("- **The paper's own weights.** See §2.")
    w("- **Engine.** CUDA (AWS) instead of the Mac pilot's Metal; weights identical to the pilot's. The machine the "
      "paper's own runs used was not recorded.\n")

    w("## 9. Cost and time\n")
    w("| Step | Instance | Status | Billed (s) | US$ |\n|---|---|---|---:|---:|")
    for stage, it, st_, bill, c in cost_rows:
        w(f"| {stage} | {it} | {st_} | {bill} | {num(c)} |")
    w(f"| **total** | | | **{total_bill}** | **{num(total_cost)}** |\n")
    w("Prices: AWS public list, SageMaker Training, sa-east-1.\n")

    w("## 10. Files\n")
    w("- `results/aws/fixed_input_aws.jsonl` — every AWS call, with the model's raw response")
    w("- `results/aws/summary_*.csv` — the tables the numbers above come from")
    w("- `results/mac_pilot/` — the Mac pilot")
    w("- `aws_results/<job>/env_*.json` — each job's environment (versions, digests, GPU)")
    (HERE.parent / "RESULTS.en.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", HERE.parent / "RESULTS.en.md", f"total cost US${total_cost:.2f}")


if __name__ == "__main__":
    main()
