"""Runs this job's audit shards concurrently against the local Ollama.

The plan arrives as the SageMaker hyperparameter "plan" (a JSON string):
    {"tag": "g5x", "max_runtime_s": 5400, "margin_s": 900,
     "slots": [{"mode": "fixed-input", "condition": "qwen9b-20", "n": 40}, ...]}
Each slot is one audit.py process with its own output file, so shards never
share a writer. No replicate is started after max_runtime - margin, which
leaves room for the call in flight and the final S3 sync.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

START = time.time()
hp = json.load(open("/opt/ml/input/config/hyperparameters.json"))
plan = json.loads(hp["plan"])
out = Path("/opt/ml/checkpoints")
deadline = START + plan["max_runtime_s"] - plan["margin_s"]
(out / "plan.json").write_text(json.dumps(plan, indent=1))

procs = []
for k, slot in enumerate(plan["slots"]):
    name = f"{plan['tag']}_slot{k}_{slot['mode'].replace('-', '_')}_{slot['condition']}"
    cmd = [sys.executable, "/opt/peel/experiments/adversarial_review_audit/audit.py", slot["mode"],
           "--n", str(slot["n"]), "--conditions", slot["condition"], "--repo", "/opt/peel",
           "--host", "http://127.0.0.1:11434", "--out", str(out / f"{name}.jsonl"),
           "--tag", plan["tag"], "--deadline-epoch", str(deadline)]
    log = open(out / f"{name}.log", "w")
    procs.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    print("started", name, flush=True)

codes = {}
for name, p, log in procs:
    codes[name] = p.wait()
    log.close()
(out / "driver_done.json").write_text(json.dumps({"exit_codes": codes, "elapsed_s": time.time() - START}))
print("driver done", codes, flush=True)
sys.exit(0 if all(c == 0 for c in codes.values()) else 1)
