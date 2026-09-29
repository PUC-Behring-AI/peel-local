"""Aborts the job unless the pulled models are byte-identical (same Ollama
digest) to the ones on the pilot host. Expected digests are read from the
pilot's environment snapshot, shipped in the bundle."""
import glob
import json
import sys

import requests

expected = {}
for path in glob.glob("/opt/peel/pilot_env/env_*.json"):
    for m in json.load(open(path))["tags"]["models"]:
        if m["name"] in ("qwen3.5:2b", "qwen3.5:9b"):
            expected[m["name"]] = m["digest"]
got = {m["name"]: m["digest"] for m in requests.get("http://127.0.0.1:11434/api/tags", timeout=10).json()["models"]}
ok = True
for name, digest in sorted(expected.items()):
    same = got.get(name) == digest
    ok &= same
    print(f"{name}: expected {digest[:12]} got {str(got.get(name))[:12]} -> {'OK' if same else 'MISMATCH'}")
if len(expected) != 2 or not ok:
    sys.exit("model digests do not match the pilot host; refusing to run")
