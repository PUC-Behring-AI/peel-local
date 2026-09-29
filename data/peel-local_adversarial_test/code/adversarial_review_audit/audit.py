#!/usr/bin/env python3
"""Measures how often Phase 2's adversarial reviewer returns nothing usable,
or returns the text it was given, on the paper's three configurations.

Nothing in the pipeline is modified or reimplemented. The reviewer is the
real `phase2/condense.py::run_adversarial_review` and generation is the real
`attempt_condensation_trials`; the only intervention is a recording wrapper
around `requests.post` inside `common/ollama_client.py`, so the raw Ollama
response -- which the pipeline itself never logs -- is kept alongside what
the pipeline decided. Nothing is written under `data/`.

Two designs (see README.md):

  fixed-input   the reviewer alone, N times, on the exact condensation each
                paper run sent to it (read from that run's decision log).
                Isolates the reviewer; only sampling noise varies.
  pipeline      generation + review end to end, N times per configuration,
                with the paper runs' own escalation choices. Checks whether
                the fixed-input rates hold when the input text varies too.

Both append one JSON line per replicate and resume where they stopped, so an
interrupted run loses at most the call in flight.

    python experiments/adversarial_review_audit/audit.py fixed-input --n 385
    python experiments/adversarial_review_audit/audit.py pipeline --n 30
    python experiments/adversarial_review_audit/audit.py fixed-input --n 2   # pilot
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]

RATE = 25          # the paper's only rate
MAX_TRIALS = 3     # max_trials_choice in all three paper runs

# The paper's three configurations. Generator and reviewer are the same
# model in every run (ollama_model_choice == adversarial_model_choice), and
# `escalate` is each run's recorded escalation_batch decision.
CONDITIONS = {
    "qwen2b-20": {"run_dir": "Boisseau-qwen2b-25cond-20informativesents",
                  "model": "qwen3.5:2b", "escalate": True},
    "qwen9b-10": {"run_dir": "Boisseau-qwen9b-25cond-10informativesents",
                  "model": "qwen3.5:9b", "escalate": False},
    "qwen9b-20": {"run_dir": "Boisseau-qwen9b-25cond-20informativesents",
                  "model": "qwen3.5:9b", "escalate": False},
}


# ------------------------------------------------------------------
# Recording wrapper: the only thing this script changes at runtime.
# ------------------------------------------------------------------

CAPTURE: list[dict] = []


def install_recorder(ollama_client) -> None:
    real_post = ollama_client.requests.post

    def recording_post(url, json=None, timeout=None, **kw):  # noqa: A002 -- mirrors requests
        payload = json or {}
        prompt = payload.get("prompt", "")
        entry = {
            "url": url, "model": payload.get("model"), "options": payload.get("options"),
            "think": payload.get("think"), "prompt_chars": len(prompt),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "timeout_s": timeout,
        }
        t0 = time.monotonic()
        try:
            resp = real_post(url, json=json, timeout=timeout, **kw)
        except Exception as e:
            entry.update(elapsed_s=time.monotonic() - t0, error=repr(e))
            CAPTURE.append(entry)
            raise
        entry["elapsed_s"] = time.monotonic() - t0
        entry["http_status"] = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            body = None
            entry["body_text_head"] = resp.text[:300]
        if isinstance(body, dict):
            ctx = body.pop("context", None)
            entry["context_tokens"] = len(ctx) if isinstance(ctx, list) else None
            entry["body"] = body   # response, done_reason, prompt_eval_count, eval_count, durations
        CAPTURE.append(entry)
        return resp

    ollama_client.requests.post = recording_post


class ListLog:
    """Stands in for common/decisions.py::DecisionLog -- keeps records in memory."""

    def __init__(self):
        self.records = []

    def record(self, **kw):
        self.records.append(kw)


# ------------------------------------------------------------------
# Inputs, exactly as the paper runs had them
# ------------------------------------------------------------------

def load_condition(repo: Path, key: str, condense) -> dict:
    cfg = CONDITIONS[key]
    run = repo / "data" / cfg["run_dir"]
    state = json.loads((run / "phase2" / "informative_sentences.json").read_text(encoding="utf-8"))
    source_text = (run / "raw" / f"{cfg['run_dir']}.txt").read_text(encoding="utf-8")
    reviewed = None
    for line in (run / "decisions" / "phase2_decisions.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("decision_type") == "adversarial_review_result":
            reviewed = rec["extra"]["original_text"]
    if reviewed is None:
        raise SystemExit(f"{key}: no adversarial_review_result in {run}/decisions")
    return {
        "key": key, **cfg, "corpus_name": cfg["run_dir"],
        "ordered_sentences": condense.gather_ordered_informative_sentences(state),
        "cluster_key_terms": condense.gather_cluster_key_terms(state),
        "source_text": source_text,
        "target_words": condense.target_word_count(source_text, RATE),
        "paper_reviewed_text": reviewed,
    }


# ------------------------------------------------------------------
# Environment metadata -- the defect may depend on it
# ------------------------------------------------------------------

def _get(requests, url):
    try:
        return requests.get(url, timeout=10).json()
    except Exception as e:  # metadata is best-effort, never fatal
        return {"error": repr(e)}


def _post(requests, url, payload):
    # Deliberately requests.api.post, not requests.post: the recorder wraps
    # requests.post, and metadata calls must not land in CAPTURE.
    try:
        return requests.api.post(url, json=payload, timeout=30).json()
    except Exception as e:
        return {"error": repr(e)}


def environment_snapshot(requests, host: str, models: list[str], repo: Path) -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        commit = f"unavailable: {e!r}"
    snap = {
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "repo_commit": commit, "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "ollama_version": _get(requests, f"{host}/api/version"),
        "tags": _get(requests, f"{host}/api/tags"), "models": {},
        "gpu": _gpu_info(),
    }
    for m in sorted(set(models)):
        show = _post(requests, f"{host}/api/show", {"model": m})
        if isinstance(show, dict):
            show.pop("license", None)
            show.pop("modelfile", None)
            show.pop("template", None)
            show.pop("tensors", None)
        snap["models"][m] = show
    return snap


def _gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None   # no NVIDIA GPU (e.g. Apple silicon)


def ps_snapshot(requests, host: str):
    got = _get(requests, f"{host}/api/ps")
    if isinstance(got, dict) and "models" in got:
        return [{k: m.get(k) for k in ("name", "context_length", "size", "size_vram", "expires_at")}
                for m in got["models"]]
    return got


# ------------------------------------------------------------------
# Runs
# ------------------------------------------------------------------

def done_count(out: Path, key: str) -> int:
    if not out.exists():
        return 0
    return sum(1 for line in out.read_text(encoding="utf-8").splitlines()
               if line.strip() and json.loads(line).get("condition") == key)


def append(out: Path, rec: dict) -> None:
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def review_once(cond, text, condense, requests, host, review_timeout):
    CAPTURE.clear()
    log = ListLog()
    t0 = time.monotonic()
    kw = {"host": host}
    if review_timeout is not None:
        kw["timeout"] = review_timeout
    final_text, was_fixed, _orig, issues = condense.run_adversarial_review(
        text, cond["ordered_sentences"], cond["target_words"], cond["corpus_name"],
        cond["model"], log, RATE, **kw,
    )
    call = CAPTURE[-1] if CAPTURE else None
    body = (call or {}).get("body") or {}
    # The pipeline's own verdict on whether the call failed: it records
    # "reviewer call failed: ..." whenever ollama_client raised OllamaError
    # (unreachable, timeout, non-200, or a body without "response").
    failed = next((i for i in issues if i.startswith("reviewer call failed")), None)
    return {
        "reviewed_text": text,
        "raw_response": None if failed else body.get("response"),
        "call_error": failed,
        "ollama": {k: body.get(k) for k in ("done", "done_reason", "prompt_eval_count", "eval_count",
                                             "total_duration", "load_duration", "prompt_eval_duration",
                                             "eval_duration")},
        "request": {k: (call or {}).get(k) for k in ("options", "think", "prompt_chars", "prompt_sha256",
                                                      "timeout_s", "elapsed_s", "context_tokens")},
        "loaded_models": ps_snapshot(requests, host),
        "pipeline_decision": log.records[-1]["choice"] if log.records else None,
        "pipeline_issues": issues, "pipeline_was_fixed": was_fixed,
        "pipeline_final_equals_input": final_text == text,
        "wall_s": time.monotonic() - t0,
    }


def generate_once(cond, condense, host):
    """One end-to-end generation, following the paper runs' web flow
    (webapp/pipeline_session.py) with each run's recorded escalation choice.

    The paper runs never reached the sanity-review step. When a replicate
    does, the researcher's choice is replaced by a fixed policy -- accept the
    candidate closest to the target -- and the replicate is tagged."""
    CAPTURE.clear()
    events = []
    args = (cond["ordered_sentences"], cond["cluster_key_terms"], cond["target_words"], cond["corpus_name"])

    res = condense.attempt_condensation_trials(*args, model=cond["model"], max_trials=MAX_TRIALS,
                                               include_full_text=False, host=host)
    trials = [dict(t, escalated=False) for t in res["trials"]]
    text = res["text"]
    if res["sanity_review_candidates"]:
        best = min(res["sanity_review_candidates"], key=lambda c: abs(c["word_count"] - c["target_words"]))
        text = best["text"]
        events.append("sanity_candidate_auto_accepted")
    elif not res["success"] and cond["escalate"]:
        events.append("escalated")
        esc = condense.attempt_condensation_trials(*args, model=cond["model"], max_trials=MAX_TRIALS,
                                                   include_full_text=True, source_text=cond["source_text"],
                                                   host=host)
        trials += [dict(t, escalated=True) for t in esc["trials"]]
        if esc["sanity_review_candidates"]:
            best = min(esc["sanity_review_candidates"], key=lambda c: abs(c["word_count"] - c["target_words"]))
            text = best["text"]
            events.append("sanity_candidate_auto_accepted")
        elif esc["text"] is not None:
            text = esc["text"]
    gen_calls = [{k: c.get(k) for k in ("elapsed_s", "prompt_chars", "options", "error", "context_tokens")}
                 | {"done_reason": (c.get("body") or {}).get("done_reason"),
                    "prompt_eval_count": (c.get("body") or {}).get("prompt_eval_count"),
                    "eval_count": (c.get("body") or {}).get("eval_count")} for c in CAPTURE]
    return text, trials, events, gen_calls


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("fixed-input", "pipeline"))
    ap.add_argument("--n", type=int, required=True, help="replicates per condition (resumes to this total)")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--host", default=None, help="Ollama host (default: the pipeline's DEFAULT_HOST)")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="repository root")
    ap.add_argument("--out", type=Path, default=None, help="JSONL output (default: results/<mode>.jsonl)")
    ap.add_argument("--tag", default=None,
                    help="free-text label stored on every record (e.g. the host/GPU a shard ran on)")
    ap.add_argument("--deadline-epoch", type=float, default=None,
                    help="do not START a replicate after this Unix time (for hosts with a hard runtime cap)")
    ap.add_argument("--review-timeout", type=float, default=None,
                    help="override the reviewer timeout; default is the pipeline's own (600 s). "
                         "Changing it changes how often call_failed occurs.")
    args = ap.parse_args(argv)

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from common import ollama_client  # noqa: E402
    from phase2 import condense       # noqa: E402
    import requests                   # noqa: E402

    host = args.host or ollama_client.DEFAULT_HOST
    if not ollama_client.is_available(host):
        raise SystemExit(f"Ollama is not reachable at {host}.")
    install_recorder(ollama_client)

    out = args.out or (HERE / "results" / f"{args.mode.replace('-', '_')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    conds = [load_condition(repo, k, condense) for k in args.conditions]

    available = set(ollama_client.list_models(host))
    missing = [c["model"] for c in conds if c["model"] not in available]
    if missing:
        raise SystemExit(f"Model(s) not pulled in Ollama: {sorted(set(missing))}")

    session = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta_path = out.parent / f"env_{args.mode.replace('-', '_')}_{session}.json"
    meta_path.write_text(json.dumps(environment_snapshot(requests, host, [c["model"] for c in conds], repo),
                                    indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"environment -> {meta_path}")

    for cond in conds:
        start = done_count(out, cond["key"])
        for i in range(start, args.n):
            if args.deadline_epoch is not None and time.time() >= args.deadline_epoch:
                print(f"deadline reached; stopping before replicate {i} of {cond['key']}", flush=True)
                return
            rec = {"mode": args.mode, "condition": cond["key"], "replicate": i, "session": session,
                   "tag": args.tag,
                   "model": cond["model"], "target_words": cond["target_words"],
                   "utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
            if args.mode == "fixed-input":
                rec.update(review_once(cond, cond["paper_reviewed_text"], condense, requests, host,
                                       args.review_timeout))
                rec.pop("reviewed_text")   # identical every time; recovered from data/ by analyze.py
            else:
                text, trials, events, gen_calls = generate_once(cond, condense, host)
                rec.update(generation_trials=trials, generation_events=events, generation_calls=gen_calls)
                if text is None:
                    rec["no_condensation"] = True
                else:
                    rec.update(review_once(cond, text, condense, requests, host, args.review_timeout))
            append(out, rec)
            print(f"[{cond['key']}] {i + 1}/{args.n}  decision={rec.get('pipeline_decision')}  "
                  f"{rec.get('wall_s', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
