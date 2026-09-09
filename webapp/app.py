#!/usr/bin/env python
"""PEEL-Local web interface: a Flask server wrapping run_pipeline.py's
same end-to-end flow (Phase 0 optional -> Phase 1 -> Phase 2 -> Phase 3
distant reading -> standalone report export) behind a browser UI -- file
upload, every CLI
config parameter as an editable field, and every interactive decision
point as a form instead of a terminal prompt.

Usage:
    python webapp/app.py
    -> open http://localhost:5000

See RUN_PIPELINE_GUIDE.md for the CLI equivalent, and README.md's
"Web interface" section for a screen-by-screen walkthrough.

One run at a time: this is a local, single-researcher tool, not a
multi-user server -- starting a new run replaces whatever the previous
one was doing.
"""

import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flask import Flask, jsonify, request, send_file, send_from_directory

from common import ollama_client, resume
from common.paths import CorpusPaths
from webapp.pipeline_session import PipelineSession

app = Flask(__name__, static_folder="static", static_url_path="")

_session = PipelineSession()

_CORPUS_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_CONFIG_DEFAULTS = {
    "frequency_percentile": ("float", 0.50),
    "max_stems": ("int", 150),
    "word_frequency_percentile": ("float", 0.0),
    "max_sentences_per_stem": ("int", 5),
    "max_synsets": ("int", 5),
    "merge_duplicate_word_occurrences": ("bool", False),
    "run_parameter_sweep": ("bool", False),
    "max_cluster_size": ("int", 10),
    "min_clusters": ("int", 5),
    "min_cluster_len": ("int", 3),
    "glossbert_model": ("str", "jvomiranda/GlossBERT_Checkpoint"),
    "lang_model": ("str", "en_core_web_sm"),
    "sentence_embedder": ("str", "all-MiniLM-L6-v2"),
}


def _error(message, status=400):
    return jsonify({"ok": False, "error": message}), status


# ------------------------------------------------------------
# Static frontend
# ------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ------------------------------------------------------------
# Status
# ------------------------------------------------------------

@app.route("/api/status")
def status():
    since = request.args.get("since", default=0, type=int)
    return jsonify(_session.status_dict(since))


# ------------------------------------------------------------
# Start a new run
# ------------------------------------------------------------

@app.route("/api/start", methods=["POST"])
def start():
    global _session

    if _session.status == "running":
        return _error("A pipeline run is already in progress.", 409)

    corpus_name = request.form.get("corpus_name", "").strip()
    if not corpus_name or not _CORPUS_NAME_RE.match(corpus_name):
        return _error("Corpus name is required and may only contain letters, digits, - and _.")

    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return _error("A corpus file (.txt, .md/.markdown, or .pdf) is required.")

    clean = request.form.get("clean") in ("true", "on", "1")

    config = {}
    for key, (kind, default) in _CONFIG_DEFAULTS.items():
        raw = request.form.get(key)
        if raw is None or raw == "":
            config[key] = default
            continue
        try:
            if kind == "bool":
                config[key] = raw in ("true", "on", "1")
            else:
                config[key] = float(raw) if kind == "float" else (int(raw) if kind == "int" else raw)
        except ValueError:
            return _error(f"Invalid value for {key}: {raw!r}")

    _session = PipelineSession()
    try:
        _session.start(corpus_name, uploaded, clean, config)
    except UnicodeDecodeError:
        return _error("The uploaded .txt/.md file isn't valid UTF-8 text.")
    except ValueError as e:
        # Unsupported extension, or a PDF with no extractable text (e.g.
        # scanned/image-only) -- a bad-input problem, not a server error.
        return _error(str(e))
    except Exception as e:  # noqa: BLE001 -- surfaced to the UI, not a bare crash
        return _error(f"Could not start the pipeline: {e}", 500)

    return jsonify({"ok": True})


@app.route("/api/corpora")
def list_corpora():
    """Every corpus under data/, with which phases it currently satisfies
    prerequisites for -- populates the setup screen's "resume an existing
    corpus" picker so a researcher can see what's actually resumable
    before picking a phase that would just fail validation."""
    return jsonify({"corpora": [resume.corpus_phase_summary(name) for name in resume.list_corpora()]})


@app.route("/api/resume", methods=["POST"])
def resume_corpus():
    global _session

    if _session.status == "running":
        return _error("A pipeline run is already in progress.", 409)

    body = request.get_json(force=True) or {}
    corpus_name = (body.get("corpus_name") or "").strip()
    if not corpus_name or not _CORPUS_NAME_RE.match(corpus_name):
        return _error("Corpus name is required and may only contain letters, digits, - and _.")

    try:
        start_phase = int(body.get("start_phase"))
    except (TypeError, ValueError):
        return _error("start_phase must be 2 or 3.")
    if start_phase not in (2, 3):
        return _error("start_phase must be 2 or 3 (use /api/start for a new corpus at Phase 1).")

    config = {"lang_model": body.get("lang_model") or _CONFIG_DEFAULTS["lang_model"][1]}

    _session = PipelineSession()
    try:
        _session.resume(corpus_name, start_phase, config)
    except RuntimeError as e:
        # check_prerequisites failure -- a real "not ready" state, not a
        # server error, so the UI can show it inline like any other
        # validation message.
        return _error(str(e))
    except Exception as e:  # noqa: BLE001 -- surfaced to the UI, not a bare crash
        return _error(f"Could not resume the pipeline: {e}", 500)

    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def reset():
    """Clears the session back to idle -- used by "Start another run" (on
    the complete screen) and the header's "Restart" button (available on
    every screen). `force=true` (sent by the header button, since it can
    be clicked mid-run) discards the session even while status=="running":
    the background thread already in flight has no cancellation hook, so
    it keeps executing to completion against the *old* PipelineSession
    object -- wasted work, but harmless, since nothing references that
    object anymore once _session points elsewhere; its result is just
    never observed. Without `force`, a run in progress is protected."""
    global _session

    body = request.get_json(silent=True) or {}
    force = bool(body.get("force"))

    if _session.status == "running" and not force:
        return _error("A pipeline run is already in progress.", 409)

    _session = PipelineSession()
    return jsonify({"ok": True})


# ------------------------------------------------------------
# Decision endpoints
# ------------------------------------------------------------

def _require_decision(expected_type):
    if _session.status != "awaiting_input" or not _session.decision:
        return _error("No decision is currently awaiting input.", 409)
    if _session.decision["type"] != expected_type:
        return _error(f"Expected decision type {expected_type!r}, current is {_session.decision['type']!r}.", 409)
    return None


@app.route("/api/decisions/flagged-terms", methods=["POST"])
def decide_flagged_terms():
    err = _require_decision("flagged_term_review")
    if err:
        return err
    body = request.get_json(force=True) or {}
    try:
        _session.apply_flagged_term_review(body.get("decisions", []))
    except (RuntimeError, KeyError, IndexError, ValueError) as e:
        return _error(str(e), 400)
    return jsonify({"ok": True})


@app.route("/api/decisions/cluster-review", methods=["POST"])
def decide_cluster_review():
    err = _require_decision("cluster_review")
    if err:
        return err
    body = request.get_json(force=True) or {}
    try:
        _session.apply_cluster_review(body.get("clusters", []))
    except (RuntimeError, KeyError) as e:
        return _error(str(e), 400)
    return jsonify({"ok": True})


@app.route("/api/decisions/parameter-sweep", methods=["POST"])
def decide_parameter_sweep():
    err = _require_decision("parameter_sweep")
    if err:
        return err
    body = request.get_json(force=True) or {}
    try:
        _session.apply_parameter_sweep(
            choice=body.get("choice", ""),
            frequency_percentile=float(body["frequency_percentile"]),
            max_stems=int(body["max_stems"]),
        )
    except (RuntimeError, KeyError, ValueError) as e:
        return _error(str(e), 400)
    return jsonify({"ok": True})


@app.route("/api/decisions/phase2-setup", methods=["POST"])
def decide_phase2_setup():
    err = _require_decision("phase2_setup")
    if err:
        return err
    body = request.get_json(force=True) or {}
    try:
        _session.apply_phase2_setup(
            top_n=int(body.get("top_n", 10)),
            density_percentile=float(body.get("density_percentile", 75)),
            run_condensation=bool(body.get("run_condensation", False)),
            rates_str=body.get("rates", ""),
            ollama_model=body.get("ollama_model", ""),
            adversarial_model=body.get("adversarial_model", ""),
            max_trials=int(body.get("max_trials", 3)),
            title=body.get("title", ""),
            authors=body.get("authors", ""),
            date=body.get("date", ""),
        )
    except (RuntimeError, ValueError) as e:
        return _error(str(e), 400)
    return jsonify({"ok": True})


@app.route("/api/detect-metadata")
def detect_metadata():
    """Runs the AI title/author/date detector against this corpus's
    untouched pre-Phase-0 source text, for the Phase 2 setup screen to
    pre-fill its fields with before the researcher sees them -- "the
    workflow already tried this" instead of asking the researcher to type
    it in blind. Synchronous (not a background job): it's one small
    retrieval-augmented LLM call, same latency class as the regeneration
    prompt preview route."""
    if _session.status != "awaiting_input" or not _session.decision or _session.decision["type"] != "phase2_setup":
        actual = _session.decision["type"] if _session.decision else None
        print(f"[detect-metadata] rejected: status={_session.status!r}, decision_type={actual!r} (expected awaiting_input/phase2_setup)")
        return _error("Metadata detection is only available on the Phase 2 setup screen.", 409)
    ollama_model = request.args.get("ollama_model", "").strip()
    if not ollama_model:
        print("[detect-metadata] rejected: no ollama_model in request")
        return _error("An Ollama model is required.")
    try:
        result = _session.detect_metadata(ollama_model)
    except RuntimeError as e:
        print(f"[detect-metadata] RuntimeError: {e}")
        return _error(str(e), 400)
    except Exception as e:  # noqa: BLE001 -- surfaced to the UI, not a bare crash
        print(f"[detect-metadata] unexpected exception: {e}")
        traceback.print_exc()
        return _error(f"Metadata detection failed: {e}", 500)
    return jsonify({"ok": True, **result})


@app.route("/api/decisions/sanity-review", methods=["POST"])
def decide_sanity_review():
    err = _require_decision("sanity_review")
    if err:
        return err
    body = request.get_json(force=True) or {}
    try:
        _session.apply_sanity_review(body.get("decisions", {}))
    except RuntimeError as e:
        return _error(str(e), 400)
    return jsonify({"ok": True})


@app.route("/api/decisions/escalation", methods=["POST"])
def decide_escalation():
    err = _require_decision("escalation")
    if err:
        return err
    body = request.get_json(force=True) or {}
    try:
        _session.apply_escalation(body.get("decisions", {}))
    except RuntimeError as e:
        return _error(str(e), 400)
    return jsonify({"ok": True})


@app.route("/api/decisions/injection-review", methods=["POST"])
def decide_injection_review():
    err = _require_decision("injection_review")
    if err:
        return err
    body = request.get_json(force=True) or {}
    try:
        _session.apply_injection_review(body.get("decisions", {}))
    except RuntimeError as e:
        return _error(str(e), 400)
    return jsonify({"ok": True})


@app.route("/api/decisions/collocation-review", methods=["POST"])
def decide_collocation_review():
    err = _require_decision("collocation_review")
    if err:
        return err
    body = request.get_json(force=True) or {}
    try:
        _session.apply_collocation_review(body.get("selected_indices", []))
    except (RuntimeError, KeyError, IndexError) as e:
        return _error(str(e), 400)
    return jsonify({"ok": True})


# ------------------------------------------------------------
# Post-completion condensation regeneration
# ------------------------------------------------------------

@app.route("/api/regenerate/prompt-preview")
def regenerate_prompt_preview():
    if _session.status != "complete":
        return _error("Can only preview a regeneration prompt once the pipeline has completed.", 409)
    rate = request.args.get("rate", type=int)
    if rate is None:
        return _error("A numeric rate is required.")
    try:
        prompt = _session.preview_regeneration_prompt(rate)
    except RuntimeError as e:
        return _error(str(e), 400)
    return jsonify({"ok": True, "prompt": prompt})


@app.route("/api/regenerate", methods=["POST"])
def regenerate():
    if _session.status != "complete":
        return _error("Can only regenerate a condensation once the pipeline has completed.", 409)
    body = request.get_json(force=True) or {}
    rate = body.get("rate")
    if not isinstance(rate, int):
        return _error("A numeric rate is required.")
    prompt_override = body.get("prompt") or None
    try:
        _session.start_regeneration(rate, prompt_override)
    except RuntimeError as e:
        return _error(str(e), 400)
    return jsonify({"ok": True})


# ------------------------------------------------------------
# Ollama models (for the Phase 2 setup screen's model dropdown)
# ------------------------------------------------------------

@app.route("/api/ollama/models")
def ollama_models():
    try:
        available = ollama_client.is_available()
        models = ollama_client.list_models() if available else []
        return jsonify({"available": available, "models": models})
    except ollama_client.OllamaError as e:
        return jsonify({"available": False, "models": [], "error": str(e)})


# ------------------------------------------------------------
# Generated file access
# ------------------------------------------------------------

@app.route("/api/files/<corpus>/<path:subpath>")
def get_file(corpus, subpath):
    if not _CORPUS_NAME_RE.match(corpus):
        return _error("Invalid corpus name.", 400)

    root = CorpusPaths(corpus).root.resolve()
    target = (root / subpath).resolve()

    if not target.is_relative_to(root) or not target.is_file():
        return _error("File not found.", 404)

    return send_file(target)


if __name__ == "__main__":
    print("PEEL-Local web interface starting at http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
