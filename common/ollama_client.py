"""Minimal client for a locally running Ollama instance
(https://ollama.com, default http://localhost:11434).

Talks to Ollama's REST API directly via `requests` rather than adding the
separate `ollama` pip package, since this pipeline only ever needs a
single blocking generate call.
"""

import requests

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


def is_available(host: str = DEFAULT_HOST, timeout: float = 2) -> bool:
    try:
        requests.get(f"{host}/api/tags", timeout=timeout)
        return True
    except requests.RequestException:
        return False


def list_models(host: str = DEFAULT_HOST, timeout: float = 5) -> list:
    resp = requests.get(f"{host}/api/tags", timeout=timeout)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def generate(model: str, prompt: str, host: str = DEFAULT_HOST, timeout: float = 600,
             options: dict | None = None, think: bool | None = None) -> str:
    """Blocking, non-streaming call to POST /api/generate.

    `options` is passed straight through as Ollama's per-request "options"
    object (e.g. {"num_ctx": 8192}) -- without it, Ollama falls back to
    the model's own default context length, which for some models is
    large enough that the KV cache no longer fits in VRAM alongside the
    weights, forcing slow CPU-offloaded generation.

    `think`, if not None, is passed as Ollama's per-request "think" flag
    -- pass False to suppress a reasoning model's visible thinking output
    (faster, and irrelevant when only the final text is used).

    Raises OllamaError on connection failure, a non-200 response, or a
    response missing the expected "response" field.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    if think is not None:
        payload["think"] = think
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise OllamaError(f"Could not reach Ollama at {host}: {e}") from e

    if resp.status_code != 200:
        raise OllamaError(f"Ollama returned {resp.status_code}: {resp.text[:300]}")

    try:
        return resp.json()["response"]
    except (ValueError, KeyError) as e:
        raise OllamaError(f"Unexpected response from Ollama: {resp.text[:300]}") from e
