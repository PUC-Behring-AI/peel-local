#!/usr/bin/env bash
# SageMaker training-job entrypoint for the adversarial-review audit.
# Installs a pinned Ollama, pulls the two models, verifies their digests
# against the ones recorded on the pilot host, then runs driver.py, which
# runs audit.py shards in parallel. Everything written to /opt/ml/checkpoints
# is synced to S3 by SageMaker while the job runs.
set -euo pipefail

OUT=/opt/ml/checkpoints
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1
echo "[$(date -u +%FT%TZ)] job start on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU')"

WORK=/opt/peel
mkdir -p "$WORK"
tar -xzf /opt/ml/input/data/code/bundle.tar.gz -C "$WORK"

OLLAMA_VERSION=0.24.0
export DEBIAN_FRONTEND=noninteractive
(apt-get update -qq && apt-get install -y -qq zstd >/dev/null) || echo "zstd install failed (only needed for .tar.zst releases)"
python -c "import requests" 2>/dev/null || pip install -q requests

install_ollama() {
  if curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=$OLLAMA_VERSION sh; then return 0; fi
  echo "install.sh failed; trying the release archive directly"
  for ext in tgz tar.zst; do
    url="https://github.com/ollama/ollama/releases/download/v$OLLAMA_VERSION/ollama-linux-amd64.$ext"
    if curl -fsSL -o /tmp/ollama.$ext "$url"; then
      if [ "$ext" = tgz ]; then tar -xzf /tmp/ollama.$ext -C /usr; else tar --zstd -xf /tmp/ollama.$ext -C /usr; fi
      return 0
    fi
  done
  return 1
}
install_ollama
ollama --version || true

# Context: 32768 covers prompt + answer with 2x margin over the largest pilot
# call (7,532 + 7,472 tokens); audit records let every call be checked for
# prompt_eval_count + eval_count < 32768 and done_reason == "stop".
export OLLAMA_CONTEXT_LENGTH=32768
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-2}"
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_HOST=127.0.0.1:11434
env | grep '^OLLAMA_' | sort
ollama serve > "$OUT/ollama_serve.log" 2>&1 &
for i in $(seq 1 60); do curl -sf http://127.0.0.1:11434/api/version && break; sleep 2; done
echo

ollama pull qwen3.5:2b
ollama pull qwen3.5:9b
python "$WORK/sagemaker/verify_digests.py"

cd "$WORK"
python "$WORK/sagemaker/driver.py"
echo "[$(date -u +%FT%TZ)] job end"
