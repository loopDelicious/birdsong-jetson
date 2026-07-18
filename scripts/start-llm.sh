#!/usr/bin/env bash
# Start Gemma 4 E2B via NVIDIA's llama.cpp container (Jetson Orin).
# Run this on the Jetson, not on your Mac.
#
# On the Orin Nano's 8GB unified memory, the default `-hf` path also pulls the
# vision projector and a large context, which can OOM at load time
# ("unable to allocate CUDA0 buffer"). To stay reliable we:
#   - download the GGUF once to a host dir and mount it,
#   - skip the vision projector (--no-mmproj),
#   - use a modest context (-c),
#   - stop Ollama so it isn't holding memory.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-birdsong-llm}"
IMAGE="${IMAGE:-ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin}"
MODEL_DIR="${MODEL_DIR:-$HOME/gguf}"
MODEL_FILE="${MODEL_FILE:-gemma-4-E2B-it-Q4_K_S.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/${MODEL_FILE}}"
CTX="${CTX:-2048}"
NGL="${NGL:-99}"
HOST_PORT="${HOST_PORT:-8080}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Install Docker with the NVIDIA container runtime on the Jetson first." >&2
  exit 1
fi

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

# Free unified memory: Ollama competes for the same 8GB pool.
if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet ollama; then
  echo "Stopping ollama to free memory..."
  sudo systemctl stop ollama || true
fi

# Download the model once to the host so it survives container recreation.
mkdir -p "$MODEL_DIR"
if [[ ! -s "$MODEL_DIR/$MODEL_FILE" ]]; then
  echo "Downloading $MODEL_FILE (~3GB) to $MODEL_DIR ..."
  wget -c -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
else
  echo "Model already present: $MODEL_DIR/$MODEL_FILE"
fi

"${DOCKER[@]}" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Starting '$CONTAINER_NAME' (ctx=$CTX, ngl=$NGL, no mmproj)..."
"${DOCKER[@]}" run -d \
  --name "$CONTAINER_NAME" \
  --runtime=nvidia \
  --network host \
  --restart unless-stopped \
  -v "$MODEL_DIR:/models" \
  "$IMAGE" \
  llama-server \
    -m "/models/$MODEL_FILE" \
    --no-mmproj \
    -c "$CTX" \
    -ngl "$NGL" \
    --host 0.0.0.0 --port "$HOST_PORT"

echo
echo "Container: $CONTAINER_NAME"
echo "OpenAI-compatible API: http://127.0.0.1:${HOST_PORT}/v1"
echo "Browser UI:            http://127.0.0.1:${HOST_PORT}"
echo
echo "Wait for 'server is listening' then 'model loaded':"
echo "  ${DOCKER[*]} logs -f $CONTAINER_NAME"
echo "Stop with: ./scripts/stop-llm.sh"
