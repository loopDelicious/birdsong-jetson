#!/usr/bin/env bash
# Set up the voice assistant stack on the Jetson:
#   - Python venv with openWakeWord + faster-whisper + openai
#   - Piper TTS binary (aarch64) + a voice model
#   - Pre-download wake-word and Whisper models
#
# Assumes audio is already working (see README "Audio setup"): PipeWire with a
# default source (USB mic) and default sink (Bluetooth speaker).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-$ROOT/.venv-voice}"
PIPER_VERSION="${PIPER_VERSION:-2023.11.14-2}"
PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_aarch64.tar.gz"
VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
VOICE_NAME="en_US-lessac-medium"

echo "== Python venv =="
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r voice/requirements.txt

echo "== Piper TTS binary =="
if [[ ! -x "$ROOT/piper/piper" ]]; then
  wget -qO /tmp/piper.tar.gz "$PIPER_URL"
  tar -xzf /tmp/piper.tar.gz -C "$ROOT/"
fi

echo "== Piper voice =="
mkdir -p "$ROOT/voices"
[[ -s "$ROOT/voices/$VOICE_NAME.onnx" ]] || wget -qO "$ROOT/voices/$VOICE_NAME.onnx" "$VOICE_BASE/$VOICE_NAME.onnx"
[[ -s "$ROOT/voices/$VOICE_NAME.onnx.json" ]] || wget -qO "$ROOT/voices/$VOICE_NAME.onnx.json" "$VOICE_BASE/$VOICE_NAME.onnx.json"

echo "== Pre-download wake-word + Whisper models =="
"$VENV/bin/python" -c "import openwakeword.utils as u; u.download_models()"
"$VENV/bin/python" -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8'); print('whisper base.en ready')"

echo
echo "Setup complete. Start the LLM (scripts/start-llm.sh), then run:"
echo "  $VENV/bin/python voice/assistant.py"
