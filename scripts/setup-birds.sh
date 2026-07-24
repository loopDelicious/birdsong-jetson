#!/usr/bin/env bash
# Set up the BirdNET bird-song detector on the Jetson:
#   - a *separate* Python venv (.venv-birds): tflite-runtime needs numpy<2,
#     while the voice assistant venv uses numpy 2
#   - pre-download the BirdNET model (birdnetlib fetches it on first use)
#
# Assumes audio is already working (see README "Audio setup"): PipeWire with
# the USB mic as the default source. The detector shares the mic with the
# voice assistant via PipeWire; no extra audio config is needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-$ROOT/.venv-birds}"

echo "== Python venv (.venv-birds) =="
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r birds/requirements.txt

echo "== Pre-download BirdNET model =="
"$VENV/bin/python" - <<'PY'
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from birdnetlib.analyzer import Analyzer
    Analyzer()
print("BirdNET model ready")
PY

echo
echo "Setup complete. Run the detector with:"
echo "  $VENV/bin/python birds/detector.py"
echo "or install the service:"
echo "  scripts/install-birds-service.sh"
