#!/usr/bin/env bash
# Install the Birdsong LLM as a systemd SYSTEM service so the Gemma container
# starts once on boot (no crash-loop). Run on the Jetson. Needs sudo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="$ROOT/voice/birdsong-llm.service"
UNIT_DST="/etc/systemd/system/birdsong-llm.service"

# Ollama competes for the same 8GB unified-memory pool at boot; disable it so
# the Gemma container can get the contiguous memory it needs.
if systemctl is-enabled --quiet ollama 2>/dev/null; then
  echo "Disabling ollama (frees unified memory for Gemma)..."
  sudo systemctl disable --now ollama || true
fi

echo "Installing $UNIT_DST ..."
sudo cp "$UNIT_SRC" "$UNIT_DST"
sudo systemctl daemon-reload
sudo systemctl enable birdsong-llm.service

echo
echo "LLM service installed and enabled for boot. Useful commands:"
echo "  sudo systemctl start birdsong-llm      # start now"
echo "  systemctl status birdsong-llm          # check state"
echo "  docker logs -f birdsong-llm            # watch model load"
