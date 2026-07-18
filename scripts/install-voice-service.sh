#!/usr/bin/env bash
# Install and start the Birdsong voice assistant as a systemd --user service.
# Run on the Jetson after scripts/setup-voice.sh and scripts/start-llm.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"

mkdir -p "$UNIT_DIR"
cp "$ROOT/voice/birdsong-voice.service" "$UNIT_DIR/birdsong-voice.service"

systemctl --user daemon-reload
systemctl --user enable --now birdsong-voice.service

echo "Service installed. Useful commands:"
echo "  systemctl --user status birdsong-voice"
echo "  systemctl --user restart birdsong-voice"
echo "  tail -f $ROOT/voice.log"
echo
echo "Tip: to keep it running after logout / on boot, enable lingering once:"
echo "  sudo loginctl enable-linger $USER"
