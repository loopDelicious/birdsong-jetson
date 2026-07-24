#!/usr/bin/env bash
# Install the Birdsong demo overlay as a systemd --user service. Run on the
# Jetson. The overlay serves http://jetson-desktop.local:8095 with the live
# prompt/response view; the voice assistant pushes events to it automatically.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"

mkdir -p "$UNIT_DIR"
cp "$ROOT/voice/birdsong-overlay.service" "$UNIT_DIR/birdsong-overlay.service"

systemctl --user daemon-reload
systemctl --user enable --now birdsong-overlay.service

echo "Overlay service installed. Useful commands:"
echo "  systemctl --user status birdsong-overlay"
echo "  systemctl --user restart birdsong-overlay"
echo "  tail -f $ROOT/overlay.log"
echo
echo "View from another machine on your network:"
echo "  http://$(hostname).local:8095"
