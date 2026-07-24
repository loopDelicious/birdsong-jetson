#!/usr/bin/env bash
# Install the JBL auto-reconnect as a systemd --user service so the speaker
# links up and becomes the default sink on login/boot. Run on the Jetson.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"

mkdir -p "$UNIT_DIR"
cp "$ROOT/voice/birdsong-jbl.service" "$UNIT_DIR/birdsong-jbl.service"

systemctl --user daemon-reload
systemctl --user enable --now birdsong-jbl.service

echo "JBL auto-reconnect service installed. Useful commands:"
echo "  systemctl --user status birdsong-jbl"
echo "  systemctl --user restart birdsong-jbl   # force a reconnect now"
echo
echo "Note: lingering must be enabled so it runs on boot without a login:"
echo "  sudo loginctl enable-linger $USER   (already enabled if you did this before)"
