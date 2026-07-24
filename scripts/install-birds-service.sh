#!/usr/bin/env bash
# Install and start the BirdNET detector as a systemd --user service.
# Run on the Jetson after scripts/setup-birds.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"

mkdir -p "$UNIT_DIR"
cp "$ROOT/voice/birdsong-birds.service" "$UNIT_DIR/birdsong-birds.service"

systemctl --user daemon-reload
systemctl --user enable --now birdsong-birds.service

echo "Service installed. Useful commands:"
echo "  systemctl --user status birdsong-birds"
echo "  systemctl --user restart birdsong-birds"
echo "  tail -f $ROOT/birds.log"
