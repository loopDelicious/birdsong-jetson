#!/usr/bin/env bash
# Connect the JBL Bluetooth speaker and make it the default audio sink.
# Run by the birdsong-jbl.service user unit at login/boot, or by hand.
# The speaker must be paired+trusted already (see README) and powered on.
set -uo pipefail

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
MAC="${JBL_MAC:-2C:FD:B4:BC:1D:B0}"
SINK="bluez_output.${MAC//:/_}.a2dp-sink"
VOL="${JBL_VOLUME:-25%}"

# Wait for Bluetooth + PipeWire to be ready before we try anything.
for _ in $(seq 1 30); do
  if bluetoothctl show >/dev/null 2>&1 && pactl info >/dev/null 2>&1; then break; fi
  sleep 2
done

bluetoothctl power on >/dev/null 2>&1 || true

# Speakers often need the host to initiate the link, sometimes a few times.
for _ in $(seq 1 10); do
  if bluetoothctl info "$MAC" 2>/dev/null | grep -q "Connected: yes"; then break; fi
  bluetoothctl connect "$MAC" >/dev/null 2>&1 || true
  sleep 3
done

# Wait for the A2DP sink to appear; nudge WirePlumber once if it's slow.
for i in $(seq 1 10); do
  if pactl list short sinks 2>/dev/null | grep -q "$SINK"; then break; fi
  if [ "$i" = "5" ]; then systemctl --user restart wireplumber || true; fi
  sleep 2
done

if pactl list short sinks 2>/dev/null | grep -q "$SINK"; then
  pactl set-default-sink "$SINK" || true
  pactl set-sink-volume "$SINK" "$VOL" || true
  echo "JBL connected; default sink set to $SINK at $VOL"
else
  echo "JBL sink not available - is the speaker powered on and in range?" >&2
  exit 1
fi
