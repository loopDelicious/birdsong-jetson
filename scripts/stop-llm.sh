#!/usr/bin/env bash
# Stop the birdsong llama.cpp container.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-birdsong-llm}"
REMOVE="${REMOVE:-0}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found." >&2
  exit 1
fi

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

if ! "${DOCKER[@]}" ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Container '$CONTAINER_NAME' not found."
  exit 0
fi

if "${DOCKER[@]}" ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Stopping '$CONTAINER_NAME'..."
  "${DOCKER[@]}" stop "$CONTAINER_NAME"
else
  echo "Container '$CONTAINER_NAME' is already stopped."
fi

if [[ "$REMOVE" == "1" ]]; then
  echo "Removing '$CONTAINER_NAME'..."
  "${DOCKER[@]}" rm "$CONTAINER_NAME"
fi
