#!/usr/bin/env python3
"""Demo overlay server for the Birdsong voice assistant.

Shows the assistant's live state (listening / thinking / speaking), the
transcribed prompt, and Birdy's reply streaming in, on a browser page.
Handy for demos when the Jetson is headless: open the page from your
laptop at http://jetson-desktop.local:8095

Standard library only (no pip dependencies): ThreadingHTTPServer plus
Server-Sent Events, so it can run with the system python3 outside the venv.

Endpoints:
  POST /event   assistant pushes JSON events, e.g. {"kind": "token", ...}
  GET  /events  SSE stream; sends a full "snapshot" on connect, then live events
  GET  /        the demo page (voice/overlay.html)

Run directly (python3 voice/overlay_server.py) or via the
birdsong-overlay.service user unit (scripts/install-overlay-service.sh).
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.environ.get("OVERLAY_HOST", "0.0.0.0")
PORT = int(os.environ.get("OVERLAY_PORT", "8095"))
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlay.html")

_lock = threading.Lock()
_subscribers: set[queue.Queue] = set()

# Retained state so a browser that connects mid-conversation (or reloads)
# immediately gets the current picture via a "snapshot" event.
_state = {
    "status": "offline",  # offline | idle | listening | thinking | speaking | error
    "wake": "hey birdy",
    "model": "",
    "prompt": "",
    "response": "",
    "ttfw": None,   # seconds until the LLM's first word
    "total": None,  # seconds for the whole turn (STT + LLM + speech)
    "error": "",
    # Recent BirdNET detections (birds/detector.py), newest first, deduped by
    # species so the panel shows each bird once with its last-heard time.
    "birds": [],
}


def _apply(event: dict) -> None:
    """Fold one event into the retained state."""
    kind = event.get("kind")
    if kind == "hello":
        _state["model"] = event.get("model", _state["model"])
        _state["wake"] = event.get("wake", _state["wake"])
        _state["status"] = "idle"
    elif kind == "state":
        _state["status"] = event.get("status", _state["status"])
        if _state["status"] in ("listening", "idle"):
            _state["error"] = ""
    elif kind == "prompt":
        _state["prompt"] = event.get("text", "")
        _state["response"] = ""
        _state["ttfw"] = None
        _state["total"] = None
    elif kind == "token":
        _state["response"] += event.get("text", "")
    elif kind == "done":
        _state["response"] = event.get("text", _state["response"])
    elif kind == "metric":
        for key in ("ttfw", "total"):
            if key in event:
                _state[key] = event[key]
    elif kind == "error":
        _state["status"] = "error"
        _state["error"] = event.get("message", "")
    elif kind == "bird":
        entry = {
            "common_name": event.get("common_name", ""),
            "scientific_name": event.get("scientific_name", ""),
            "confidence": event.get("confidence", 0),
            "ts": event.get("ts", time.time()),
        }
        birds = [b for b in _state["birds"] if b["common_name"] != entry["common_name"]]
        _state["birds"] = [entry] + birds[:11]  # newest first, max 12 species


def _broadcast(event: dict) -> None:
    with _lock:
        _apply(event)
        for q in list(_subscribers):
            try:
                q.put_nowait(event)
            except queue.Full:
                _subscribers.discard(q)  # viewer stopped reading; drop it


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the service log quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str = "text/plain") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _sse(self, event: dict) -> None:
        payload = json.dumps(event, ensure_ascii=False)
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open(PAGE, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                self._send(500, b"overlay.html not found next to overlay_server.py")
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            q: queue.Queue = queue.Queue(maxsize=1024)
            with _lock:
                snapshot = dict(_state)
                _subscribers.add(q)
            try:
                self._sse({"kind": "snapshot", **snapshot})
                while True:
                    try:
                        ev = q.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")  # keep proxies/browser happy
                        self.wfile.flush()
                        continue
                    self._sse(ev)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass  # viewer closed the tab
            finally:
                with _lock:
                    _subscribers.discard(q)
        else:
            self._send(404, b"not found")

    def do_POST(self):
        if self.path != "/event":
            self._send(404, b"not found")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            event = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(event, dict):
                raise ValueError
        except (ValueError, json.JSONDecodeError):
            self._send(400, b"bad json")
            return
        _broadcast(event)
        self._send(200, b"ok")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"Birdsong overlay listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
