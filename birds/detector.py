#!/usr/bin/env python3
"""Continuous BirdNET bird-song detection for the Jetson.

Ported from loopDelicious/birdsong (Raspberry Pi): records 3-second windows
from the USB mic and runs BirdNET (tflite, CPU) on each one. Runs alongside
the voice assistant on the *same* microphone: capture goes through PipeWire
(parecord), which multiplexes one hardware device across any number of
clients, so this 48 kHz stream coexists with the assistant's 16 kHz stream.

On each detection:
  - a row is appended to a local SQLite log (birds/birdsong.db)
  - an event is POSTed to the demo overlay (voice/overlay_server.py), which
    shows a live "Birds heard" panel
  - a small JSON file of recent detections is rewritten, which the voice
    assistant injects into Birdy's context so you can ask about the birds

Run on the Jetson:

    .venv-birds/bin/python birds/detector.py --location

(or via the birdsong-birds.service user unit).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import json
import os
import select
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave

RATE = 48000  # BirdNET models are trained on 48 kHz audio
HERE = os.path.dirname(os.path.abspath(__file__))

# Demo overlay (voice/overlay_server.py). Fire-and-forget: if the overlay
# isn't running, emit() fails silently and detection is unaffected.
OVERLAY_URL = os.environ.get("BIRDSONG_OVERLAY", "http://127.0.0.1:8095/event")

# Runtime settings written by the overlay's control panel (POST /control on
# overlay_server.py) and picked up here between analysis windows, so min-conf
# and the location filter can be adjusted live from the browser.
SETTINGS_JSON = os.environ.get("BIRDS_SETTINGS_JSON", "/tmp/birdsong_settings.json")


def emit(kind: str, **data) -> None:
    """Push an event to the demo overlay (best-effort, never raises)."""
    if not OVERLAY_URL:
        return
    try:
        body = json.dumps({"kind": kind, **data}).encode("utf-8")
        req = urllib.request.Request(
            OVERLAY_URL, data=body, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=0.3).close()
    except Exception:  # noqa: BLE001 - overlay is optional
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BirdNET live detector (PipeWire mic)")
    p.add_argument("--source", default=os.environ.get("BIRDSONG_SOURCE", ""),
                   help="PipeWire source name (default: system default source)")
    p.add_argument("--seconds", type=float, default=3.0,
                   help="analysis window length (BirdNET uses 3 s)")
    p.add_argument("--min-conf", type=float,
                   default=float(os.environ.get("BIRDS_MIN_CONF", "0.5")))
    p.add_argument("--lat", type=float, default=float(os.environ.get("BIRDS_LAT", "37.77")))
    p.add_argument("--lon", type=float, default=float(os.environ.get("BIRDS_LON", "-122.42")))
    p.add_argument("--location", action="store_true",
                   default=os.environ.get("BIRDS_LOCATION", "1") == "1",
                   help="filter to species plausible at --lat/--lon this time of year")
    p.add_argument("--no-location", dest="location", action="store_false")
    p.add_argument("--db", default=os.path.join(HERE, "birdsong.db"))
    p.add_argument("--recent-json",
                   default=os.environ.get("BIRDS_RECENT_JSON", "/tmp/birdsong_recent.json"))
    p.add_argument("--recent-window", type=float, default=15 * 60,
                   help="seconds a detection stays in the recent-birds JSON")
    return p.parse_args()


class MicStream:
    """Continuous 48 kHz mono capture via parecord (a PipeWire client).

    parecord can wedge: it connects to the audio server but its recording
    stream never materializes (seen after boot races and WirePlumber
    restarts), leaving a silent pipe and a "deaf" service. The constructor
    therefore probes that audio actually flows, respawning until it does.
    """

    def __init__(self, source: str = ""):
        self.cmd = ["parecord", "--format=s16le", "--rate=%d" % RATE,
                    "--channels=1", "--raw"]
        if source:
            self.cmd += ["--device", source]
        self._connect()

    def _connect(self) -> None:
        """Spawn parecord and verify audio actually flows, retrying as needed."""
        for attempt in range(5):
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL)
            readable, _, _ = select.select([self.proc.stdout], [], [], 8.0)
            if readable:
                return  # audio is flowing
            print("[mic] parecord produced no audio in 8s; respawning",
                  file=sys.stderr, flush=True)
            self.close()
            time.sleep(2.0)
        raise RuntimeError("microphone capture failed: parecord never produced audio")

    def read_exact(self, nbytes: int) -> bytes:
        buf = b""
        while len(buf) < nbytes:
            readable, _, _ = select.select([self.proc.stdout], [], [], 10.0)
            part = self.proc.stdout.read1(nbytes - len(buf)) if readable else b""
            if not part:
                # Silent for 10 s (or EOF): the stream died out from under us
                # (audio server / WirePlumber restart). Reconnect and carry on.
                print("[mic] capture stalled or ended; respawning parecord",
                      file=sys.stderr, flush=True)
                self.close()
                self._connect()
            else:
                buf += part
        return buf

    def flush(self) -> None:
        """Discard audio buffered while we were busy analyzing, so the next
        window starts fresh instead of replaying a stale pipe backlog."""
        fd = self.proc.stdout
        while True:
            r, _, _ = select.select([fd], [], [], 0.0)
            if not r:
                break
            if not fd.read1(65536):
                break

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def open_db(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute(
        """CREATE TABLE IF NOT EXISTS detections (
               id INTEGER PRIMARY KEY,
               ts REAL NOT NULL,
               iso TEXT NOT NULL,
               common_name TEXT NOT NULL,
               scientific_name TEXT NOT NULL,
               confidence REAL NOT NULL
           )"""
    )
    db.commit()
    return db


def write_recent(path: str, recent: dict[str, dict], today: list[dict]) -> None:
    """Atomically rewrite the recent-detections JSON the assistant reads.

    Includes the day's per-species tally (from SQLite) alongside the
    last-few-minutes window, so Birdy can answer both "what was that?" and
    "what birds have you heard today?" -- and the day's memory survives
    detector restarts.
    """
    entries = sorted(recent.values(), key=lambda e: e["ts"], reverse=True)[:10]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"updated": time.time(), "birds": entries, "today": today}, f)
    os.replace(tmp, path)


def read_settings(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def today_summary(db: sqlite3.Connection) -> list[dict]:
    """Per-species tally for today, busiest species first. Backs both the
    wall's resting screen and the Birds Heard list; reads from SQLite, so
    the day's birds survive service restarts."""
    day_start = datetime.datetime.combine(
        datetime.date.today(), datetime.time.min).timestamp()
    rows = db.execute(
        "SELECT common_name, scientific_name, COUNT(*), MAX(ts), MAX(confidence)"
        " FROM detections WHERE ts >= ?"
        " GROUP BY common_name, scientific_name ORDER BY COUNT(*) DESC",
        (day_start,),
    ).fetchall()
    return [{"common_name": r[0], "scientific_name": r[1], "count": r[2],
             "last_ts": r[3], "confidence": r[4]} for r in rows]


def main() -> int:
    args = parse_args()

    print("Loading BirdNET analyzer (first run downloads the model)...", flush=True)
    # birdnetlib is chatty on import/load/analyze; keep the service log clean.
    with contextlib.redirect_stdout(io.StringIO()):
        from birdnetlib import Recording
        from birdnetlib.analyzer import Analyzer
        analyzer = Analyzer()

    db = open_db(args.db)
    mic = MicStream(args.source)
    window_bytes = int(args.seconds * RATE) * 2

    # Live-tunable settings: CLI/env defaults, overridden by the overlay's
    # control panel via SETTINGS_JSON (checked between windows by mtime).
    min_conf = args.min_conf
    use_location = args.location
    settings_mtime = -1.0

    def apply_settings() -> None:
        nonlocal min_conf, use_location, settings_mtime
        try:
            mtime = os.path.getmtime(SETTINGS_JSON)
        except OSError:
            return
        if mtime == settings_mtime:
            return
        settings_mtime = mtime
        data = read_settings(SETTINGS_JSON)
        new_conf = max(0.1, min(0.95, float(data.get("min_conf", min_conf))))
        new_loc = bool(data.get("use_location", use_location))
        if (new_conf, new_loc) != (min_conf, use_location):
            min_conf, use_location = new_conf, new_loc
            if not use_location:
                # birdnetlib leaves the location allow-list set on the reused
                # analyzer; clear it or the filter sticks after toggling off.
                analyzer.custom_species_list = []
            print(f"[config] min_conf={min_conf:.2f}"
                  f" location={'on' if use_location else 'off'}", flush=True)
        emit("birds_config", min_conf=min_conf, use_location=use_location)

    apply_settings()  # pick up panel settings from before a restart
    loc = f" | location ({args.lat},{args.lon})" if use_location else ""
    print(f"Listening | {args.seconds:.0f}s windows | min_conf={min_conf}{loc}", flush=True)
    emit("birds_config", min_conf=min_conf, use_location=use_location)
    emit("birds_today", species=today_summary(db))

    # species -> {"common_name", "scientific_name", "confidence", "ts", "count"}
    recent: dict[str, dict] = {}
    tmp_wav = os.path.join(tempfile.gettempdir(), "birdnet_window.wav")

    try:
        debug = os.environ.get("BIRDS_DEBUG") == "1"
        while True:
            apply_settings()
            t0 = time.time()
            mic.flush()  # drop backlog accumulated during the previous analysis
            t1 = time.time()
            data = mic.read_exact(window_bytes)
            if debug:
                print(f"[debug] flush={t1-t0:.2f}s read={time.time()-t1:.2f}s", flush=True)
            if len(data) < window_bytes:
                print("[mic] capture stream ended", file=sys.stderr, flush=True)
                return 1
            with wave.open(tmp_wav, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(data)

            kwargs = {"min_conf": min_conf}
            if use_location:
                kwargs.update(lat=args.lat, lon=args.lon, date=datetime.datetime.now())
            with contextlib.redirect_stdout(io.StringIO()):
                rec = Recording(analyzer, tmp_wav, **kwargs)
                rec.analyze()

            now = time.time()
            iso = datetime.datetime.now().isoformat(timespec="seconds")
            for d in rec.detections:
                name = d["common_name"]
                conf = float(d["confidence"])
                print(f"[{iso}] {name} ({d['scientific_name']}) conf={conf:.2f}", flush=True)
                db.execute(
                    "INSERT INTO detections (ts, iso, common_name, scientific_name, confidence)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (now, iso, name, d["scientific_name"], conf),
                )
                db.commit()
                prev = recent.get(name)
                recent[name] = {
                    "common_name": name,
                    "scientific_name": d["scientific_name"],
                    "confidence": max(conf, prev["confidence"]) if prev else conf,
                    "ts": now,
                    "count": (prev["count"] + 1) if prev else 1,
                }
                emit("bird", common_name=name, scientific_name=d["scientific_name"],
                     confidence=round(conf, 2), ts=now)

            if rec.detections:
                emit("birds_today", species=today_summary(db))

            # Age out stale species and refresh the JSON the assistant reads.
            # (Every window, not just on detections: the assistant must see the
            # day's tally right after a restart, before any new bird sings.)
            recent = {k: v for k, v in recent.items() if now - v["ts"] < args.recent_window}
            write_recent(args.recent_json, recent, today_summary(db))
    except KeyboardInterrupt:
        return 0
    finally:
        mic.close()
        db.close()


if __name__ == "__main__":
    sys.exit(main())
