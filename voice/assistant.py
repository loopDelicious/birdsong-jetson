#!/usr/bin/env python3
"""Voice assistant loop for the Jetson birdsong chatbot.

Pipeline:  wake word (openWakeWord) -> record utterance -> speech-to-text
(faster-whisper) -> Gemma 4 E2B (llama.cpp OpenAI API) -> text-to-speech
(Piper) -> Bluetooth speaker.

Audio I/O goes through PipeWire's command-line tools (parecord / paplay) so we
don't need PortAudio. Capture is 16 kHz mono s16le, which is what both
openWakeWord and Whisper expect.

Run on the Jetson (with the LLM server already up via scripts/start-llm.sh):

    .venv-voice/bin/python voice/assistant.py

Say the wake word ("hey jarvis" by default), wait for the beep, then speak.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave

import numpy as np
from openai import OpenAI
from openwakeword.model import Model as WakeModel
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80 ms; openWakeWord's expected frame size
CHUNK_BYTES = CHUNK_SAMPLES * 2

# Bluetooth A2DP has real wake-from-idle latency: after a moment of silence the
# link/sink needs time to resume, and whatever audio is sent during that window
# gets clipped. A short silent lead-in on the *same* playback (beep, or the
# first sentence of a reply) gives that latency something harmless to eat
# instead of the tone or the first word of speech.
BT_WAKE_LEAD_MS = 300

# Demo overlay (voice/overlay_server.py). Events are fire-and-forget: if the
# overlay isn't running, emit() fails silently and the assistant is unaffected.
# Set BIRDSONG_OVERLAY="" to disable entirely.
OVERLAY_URL = os.environ.get("BIRDSONG_OVERLAY", "http://127.0.0.1:8095/event")

# Recent BirdNET detections, written by birds/detector.py. If the detector
# isn't running the file simply doesn't exist and Birdy answers from general
# knowledge only.
RECENT_BIRDS_JSON = os.environ.get("BIRDS_RECENT_JSON", "/tmp/birdsong_recent.json")


def emit(kind: str, **data) -> None:
    """Push a status event to the demo overlay (best-effort, never raises)."""
    if not OVERLAY_URL:
        return
    try:
        body = json.dumps({"kind": kind, **data}).encode("utf-8")
        req = urllib.request.Request(
            OVERLAY_URL, data=body, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=0.3).close()
    except Exception:  # noqa: BLE001 - overlay is optional; never disturb the voice loop
        pass

SYSTEM_PROMPT = (
    "You are Birdy, a helpful voice assistant that knows a lot about birds. "
    "You are speaking out loud, so keep answers short and conversational: 1-3 sentences, "
    "plain spoken language, no lists or markdown. Answer confidently from your general "
    "knowledge of birds (identification, behavior, habitat, songs, range, seasonality). "
    "For example, if asked about hummingbirds in San Francisco, name likely species such as "
    "Anna's and Allen's hummingbirds. Only say you are unsure for genuinely ambiguous questions. "
    "Do not greet the user or introduce yourself; just answer the question directly. "
    "A bird-song detector shares your microphone; when your context lists birds heard nearby "
    "recently, use it to answer questions like 'what bird was that?' (most recent first). "
    "If asked about detections and none are listed, say you have not heard any birds lately."
)


def recent_birds_context() -> str:
    """One line describing birds BirdNET heard recently (see birds/detector.py).

    Returns "" when the detector isn't running or nothing was heard lately, so
    the caller can skip injecting anything.
    """
    try:
        with open(RECENT_BIRDS_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return ""
    now = time.time()
    parts = []
    for bird in data.get("birds", [])[:6]:
        age_min = int(max(0.0, now - float(bird.get("ts", 0))) // 60)
        if age_min > 20:
            continue
        when = "just now" if age_min == 0 else f"{age_min} min ago"
        conf = int(round(float(bird.get("confidence", 0)) * 100))
        parts.append(f"{bird['common_name']} ({when}, {conf}% confidence)")
    if not parts:
        return ""
    return "Birds heard nearby recently, most recent first: " + "; ".join(parts) + "."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Birdsong voice assistant")
    p.add_argument("--base-url", default=os.environ.get("BIRDSONG_BASE_URL", "http://127.0.0.1:8080/v1"))
    p.add_argument("--model", default=os.environ.get("BIRDSONG_MODEL", "gemma-4-E2B-it-Q4_K_S.gguf"))
    p.add_argument("--wake-model", default=os.environ.get("BIRDSONG_WAKE", "hey_jarvis"),
                   help="openWakeWord model name, e.g. hey_jarvis, alexa, hey_mycroft")
    p.add_argument("--wake-threshold", type=float, default=float(os.environ.get("BIRDSONG_WAKE_THRESH", "0.5")))
    p.add_argument("--source", default=os.environ.get("BIRDSONG_SOURCE", ""),
                   help="PulseAudio/PipeWire source name (default: system default source)")
    p.add_argument("--piper-bin", default=os.environ.get("BIRDSONG_PIPER", "./piper/piper"))
    p.add_argument("--piper-voice", default=os.environ.get("BIRDSONG_VOICE", "voices/en_US-lessac-medium.onnx"))
    p.add_argument("--whisper-model", default=os.environ.get("BIRDSONG_WHISPER", "base.en"))
    p.add_argument("--max-record", type=float, default=10.0, help="max seconds to record an utterance")
    p.add_argument("--silence-hang", type=float, default=0.7, help="seconds of silence that ends an utterance")
    p.add_argument("--max-tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=float(os.environ.get("BIRDSONG_TEMP", "0.6")),
                   help="sampling temperature; higher = more playful/varied phrasing")
    p.add_argument("--follow-up", type=float, default=6.0,
                   help="seconds to keep listening for a follow-up (no wake word) after a reply; 0 disables")
    return p.parse_args()


def make_beep(path: str, freq: int = 660, ms: int = 160, volume: float = 0.3,
              lead_silence_ms: int = BT_WAKE_LEAD_MS) -> None:
    """Write a short tone to `path`, preceded by silence (see BT_WAKE_LEAD_MS)."""
    n = int(SAMPLE_RATE * ms / 1000)
    n_lead = int(SAMPLE_RATE * lead_silence_ms / 1000)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframesraw(b"\x00\x00" * n_lead)
        for i in range(n):
            # brief fade in/out to avoid clicks
            env = min(1.0, i / 200, (n - i) / 200)
            s = int(volume * env * 32767 * np.sin(2 * np.pi * freq * i / SAMPLE_RATE))
            w.writeframesraw(struct.pack("<h", s))


def _prepend_silence(wav_path: str, ms: int) -> None:
    """Prepend `ms` of silence to an existing wav file in place (see BT_WAKE_LEAD_MS)."""
    with wave.open(wav_path, "rb") as r:
        params = r.getparams()
        frames = r.readframes(r.getnframes())
    n_lead = int(params.framerate * ms / 1000)
    silence = b"\x00" * (n_lead * params.nchannels * params.sampwidth)
    with wave.open(wav_path, "wb") as w:
        w.setparams(params)
        w.writeframes(silence + frames)


def play(path: str) -> None:
    subprocess.run(["paplay", path], check=False)


class MicStream:
    """Continuous 16 kHz mono capture via parecord, read in fixed chunks.

    parecord can wedge: it connects to the audio server but its recording
    stream never materializes (seen after boot races and WirePlumber
    restarts), leaving a silent pipe and a "deaf" service. The constructor
    therefore probes that audio actually flows, respawning until it does.
    """

    def __init__(self, source: str = ""):
        self.cmd = ["parecord", "--format=s16le", "--rate=%d" % SAMPLE_RATE,
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

    def read_chunk(self) -> bytes:
        buf = b""
        while len(buf) < CHUNK_BYTES:
            readable, _, _ = select.select([self.proc.stdout], [], [], 10.0)
            part = self.proc.stdout.read1(CHUNK_BYTES - len(buf)) if readable else b""
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

    def flush(self, seconds: float = 0.4) -> None:
        """Discard any buffered audio (e.g. the tail of our own TTS playback)."""
        deadline = time.time() + seconds
        fd = self.proc.stdout
        while time.time() < deadline:
            r, _, _ = select.select([fd], [], [], 0.05)
            if not r:
                break
            fd.read1(CHUNK_BYTES)

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


def record_utterance(mic: MicStream, max_record: float, silence_hang: float,
                     wait_for_speech: float = 6.0) -> np.ndarray:
    """Record until a hang of silence follows detected speech (adaptive threshold)."""
    # Quick ambient-noise calibration (~0.16s) so we start capturing speech
    # almost immediately after the beep. The floor is clamped below, so a word
    # spoken during calibration can't push the threshold out of range.
    noise = []
    for _ in range(2):
        c = np.frombuffer(mic.read_chunk(), dtype=np.int16)
        noise.append(rms(c))
    # Clamp the floor so a noisy moment (e.g. speaker tail) can't push the
    # speech threshold above normal speaking level (~1500+ RMS on this mic).
    floor = min(500.0, max(100.0, float(np.median(noise)) if noise else 100.0))
    start_thresh = floor * 1.5
    silence_thresh = floor * 1.2

    frames: list[np.ndarray] = []
    started = False
    t0 = time.time()
    silence_start = None
    peak = 0.0
    while time.time() - t0 < max_record:
        c = np.frombuffer(mic.read_chunk(), dtype=np.int16)
        level = rms(c)
        peak = max(peak, level)
        if not started:
            if level > start_thresh:
                started = True
                frames.append(c)
            elif time.time() - t0 > wait_for_speech:
                break  # no speech began
            continue
        frames.append(c)
        if level < silence_thresh:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= silence_hang:
                break
        else:
            silence_start = None

    dur = sum(f.size for f in frames) / SAMPLE_RATE
    print(f"[rec] floor={floor:.0f} start_thr={start_thresh:.0f} peak_rms={peak:.0f} "
          f"speech={'yes' if started else 'no'} dur={dur:.1f}s")
    if not frames:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(frames).astype(np.float32) / 32768.0
    return audio


def synth_and_play(args: argparse.Namespace, text: str, lead_pad: bool = False) -> None:
    """Synthesize `text` with Piper and play it. Set lead_pad=True for the first
    utterance after a period of silence (see BT_WAKE_LEAD_MS)."""
    if not text.strip():
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        subprocess.run(
            [args.piper_bin, "--model", args.piper_voice, "--output_file", wav_path],
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        if lead_pad:
            _prepend_silence(wav_path, BT_WAKE_LEAD_MS)
        play(wav_path)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def speak_streaming(client: OpenAI, model: str, history: list[dict], max_tokens: int,
                    args: argparse.Namespace) -> str:
    """Stream the reply and synthesize speech sentence-by-sentence so audio starts
    sooner. Returns the full reply text."""
    t_req = time.time()
    stream = client.chat.completions.create(
        model=model,
        messages=history,
        max_tokens=max_tokens,
        temperature=args.temperature,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    full: list[str] = []
    pending = ""
    sent_first = False
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        if not full:  # first token of the reply
            emit("state", status="speaking")
            emit("metric", ttfw=round(time.time() - t_req, 2))
        emit("token", text=delta)
        full.append(delta)
        pending += delta
        # Flush complete sentences to TTS as they arrive.
        while True:
            idx = _sentence_break(pending)
            if idx is None:
                break
            sentence, pending = pending[:idx + 1].strip(), pending[idx + 1:]
            if sentence:
                # Only the very first sentence needs the wake-up pad; by the
                # time later sentences play the Bluetooth link is already warm.
                synth_and_play(args, sentence, lead_pad=not sent_first)
                sent_first = True
    if pending.strip():
        synth_and_play(args, pending.strip(), lead_pad=not sent_first)
    return "".join(full).strip()


def _sentence_break(text: str):
    """Index of the first sentence-ending punctuation once we have enough text."""
    for i, ch in enumerate(text):
        if ch in ".!?" and i >= 12:
            # avoid splitting on decimals like "3.5"
            if ch == "." and i + 1 < len(text) and text[i + 1].isdigit():
                continue
            return i
    return None


def main() -> int:
    args = parse_args()

    print("Loading models (wake word + Whisper)...")
    wake = WakeModel(wakeword_models=[args.wake_model], inference_framework="onnx")
    stt = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "not-needed"))

    beep_path = os.path.join(tempfile.gettempdir(), "birdsong_beep.wav")
    make_beep(beep_path, freq=660)  # start-listening cue (higher pitch, rising)
    stop_beep_path = os.path.join(tempfile.gettempdir(), "birdsong_stop_beep.wav")
    make_beep(stop_beep_path, freq=330, ms=200)  # stop-listening cue (lower pitch)

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    mic = MicStream(args.source)
    # openWakeWord keys its prediction dict by the model's base name (no dir or
    # extension), whether you pass a built-in name ("hey_jarvis") or a path to a
    # custom-trained model ("voice/models/hey_birdy.onnx").
    wake_key = os.path.splitext(os.path.basename(args.wake_model))[0]
    wake_label = wake_key.replace("_", " ")

    def transcribe(audio: np.ndarray) -> str:
        if os.environ.get("BIRDSONG_DEBUG_WAV"):
            dbg = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
            with wave.open("/tmp/last_utt.wav", "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
                w.writeframes(dbg.tobytes())
        segs, _ = stt.transcribe(audio, language="en", beam_size=1, vad_filter=True)
        return " ".join(s.text for s in segs).strip()

    # Common Whisper hallucinations on near-silent audio.
    JUNK = {"you", "you.", "thanks for watching", "thanks for watching.", ".", "uh", "um"}

    def handle_turn(audio: np.ndarray) -> bool:
        """STT -> Gemma -> TTS for one utterance. Returns False to end the conversation."""
        t_turn = time.time()
        emit("state", status="thinking")
        text = transcribe(audio)
        print(f"[you] {text}")
        if not text or text.lower().strip() in JUNK:
            # Likely a false speech-detection (ambient noise), not a real question.
            # End the listening session here rather than looping for another
            # follow-up round -- otherwise a noisy room can trigger a cascade of
            # "listening again" beeps that never seem to stop.
            print("[skip] no usable speech (noise/hallucination) - ending turn")
            return False
        if text.lower().strip(" .!?") in {"stop", "quit", "exit", "goodbye", "never mind", "that's all", "thank you"}:
            emit("prompt", text=text)
            emit("done", text="Okay, talk soon.")
            synth_and_play(args, "Okay, talk soon.", lead_pad=True)
            return False
        emit("prompt", text=text)
        history.append({"role": "user", "content": text})
        # Transient copy of the conversation with live BirdNET detections folded
        # into the system prompt -- fresh every turn, never stored in history.
        messages = list(history)
        birds_ctx = recent_birds_context()
        if birds_ctx:
            messages[0] = {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + birds_ctx}
        try:
            reply = speak_streaming(client, args.model, messages, args.max_tokens, args)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] LLM request failed: {exc}", file=sys.stderr)
            emit("error", message="Could not reach the language model.")
            synth_and_play(args, "Sorry, I could not reach the language model.", lead_pad=True)
            history.pop()
            return True
        print(f"[birdsong] {reply}")
        emit("done", text=reply)
        emit("metric", total=round(time.time() - t_turn, 2))
        history.append({"role": "assistant", "content": reply})
        if len(history) > 13:  # keep context bounded
            history[:] = [history[0]] + history[-12:]
        return True

    print(f'Listening for wake word: "{wake_label}"  (Ctrl+C to quit)')
    emit("hello", model=args.model, wake=wake_label)
    try:
        while True:
            chunk = mic.read_chunk()
            if len(chunk) < CHUNK_BYTES:
                continue
            scores = wake.predict(np.frombuffer(chunk, dtype=np.int16))
            if scores.get(wake_key, 0.0) < args.wake_threshold:
                continue

            # Wake detected -> converse until a listening window passes in silence.
            wake.reset()
            print("\n[wake] listening...")
            # Re-announce model/wake word: the overlay server loses them if it
            # restarted after our startup "hello" (footer shows an empty model).
            emit("hello", model=args.model, wake=wake_label)
            first = True
            while True:
                play(beep_path)
                # Flush the beep's own echo before calibrating ambient noise. Bluetooth
                # A2DP playback has real transmission latency, so audio keeps trickling
                # out of the speaker for a bit after paplay's process returns; too short
                # a flush lets that tail get measured as "ambient noise" and inflates the
                # speech threshold above normal talking volume.
                mic.flush(0.35)
                emit("state", status="listening")
                wait = 6.0 if first else args.follow_up
                if not first:
                    print(f"[listening for follow-up ~{wait:.0f}s]")
                audio = record_utterance(mic, args.max_record, args.silence_hang, wait_for_speech=wait)
                if audio.size < SAMPLE_RATE // 2:  # timed out with no speech -> stop listening
                    print("[listen] timed out, no speech detected")
                    break
                cont = handle_turn(audio)
                mic.flush(0.6)  # avoid hearing our own speech tail
                if not cont or args.follow_up <= 0:
                    break
                first = False

            # Exactly one "done listening" cue, whichever way the session ended
            # (timeout, stop word, or a false speech-detection with nothing usable).
            play(stop_beep_path)
            mic.flush(0.2)
            wake.reset()
            emit("state", status="idle")
            print(f'\nListening for wake word: "{wake_label}"')
    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        emit("state", status="offline")
        mic.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
