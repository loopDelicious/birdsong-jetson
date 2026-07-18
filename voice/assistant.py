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
import os
import select
import struct
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np
from openai import OpenAI
from openwakeword.model import Model as WakeModel
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80 ms; openWakeWord's expected frame size
CHUNK_BYTES = CHUNK_SAMPLES * 2

SYSTEM_PROMPT = (
    "You are Birdsong, a friendly voice assistant that knows a lot about birds. "
    "You are speaking out loud, so keep answers short and conversational: 1-3 sentences, "
    "plain spoken language, no lists or markdown. Answer confidently from your general "
    "knowledge of birds (identification, behavior, habitat, songs, range, seasonality). "
    "For example, if asked about hummingbirds in San Francisco, name likely species such as "
    "Anna's and Allen's hummingbirds. Only say you are unsure for genuinely ambiguous questions."
)


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
    p.add_argument("--follow-up", type=float, default=6.0,
                   help="seconds to keep listening for a follow-up (no wake word) after a reply; 0 disables")
    return p.parse_args()


def make_beep(path: str, freq: int = 660, ms: int = 160, volume: float = 0.3) -> None:
    n = int(SAMPLE_RATE * ms / 1000)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        for i in range(n):
            # brief fade in/out to avoid clicks
            env = min(1.0, i / 200, (n - i) / 200)
            s = int(volume * env * 32767 * np.sin(2 * np.pi * freq * i / SAMPLE_RATE))
            w.writeframesraw(struct.pack("<h", s))


def play(path: str) -> None:
    subprocess.run(["paplay", path], check=False)


class MicStream:
    """Continuous 16 kHz mono capture via parecord, read in fixed chunks."""

    def __init__(self, source: str = ""):
        cmd = ["parecord", "--format=s16le", "--rate=%d" % SAMPLE_RATE, "--channels=1", "--raw"]
        if source:
            cmd += ["--device", source]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read_chunk(self) -> bytes:
        buf = b""
        while len(buf) < CHUNK_BYTES:
            part = self.proc.stdout.read(CHUNK_BYTES - len(buf))
            if not part:
                break
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
            fd.read(CHUNK_BYTES)

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
    # Calibrate ambient noise for ~0.3s.
    noise = []
    for _ in range(4):
        c = np.frombuffer(mic.read_chunk(), dtype=np.int16)
        noise.append(rms(c))
    # Clamp the floor so a noisy moment (e.g. speaker tail) can't push the
    # speech threshold above normal speaking level (~1500+ RMS on this mic).
    floor = min(500.0, max(100.0, float(np.median(noise)) if noise else 100.0))
    start_thresh = floor * 1.8
    silence_thresh = floor * 1.4

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


def synth_and_play(args: argparse.Namespace, text: str) -> None:
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
    stream = client.chat.completions.create(
        model=model,
        messages=history,
        max_tokens=max_tokens,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    full: list[str] = []
    pending = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        full.append(delta)
        pending += delta
        # Flush complete sentences to TTS as they arrive.
        while True:
            idx = _sentence_break(pending)
            if idx is None:
                break
            sentence, pending = pending[:idx + 1].strip(), pending[idx + 1:]
            if sentence:
                synth_and_play(args, sentence)
    if pending.strip():
        synth_and_play(args, pending.strip())
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
    make_beep(beep_path)

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    mic = MicStream(args.source)
    wake_label = args.wake_model.replace("_", " ")

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
        text = transcribe(audio)
        print(f"[you] {text}")
        if not text:
            return True
        if text.lower().strip() in JUNK:
            print("[skip] likely noise/hallucination")
            return True
        if text.lower().strip(" .!?") in {"stop", "quit", "exit", "goodbye", "never mind", "that's all", "thank you"}:
            synth_and_play(args, "Okay, talk soon.")
            return False
        history.append({"role": "user", "content": text})
        try:
            reply = speak_streaming(client, args.model, history, args.max_tokens, args)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] LLM request failed: {exc}", file=sys.stderr)
            synth_and_play(args, "Sorry, I could not reach the language model.")
            history.pop()
            return True
        print(f"[birdsong] {reply}")
        history.append({"role": "assistant", "content": reply})
        if len(history) > 13:  # keep context bounded
            history[:] = [history[0]] + history[-12:]
        return True

    print(f'Listening for wake word: "{wake_label}"  (Ctrl+C to quit)')
    try:
        while True:
            chunk = mic.read_chunk()
            if len(chunk) < CHUNK_BYTES:
                continue
            scores = wake.predict(np.frombuffer(chunk, dtype=np.int16))
            if scores.get(args.wake_model, 0.0) < args.wake_threshold:
                continue

            # Wake detected -> converse until a listening window passes in silence.
            wake.reset()
            print("\n[wake] listening...")
            first = True
            while True:
                play(beep_path)
                mic.flush()
                wait = 6.0 if first else args.follow_up
                if not first:
                    print(f"[listening for follow-up ~{wait:.0f}s]")
                audio = record_utterance(mic, args.max_record, args.silence_hang, wait_for_speech=wait)
                if audio.size < SAMPLE_RATE // 2:  # silence -> end conversation
                    break
                cont = handle_turn(audio)
                mic.flush(0.6)  # avoid hearing our own speech tail
                if not cont or args.follow_up <= 0:
                    break
                first = False

            wake.reset()
            print(f'\nListening for wake word: "{wake_label}"')
    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        mic.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
