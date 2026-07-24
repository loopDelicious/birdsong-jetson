# Birdsong Jetson

Local chatbot scaffold for an NVIDIA Jetson Orin Nano, using **Gemma 4 E2B** through NVIDIA’s **llama.cpp** container.

Develop on a Mac, then copy the project to the Jetson over SSH. The CUDA/GPU stack runs only on the Jetson.

Later this repo will grow into a multimodal (audio + video) bird detection station you can talk to about what shows up at the feeder. Today is text chat only.

## Architecture

```
Mac (edit)  --rsync/scp-->  Jetson Orin Nano
                              └─ Docker: llama.cpp (Gemma 4 E2B Q4_K_S) :8080
                              └─ Python CLI → OpenAI-compatible /v1 API
```

| Piece | Choice |
| --- | --- |
| Model | `unsloth/gemma-4-E2B-it-GGUF` (`Q4_K_S`, ~3GB) |
| Runtime | `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin` |
| Chat UI | Browser at `http://127.0.0.1:8080` or `python chat/chat.py` |

Ollama is intentionally not used for Gemma 4 on Orin Nano right now (unreliable per NVIDIA Jetson AI Lab). When audio input matters later, vLLM is the better path for E2B.

### Why not just `-hf`?

The one-liner `llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S` also downloads the
vision projector and uses a large default context. On the Orin Nano's 8GB unified
memory that often fails at load time with `unable to allocate CUDA0 buffer`. So
[`scripts/start-llm.sh`](scripts/start-llm.sh) instead downloads the GGUF once to a
host dir, mounts it, and launches with `--no-mmproj -c 2048 -ngl 99` (text-only, GPU
offloaded). Measured ~19 tokens/sec on GPU.

Gemma 4 also enables a chain-of-thought "thinking" mode by default. The chat client
disables it for snappy answers; toggle it in-session with `/think`.

## Prerequisites (Jetson)

- Jetson Orin Nano with **JetPack 6** (L4T r36.x)
- Docker with the **NVIDIA container runtime**
- Enough free disk for the GGUF + Docker image (**NVMe SSD strongly recommended**; eMMC is painful for model downloads)
- Network on first run (model pull from Hugging Face)

Optional performance:

```bash
sudo nvpmodel -m 0    # MAXN / Super mode where available
sudo jetson_clocks
```

The Orin Nano has **8GB unified memory**. Close heavy apps while the model loads; avoid relying on swap on eMMC.

## Copy from Mac → Jetson

From this repo on your Mac (replace user and host):

```bash
rsync -avz --exclude '.git' --exclude '.venv' ./ joyce@JETSON_IP:~/birdsong-jetson/
```

Or with `scp`:

```bash
scp -r README.md scripts chat .gitignore joyce@JETSON_IP:~/birdsong-jetson/
```

SSH in:

```bash
ssh joyce@JETSON_IP
cd ~/birdsong-jetson
```

## Run on the Jetson

### 1. Start the LLM server

```bash
chmod +x scripts/*.sh
./scripts/start-llm.sh
```

The script stops Ollama (frees memory), downloads the GGUF to `~/gguf` on first run,
then launches the container. Watch until you see `server is listening` and `model loaded`:

```bash
docker logs -f birdsong-llm
```

When ready:

- Browser UI: `http://127.0.0.1:8080` (or `http://JETSON_IP:8080` from another machine on the LAN)
- API base: `http://127.0.0.1:8080/v1`

Stop:

```bash
./scripts/stop-llm.sh
# REMOVE=1 ./scripts/stop-llm.sh   # also delete the container
```

### 2. Chat from the terminal

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r chat/requirements.txt
python chat/chat.py
```

Useful flags / env:

```bash
python chat/chat.py --base-url http://127.0.0.1:8080/v1
# or: export BIRDSONG_BASE_URL=http://127.0.0.1:8080/v1
```

In-chat commands: `/help`, `/clear`, `/think` (toggle chain-of-thought), `/quit`.

If the default `--model` name does not match what llama-server reports, list models:

```bash
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool
```

Then pass the `id` with `--model`.

## Repo layout

```
birdsong-jetson/
├── README.md
├── .gitignore
├── scripts/
│   ├── start-llm.sh    # Jetson: start llama.cpp + E2B
│   └── stop-llm.sh
└── chat/
    ├── requirements.txt
    └── chat.py         # streaming CLI against /v1
```

## Audio setup (for the future voice loop)

Groundwork for a spoken assistant: speaker output over Bluetooth + a USB mic for input.

### The one gotcha: NVIDIA disables Bluetooth audio by default

Jetson ships `/lib/systemd/system/bluetooth.service.d/nv-bluetooth-service.conf`
which starts `bluetoothd` with `--noplugin=audio,a2dp,avrcp`. That removes the
`org.bluez.Media1` interface, so any Bluetooth speaker pairs but fails to connect
its audio profile (`br-connection-profile-unavailable`). Override it:

```bash
sudo mkdir -p /etc/systemd/system/bluetooth.service.d
sudo tee /etc/systemd/system/bluetooth.service.d/nv-bluetooth-service.conf >/dev/null <<'EOF'
[Service]
ExecStart=
ExecStart=/usr/lib/bluetooth/bluetoothd
EOF
sudo systemctl daemon-reload
sudo systemctl restart bluetooth
```

### Audio stack

Use PipeWire (it handles Bluetooth A2DP more reliably than the half-installed
PulseAudio that ships by default):

```bash
sudo dpkg --configure -a                       # if a prior apt run was interrupted
sudo apt install -y pipewire pipewire-pulse wireplumber libspa-0.2-bluetooth
systemctl --user --now disable pulseaudio.service pulseaudio.socket
systemctl --user mask pulseaudio.service pulseaudio.socket
systemctl --user --now enable pipewire pipewire-pulse wireplumber
sudo reboot
```

`libspa-0.2-bluetooth` is required for PipeWire Bluetooth audio.

### Pair a Bluetooth speaker (example: JBL Flip 5)

Put the speaker in pairing mode (blinking), then:

```bash
export XDG_RUNTIME_DIR=/run/user/1000
bluetoothctl --timeout 12 scan on            # find its MAC
bluetoothctl pair <MAC>
bluetoothctl trust <MAC>                      # auto-reconnect on boot
bluetoothctl connect <MAC>
pactl set-default-sink bluez_output.<MAC_with_underscores>.a2dp-sink
pactl set-sink-volume  bluez_output.<MAC_with_underscores>.a2dp-sink 25%
spd-say "audio path is working"               # quick spoken test
```

The USB sound device (e.g. C-Media / PCM2902) shows up as a microphone
(`arecord -l`) and is the input side for the future speech-to-text step.

## Voice assistant (wake word -> speak)

Talk to the chatbot hands-free: say a wake word, ask a question, hear the answer
through the Bluetooth speaker, and keep a short back-and-forth conversation.

```
mic (USB) -> openWakeWord -> record -> Whisper (STT) -> Gemma -> Piper (TTS) -> speaker (Bluetooth)
```

| Stage | Tool | Notes |
| --- | --- | --- |
| Wake word | [openWakeWord](https://github.com/dscripka/openWakeWord) | default phrase "hey jarvis" (CPU) |
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | `base.en`, int8, CPU |
| LLM | Gemma 4 E2B | the same llama.cpp server on `:8080` |
| Text-to-speech | [Piper](https://github.com/rhasspy/piper) | aarch64 binary + `en_US-lessac-medium` voice |
| Audio I/O | PipeWire `parecord` / `paplay` | no PortAudio build needed |

### Setup (on the Jetson)

Requires the audio setup above (PipeWire + Bluetooth speaker + USB mic) and the
LLM server running.

```bash
./scripts/setup-voice.sh          # venv + deps + Piper binary/voice + model downloads
./scripts/start-llm.sh            # ensure Gemma is up
./scripts/install-voice-service.sh
```

`setup-voice.sh` creates `.venv-voice/` and downloads Piper into `piper/` and voices
into `voices/` (all git-ignored). `install-voice-service.sh` runs the assistant as a
`systemd --user` service (`birdsong-voice`) that auto-restarts and starts on login.

### Use it

1. Say the wake word: **"hey birdy"** (custom openWakeWord model at `voice/models/hey_birdy.onnx`)
2. Wait for the beep, then ask (e.g. "What hummingbirds live in San Francisco?")
3. After the reply + beep, ask a follow-up with **no wake word** (short conversation window)
4. Say "thank you" / "stop" / "goodbye", or just stay silent, to end the turn

Run manually instead of as a service:

```bash
.venv-voice/bin/python voice/assistant.py           # options: --wake-model, --follow-up, --max-tokens ...
```

Watch logs / control the service:

```bash
tail -f voice.log
systemctl --user restart birdsong-voice
```

Tuning knobs (env or flags): `BIRDSONG_WAKE` (wake model, e.g. `alexa`, `hey_mycroft`),
`--wake-threshold`, `--follow-up` (seconds; `0` disables continuous mode),
`--silence-hang`, `--max-tokens`. A custom wake phrase like "hey birdsong" needs a
trained openWakeWord model (or Picovoice Porcupine).

## Demo overlay (watch the conversation live)

A small web page that shows the assistant's live status (idle / listening /
thinking / speaking), the transcribed prompt, and Birdy's reply streaming in
word-by-word — handy for demos and livestreams since the Jetson runs headless.
Inspired by NVIDIA's [live-vlm-webui](https://github.com/nvidia-ai-iot/live-vlm-webui).

```bash
# one-time install on the Jetson (systemd --user service, starts on boot)
./scripts/install-overlay-service.sh
```

Then open **`http://jetson-desktop.local:8095`** from any machine on your
network (e.g. your Mac). For a livestream, add that URL as an OBS Browser Source.

How it works: `voice/assistant.py` fires small JSON events (state changes,
prompt, response tokens, latency) at `voice/overlay_server.py` (stdlib-only,
port 8095), which relays them to the browser over Server-Sent Events. The
overlay is optional — if it isn't running, the assistant works normally.
Disable event emission with `BIRDSONG_OVERLAY=""`.

## Bird detection (BirdNET, shares the mic)

Continuous bird-song identification, ported from the Raspberry Pi
[birdsong](https://github.com/loopDelicious/birdsong) project. BirdNET (tflite)
analyzes 3-second windows from the **same USB mic** the voice assistant uses —
PipeWire multiplexes one capture device across any number of clients, so the
detector's 48 kHz stream and the assistant's 16 kHz stream coexist happily.
BirdNET runs on the CPU (a few hundred MB), so nothing changes for the LLM.

```bash
# one-time setup on the Jetson (separate venv: tflite needs numpy<2)
./scripts/setup-birds.sh
./scripts/install-birds-service.sh   # systemd --user service, starts on boot
```

Each detection is:

- printed to `birds.log` and logged to SQLite (`birds/birdsong.db`)
- shown live in the demo overlay's "Birds heard" panel
- written to `/tmp/birdsong_recent.json`, which the voice assistant folds into
  Birdy's context — so after a detection you can ask *"hey Birdy, what bird did
  you just hear?"*

Tuning knobs (flags on `birds/detector.py`, or env in the service file):
`--min-conf` (default 0.5), `--lat/--lon` + `--location` (species plausibility
filter, defaults to San Francisco and on; `--no-location` to disable).

## What’s next (not today)

- USB webcam at a feeder (close range) for visual verification via E2B's vision
- Multimodal prompting (E2B supports image/audio; prefer vLLM when audio is required)
