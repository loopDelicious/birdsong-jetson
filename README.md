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

## What’s next (not today)

- Audio / video capture and bird detection
- Injecting live detections into the chat context
- Multimodal prompting (E2B supports image/audio; prefer vLLM when audio is required)
