#!/usr/bin/env python3
"""Streaming terminal chat against a local llama.cpp OpenAI-compatible server.

Targets Gemma 4 E2B served by NVIDIA's llama.cpp container on a Jetson Orin Nano.
Gemma 4 enables a chain-of-thought "thinking" mode by default, which streams into
`reasoning_content` and can eat the whole token budget before any answer. We turn
it off by default for snappy replies; toggle it with /think.
"""

from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "gemma-4-E2B-it-Q4_K_S.gguf"

SYSTEM_PROMPT = """You are Birdsong, a friendly on-device assistant for a backyard bird observation station.
You help the user talk about birds they see or hear: identification tips, behavior, habitat, and seasonal patterns.
Be concise and practical. If you are unsure, say so rather than inventing a species.
Later this station will feed you live audio/video detections; for now answer from conversation alone."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the local Gemma 4 E2B server")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BIRDSONG_BASE_URL", DEFAULT_BASE_URL),
        help=f"OpenAI-compatible API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("BIRDSONG_MODEL", DEFAULT_MODEL),
        help=f"Model name as reported by the server (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--system",
        default=SYSTEM_PROMPT,
        help="System prompt override",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable Gemma 4 thinking mode (chain-of-thought) from the start",
    )
    return parser.parse_args()


def stream_reply(client: OpenAI, model: str, messages: list[dict], think: bool) -> str:
    """Stream one assistant turn. Returns the final answer text."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": think}},
    )

    answer: list[str] = []
    in_thinking = False
    for chunk in stream:
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if not in_thinking:
                print("\n\033[2m[thinking] ", end="", flush=True)
                in_thinking = True
            print(reasoning, end="", flush=True)
            continue
        content = delta.content or ""
        if content:
            if in_thinking:
                print("\033[0m\n", end="", flush=True)
                in_thinking = False
            answer.append(content)
            print(content, end="", flush=True)
    if in_thinking:
        print("\033[0m", end="", flush=True)
    print("\n")
    return "".join(answer)


def main() -> int:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "not-needed"))

    messages: list[dict[str, str]] = [{"role": "system", "content": args.system}]
    think = args.think

    print(f"Birdsong chat -> {args.base_url}  (model: {args.model}, thinking: {'on' if think else 'off'})")
    print("Type a message and press Enter. Commands: /quit, /clear, /think, /help\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not user:
            continue
        if user in {"/quit", "/exit", "/q"}:
            print("Bye.")
            return 0
        if user == "/clear":
            messages = [{"role": "system", "content": args.system}]
            print("(conversation cleared)\n")
            continue
        if user == "/think":
            think = not think
            print(f"(thinking mode {'on' if think else 'off'})\n")
            continue
        if user == "/help":
            print("Commands: /quit  /clear  /think  /help\n")
            continue

        messages.append({"role": "user", "content": user})
        print("birdsong> ", end="", flush=True)

        try:
            answer = stream_reply(client, args.model, messages, think)
            messages.append({"role": "assistant", "content": answer})
        except Exception as exc:  # noqa: BLE001 — surface server errors to the user
            messages.pop()
            print(f"\n[error] {exc}\n", file=sys.stderr)
            print("Is the LLM server running? Try: ./scripts/start-llm.sh\n", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
