"""Streaming client for Ollama /api/generate."""

import json
import os
import time

import httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def generate(
    model: str,
    prompt: str,
    num_predict: int = 128,
    num_ctx: int = 2048,
    timeout: float = 120.0,
) -> dict:
    """Send a streaming generate request and return timing/token metrics."""
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": num_predict, "num_ctx": num_ctx},
    }

    t_start = time.perf_counter()
    ttft_s = None

    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if ttft_s is None and chunk.get("response"):
                    ttft_s = time.perf_counter() - t_start
                if chunk.get("done"):
                    total_s = time.perf_counter() - t_start
                    eval_count = chunk.get("eval_count", 0)
                    eval_duration_s = chunk.get("eval_duration", 0) / 1e9
                    load_duration_s = chunk.get("load_duration", 0) / 1e9
                    tokens_per_s = (
                        eval_count / eval_duration_s if eval_duration_s > 0 else 0.0
                    )
                    return {
                        "ttft_s": ttft_s,
                        "total_s": total_s,
                        "eval_count": eval_count,
                        "eval_duration_s": eval_duration_s,
                        "load_duration_s": load_duration_s,
                        "tokens_per_s": tokens_per_s,
                    }

    raise RuntimeError("Stream ended without done=true")
