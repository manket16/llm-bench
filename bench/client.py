"""Streaming client for Ollama /api/generate."""

import json
import os
import time

import httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def gpu_share(model: str, timeout: float = 10.0) -> float | None:
    """Return size_vram / size for *model* from /api/ps, or None if not found."""
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(f"{OLLAMA_URL}/api/ps")
        resp.raise_for_status()
    for entry in resp.json().get("models", []):
        if entry.get("name") == model or entry.get("model") == model:
            size = entry.get("size", 0)
            size_vram = entry.get("size_vram", 0)
            return size_vram / size if size > 0 else None
    return None


def unload(model: str, timeout: float = 30.0) -> None:
    """Ask Ollama to evict *model* from memory (keep_alive=0)."""
    with httpx.Client(timeout=timeout) as client:
        client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
        )


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
