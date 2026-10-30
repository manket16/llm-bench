"""Utilities for tracking model memory usage during inference."""

import os
import psutil


def cpu_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def gpu_memory_mb() -> dict[int, float]:
    """Return {gpu_id: used_mb} for each visible GPU, or empty dict if none."""
    try:
        import GPUtil
        return {g.id: g.memoryUsed for g in GPUtil.getGPUs()}
    except Exception:
        return {}


def snapshot() -> dict:
    return {"cpu_mb": cpu_memory_mb(), "gpu": gpu_memory_mb()}


def delta(before: dict, after: dict) -> dict:
    gpu_delta = {
        k: after["gpu"].get(k, 0) - before["gpu"].get(k, 0)
        for k in after["gpu"]
    }
    return {
        "cpu_mb": after["cpu_mb"] - before["cpu_mb"],
        "gpu": gpu_delta,
    }
