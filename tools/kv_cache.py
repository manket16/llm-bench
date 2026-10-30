"""KV-cache size estimation and profiling helpers."""


def estimate_kv_bytes(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2,  # fp16 by default
) -> int:
    """Estimate KV-cache memory in bytes for a transformer model.

    Formula: 2 (K+V) * layers * heads * head_dim * seq_len * dtype_bytes
    """
    return 2 * n_layers * n_heads * head_dim * seq_len * dtype_bytes


def estimate_kv_mb(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2,
) -> float:
    return estimate_kv_bytes(n_layers, n_heads, head_dim, seq_len, dtype_bytes) / 1024 / 1024


# Rough parameter presets for common open models
MODEL_PRESETS: dict[str, dict] = {
    "llama-3-8b":   {"n_layers": 32,  "n_heads": 32, "head_dim": 128},
    "llama-3-70b":  {"n_layers": 80,  "n_heads": 64, "head_dim": 128},
    "mistral-7b":   {"n_layers": 32,  "n_heads": 32, "head_dim": 128},
    "mixtral-8x7b": {"n_layers": 32,  "n_heads": 32, "head_dim": 128},
}


def estimate_for_model(model_key: str, seq_len: int, dtype_bytes: int = 2) -> float:
    params = MODEL_PRESETS[model_key]
    return estimate_kv_mb(seq_len=seq_len, dtype_bytes=dtype_bytes, **params)
