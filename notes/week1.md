# Week 1 — 2026-10-30

## Goals
- Stand up the benchmarking harness
- Baseline latency and streaming numbers for Anthropic and OpenAI models
- Understand KV-cache memory footprint for common open models

## Setup notes
- Created `bench/`, `tools/`, `notes/` structure
- `bench/call_model.py` — end-to-end latency (non-streaming)
- `bench/stream_model.py` — TTFT and token throughput (streaming)
- `tools/model_memory.py` — CPU/GPU memory snapshots
- `tools/kv_cache.py` — KV-cache size estimates

## Early observations
- [ ] Run baseline on `claude-haiku-5-5` and `gpt-4o-mini`
- [ ] Compare TTFT across 5 prompts
- [ ] Estimate KV-cache MB for llama-3-8b at seq_len 2048 vs 8192

## Model size: predicted vs real

Predicted size = parameters × 2 bytes (fp16 baseline).
Real size = default Ollama GGUF quantisation (Q4\_K\_M unless noted).
Bytes/param = real size ÷ parameter count.

| Model | Parameters | Predicted (fp16) | Real (Q4\_K\_M) | Bytes / param |
|---|---:|---:|---:|---:|
| llama3.2:1b | 1.24 B | 2.5 GB | 0.77 GB | 0.62 |
| llama3.2:3b | 3.21 B | 6.4 GB | 1.88 GB | 0.59 |
| llama3.1:8b | 8.03 B | 16.1 GB | 4.92 GB | 0.61 |
| mistral:7b | 7.24 B | 14.5 GB | 4.11 GB | 0.57 |
| llama3.1:70b | 70.6 B | 141.2 GB | 40.0 GB | 0.57 |

Q4\_K\_M packs ~4.5 bits per weight, so the expected ratio is 4.5 / 16 ≈ 0.28 of fp16 —
real bytes/param land around 0.57–0.62 because embeddings and norms stay in fp16.

## Next steps
- Add result logging to JSONL
- Wire `tools/model_memory.py` into bench runners for per-call memory delta
