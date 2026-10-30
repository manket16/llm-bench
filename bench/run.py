"""Command-line benchmark runner.

Usage:
    python -m bench.run --model llama3.2:1b --runs 5 --label laptop-mx250
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from bench.client import generate
from bench.stats import percentiles

logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_PROMPT = "Why is the sky blue? Answer in two sentences."


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM inference benchmark")
    p.add_argument("--model", required=True, help="Ollama model tag, e.g. llama3.2:1b")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--num-predict", type=int, default=128)
    p.add_argument("--num-ctx", type=int, default=2048)
    p.add_argument("--label", default="unknown", help="Hardware label, e.g. laptop-mx250")
    p.add_argument("--out", default="results/runs.jsonl")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        log.info("Warm-up run (excluded from results)…")
        generate(args.model, args.prompt, args.num_predict, args.num_ctx)
        log.info("Warm-up complete.")

        results = []
        for i in range(args.runs):
            log.info("Run %d/%d…", i + 1, args.runs)
            r = generate(args.model, args.prompt, args.num_predict, args.num_ctx)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "label": args.label,
                "prompt": args.prompt,
                "num_predict": args.num_predict,
                "num_ctx": args.num_ctx,
                **r,
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            results.append(r)

    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        log.error("Connection failed: %s", exc)
        sys.exit(1)

    ttft_p = percentiles([r["ttft_s"] for r in results if r["ttft_s"] is not None])
    tps_p = percentiles([r["tokens_per_s"] for r in results])

    sep = "─" * 42
    print(f"\n{sep}")
    print(f"  Model : {args.model}   Label: {args.label}")
    print(f"  Runs  : {args.runs}    Output: {out_path}")
    print(sep)
    print(f"  {'Metric':<22} {'p50':>8} {'p95':>8}")
    print(f"  {'TTFT (s)':<22} {ttft_p['p50']:>8.3f} {ttft_p['p95']:>8.3f}")
    print(f"  {'tokens/s':<22} {tps_p['p50']:>8.1f} {tps_p['p95']:>8.1f}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
