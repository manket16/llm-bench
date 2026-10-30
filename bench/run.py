"""Command-line benchmark runner.

Usage:
    python -m bench.run --model llama3.2:1b --runs 5 --label laptop-mx250
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from bench.client import generate, gpu_share, unload
from bench.stats import percentiles

logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_PROMPT = "Why is the sky blue? Answer in two sentences."


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM inference benchmark")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="Single Ollama model tag, e.g. llama3.2:1b")
    g.add_argument("--models", nargs="+", metavar="MODEL",
                   help="One or more model tags; each gets --runs measured runs")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--num-predict", type=int, default=128)
    p.add_argument("--num-ctx", type=int, default=2048)
    p.add_argument("--label", default="unknown", help="Hardware label, e.g. laptop-mx250")
    p.add_argument("--out", default="results/runs.jsonl")
    p.add_argument("--csv", default="results/summary.csv",
                   help="CSV file to append one summary row per model per session")
    return p


def _write_csv_row(csv_path: Path, model: str, label: str, runs: int,
                   ttft_p: dict, tps_p: dict, share: float | None) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "model", "label", "runs",
                "ttft_p50_s", "ttft_p95_s", "tps_p50", "tps_p95", "gpu_share",
            ])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            model, label, runs,
            round(ttft_p["p50"], 3), round(ttft_p["p95"], 3),
            round(tps_p["p50"], 1), round(tps_p["p95"], 1),
            round(share, 4) if share is not None else "",
        ])


def _bench_model(model: str, args: argparse.Namespace,
                 out_path: Path) -> list[dict]:
    """Warm up, measure --runs runs, log each to JSONL. Returns result dicts."""
    log.info("[%s] Warm-up run…", model)
    generate(model, args.prompt, args.num_predict, args.num_ctx)
    log.info("[%s] Warm-up complete.", model)

    share = gpu_share(model)
    if share is not None:
        log.info("[%s] GPU share: %.1f%%", model, share * 100)
    else:
        log.info("[%s] GPU share: n/a (model not in /api/ps or no VRAM info)", model)

    results = []
    for i in range(args.runs):
        log.info("[%s] Run %d/%d…", model, i + 1, args.runs)
        r = generate(model, args.prompt, args.num_predict, args.num_ctx)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "label": args.label,
            "prompt": args.prompt,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
            "gpu_share": share,
            **r,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        results.append(r)

    return results


def _print_summary(model: str, args: argparse.Namespace, out_path: Path,
                   results: list[dict], share: float | None) -> None:
    ttft_p = percentiles([r["ttft_s"] for r in results if r["ttft_s"] is not None])
    tps_p = percentiles([r["tokens_per_s"] for r in results])

    sep = "─" * 46
    share_str = f"{share * 100:.1f}%" if share is not None else "n/a"
    print(f"\n{sep}")
    print(f"  Model : {model}   Label: {args.label}")
    print(f"  Runs  : {args.runs}    GPU share: {share_str}    Output: {out_path}")
    print(sep)
    print(f"  {'Metric':<22} {'p50':>8} {'p95':>8}")
    print(f"  {'TTFT (s)':<22} {ttft_p['p50']:>8.3f} {ttft_p['p95']:>8.3f}")
    print(f"  {'tokens/s':<22} {tps_p['p50']:>8.1f} {tps_p['p95']:>8.1f}")
    print(f"{sep}\n")

    return ttft_p, tps_p


def main() -> None:
    args = build_parser().parse_args()
    models = args.models if args.models else [args.model]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for idx, model in enumerate(models):
            results = _bench_model(model, args, out_path)
            share = gpu_share(model)  # post-run snapshot for display

            ttft_p, tps_p = _print_summary(model, args, out_path, results, share)
            _write_csv_row(csv_path, model, args.label, args.runs,
                           ttft_p, tps_p, share)
            log.info("Summary appended to %s", csv_path)

            if idx < len(models) - 1:
                log.info("[%s] Unloading model before next…", model)
                unload(model)

    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        log.error("Connection failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
