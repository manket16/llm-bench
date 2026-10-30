"""Single-call latency benchmark: measures end-to-end request time and token counts."""

import time
import statistics
import typer
from typing import Optional
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer()

SAMPLE_PROMPTS = [
    "Explain the difference between mutex and semaphore in two sentences.",
    "What is the capital of France?",
    "Write a haiku about distributed systems.",
    "What is 17 * 43?",
    "Name three sorting algorithms and their average time complexity.",
]


def call_anthropic(model: str, prompt: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    t0 = time.perf_counter()
    message = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.perf_counter() - t0
    return {
        "latency_s": elapsed,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }


def call_openai(model: str, prompt: str) -> dict:
    import openai
    client = openai.OpenAI()
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.perf_counter() - t0
    usage = response.usage
    return {
        "latency_s": elapsed,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }


@app.command()
def run(
    model: str = typer.Option("claude-haiku-5-5", help="Model identifier"),
    provider: str = typer.Option("anthropic", help="Provider: anthropic | openai"),
    prompts: int = typer.Option(5, help="Number of prompts to run"),
):
    results = []
    for i, prompt in enumerate(SAMPLE_PROMPTS[:prompts]):
        console.print(f"[dim]Run {i+1}/{prompts}…[/dim]")
        if provider == "anthropic":
            r = call_anthropic(model, prompt)
        elif provider == "openai":
            r = call_openai(model, prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        results.append(r)

    latencies = [r["latency_s"] for r in results]
    out_tokens = [r["output_tokens"] for r in results]

    table = Table(title=f"Latency — {model}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Runs", str(len(results)))
    table.add_row("Mean latency (s)", f"{statistics.mean(latencies):.3f}")
    table.add_row("Median latency (s)", f"{statistics.median(latencies):.3f}")
    table.add_row("P95 latency (s)", f"{sorted(latencies)[int(len(latencies)*0.95)]:.3f}")
    table.add_row("Mean output tokens", f"{statistics.mean(out_tokens):.1f}")
    console.print(table)


if __name__ == "__main__":
    app()
