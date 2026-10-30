"""Streaming benchmark: measures time-to-first-token (TTFT) and token throughput."""

import time
import statistics
import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer()

SAMPLE_PROMPTS = [
    "Write a short poem about the speed of light.",
    "Summarize the history of the internet in three sentences.",
    "Explain async/await in Python.",
    "List five uses of Fourier transforms.",
    "Describe how a neural network learns.",
]


def stream_anthropic(model: str, prompt: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    t_start = time.perf_counter()
    ttft = None
    output_tokens = 0

    with client.messages.stream(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            if ttft is None:
                ttft = time.perf_counter() - t_start
            output_tokens += 1

    elapsed = time.perf_counter() - t_start
    return {
        "ttft_s": ttft,
        "total_s": elapsed,
        "output_tokens": output_tokens,
        "tokens_per_s": output_tokens / elapsed if elapsed > 0 else 0,
    }


def stream_openai(model: str, prompt: str) -> dict:
    import openai
    client = openai.OpenAI()
    t_start = time.perf_counter()
    ttft = None
    output_tokens = 0

    stream = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta and ttft is None:
            ttft = time.perf_counter() - t_start
        output_tokens += len(delta.split())

    elapsed = time.perf_counter() - t_start
    return {
        "ttft_s": ttft,
        "total_s": elapsed,
        "output_tokens": output_tokens,
        "tokens_per_s": output_tokens / elapsed if elapsed > 0 else 0,
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
            r = stream_anthropic(model, prompt)
        elif provider == "openai":
            r = stream_openai(model, prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        results.append(r)

    ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    throughputs = [r["tokens_per_s"] for r in results]

    table = Table(title=f"Streaming — {model}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Runs", str(len(results)))
    table.add_row("Mean TTFT (s)", f"{statistics.mean(ttfts):.3f}")
    table.add_row("Median TTFT (s)", f"{statistics.median(ttfts):.3f}")
    table.add_row("Mean throughput (tok/s)", f"{statistics.mean(throughputs):.1f}")
    console.print(table)


if __name__ == "__main__":
    app()
