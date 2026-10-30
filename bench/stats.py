"""Pure statistical helpers."""

import statistics


def percentiles(values: list[float]) -> dict[str, float]:
    """Return p50 and p95 for a list of floats.

    Raises ValueError on empty input.
    """
    if not values:
        raise ValueError("Cannot compute percentiles of an empty list")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    p95_idx = min(int(n * 0.95), n - 1)
    return {
        "p50": statistics.median(sorted_vals),
        "p95": sorted_vals[p95_idx],
    }
