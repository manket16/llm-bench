import pytest
from bench.stats import percentiles


def test_known_median():
    result = percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["p50"] == 3.0


def test_known_p95():
    values = list(range(1, 21))  # 1..20, p95 index = min(19, 19) = 19 → value 20
    result = percentiles([float(v) for v in values])
    assert result["p95"] == 20.0


def test_p95_not_less_than_p50():
    result = percentiles([1.0, 1.5, 2.0, 3.0, 10.0, 50.0])
    assert result["p95"] >= result["p50"]


def test_single_element():
    result = percentiles([7.0])
    assert result["p50"] == 7.0
    assert result["p95"] == 7.0


def test_empty_raises():
    with pytest.raises(ValueError):
        percentiles([])
