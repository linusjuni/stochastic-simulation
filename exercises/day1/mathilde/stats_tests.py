from __future__ import annotations

import math


def bin_histogram(values: list[float], bins: int, low: float = 0.0, high: float = 1.0) -> tuple[list[int], list[float]]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if high <= low:
        raise ValueError("high must be greater than low")

    counts = [0] * bins
    width = high - low
    for value in values:
        index = int((value - low) * bins / width)
        if index < 0:
            index = 0
        elif index >= bins:
            index = bins - 1
        counts[index] += 1

    edges = [low + index * width / bins for index in range(bins + 1)]
    return counts, edges


def chi_square_test(values: list[float], bins: int = 10, low: float = 0.0, high: float = 1.0) -> dict[str, float]:
    """Check whether the binned counts are close to the expected uniform counts."""

    counts, _ = bin_histogram(values, bins, low, high)
    expected = len(values) / bins
    statistic = sum((count - expected) ** 2 / expected for count in counts)
    return {"statistic": statistic, "df": float(bins - 1)}


def kolmogorov_smirnov_test(values: list[float]) -> dict[str, float]:
    """Check the largest distance between the empirical and uniform distribution."""

    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 0:
        return {"statistic": 0.0, "d_plus": 0.0, "d_minus": 0.0}

    d_plus = 0.0
    d_minus = 0.0
    for index, value in enumerate(sorted_values, start=1):
        d_plus = max(d_plus, index / n - value)
        d_minus = max(d_minus, value - (index - 1) / n)
    return {"statistic": max(d_plus, d_minus), "d_plus": d_plus, "d_minus": d_minus}


def runs_test(values: list[float], threshold: float = 0.5) -> dict[str, float]:
    n = len(values)
    if n < 2:
        return {"runs": float(n), "z": 0.0, "expected": float(n), "variance": 0.0}

    signs = [value >= threshold for value in values]
    n1 = sum(signs)
    n0 = n - n1
    runs = 1
    for previous, current in zip(signs, signs[1:]):
        if current != previous:
            runs += 1

    if n0 == 0 or n1 == 0:
        return {"runs": float(runs), "z": float("inf"), "expected": float(runs), "variance": 0.0}

    expected = 1 + 2 * n0 * n1 / (n0 + n1)
    variance = (2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / (((n0 + n1) ** 2) * (n0 + n1 - 1))
    z = abs(runs - expected) / math.sqrt(variance) if variance > 0 else 0.0
    return {"runs": float(runs), "z": z, "expected": expected, "variance": variance}


def correlation_test(values: list[float], lag: int = 1) -> dict[str, float]:
    if lag <= 0:
        raise ValueError("lag must be positive")
    if len(values) <= lag:
        return {"lag": float(lag), "correlation": 0.0}

    x = values[:-lag]
    y = values[lag:]
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y))
    denom_x = sum((x_i - mean_x) ** 2 for x_i in x)
    denom_y = sum((y_i - mean_y) ** 2 for y_i in y)
    denominator = math.sqrt(denom_x * denom_y)
    return {"lag": float(lag), "correlation": numerator / denominator if denominator else 0.0}


def analyze(values: list[float]) -> dict[str, dict[str, float]]:
    return {
        "chi_square": chi_square_test(values),
        "ks": kolmogorov_smirnov_test(values),
        "runs": runs_test(values),
        "correlation_h1": correlation_test(values, lag=1),
        "correlation_h2": correlation_test(values, lag=2),
        "correlation_h5": correlation_test(values, lag=5),
    }


def run_smoke_tests() -> None:
    counts, _ = bin_histogram([0.0, 0.2, 0.4, 0.6, 0.8], 5)
    assert counts == [1, 1, 1, 1, 1]
    assert chi_square_test([0.0, 0.2, 0.4, 0.6, 0.8], 5)["statistic"] == 0.0
