from __future__ import annotations

import numpy as np


def bootstrap_replicates(data: np.ndarray, statistic, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(data)
    samples = rng.choice(data, size=(k, n), replace=True)
    return np.apply_along_axis(statistic, 1, samples)


def bootstrap_variance(data: np.ndarray, statistic, k: int, rng: np.random.Generator) -> tuple[float, float]:
    replicates = bootstrap_replicates(data, statistic, k, rng)
    return statistic(data), replicates.var(ddof=1)


def pareto_sample(beta: float, k: float, size: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(size=size)
    return beta * u ** (-1.0 / k)


def sample_variance(x: np.ndarray) -> float:
    return x.var(ddof=1)
