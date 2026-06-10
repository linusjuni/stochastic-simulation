from __future__ import annotations

import numpy as np


def geometric_sample(p: float, size: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(size=size)
    return np.ceil(np.log(u) / np.log(1 - p)).astype(int)


def geometric_pmf(p: float, k: int) -> float:
    return (1 - p) ** (k - 1) * p


def crude_sample(probs: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    cdf = np.cumsum(probs)
    u = rng.uniform(size=size)
    return np.searchsorted(cdf, u, side="left") + 1  # 1-indexed


def rejection_sample(probs: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    k = len(probs)
    c = probs.max()
    results = np.empty(size, dtype=int)
    filled = 0
    while filled < size:
        batch = min((size - filled) * 4, 200_000)
        idx = rng.integers(0, k, size=batch)
        u = rng.uniform(size=batch)
        accepted = idx[u <= probs[idx] / c]
        take = min(len(accepted), size - filled)
        results[filled : filled + take] = accepted[:take] + 1  # 1-indexed
        filled += take
    return results


def alias_build(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(probs)
    prob = np.array(probs, dtype=float) * n
    alias = np.zeros(n, dtype=int)
    small = [i for i in range(n) if prob[i] < 1.0]
    large = [i for i in range(n) if prob[i] >= 1.0]
    while small and large:
        l = small.pop()
        g = large[-1]
        alias[l] = g
        prob[g] += prob[l] - 1.0
        if prob[g] < 1.0:
            large.pop()
            small.append(g)
    for i in small + large:
        prob[i] = 1.0
    return prob, alias


def alias_sample(
    prob: np.ndarray, alias: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    n = len(prob)
    i = rng.integers(0, n, size=size)
    u = rng.uniform(size=size)
    return np.where(u <= prob[i], i, alias[i]) + 1  # 1-indexed
