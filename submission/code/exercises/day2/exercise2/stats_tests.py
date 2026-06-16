from __future__ import annotations

import numpy as np
from scipy.stats import chisquare


def chi_square_gof(
    observed: np.ndarray, expected_probs: np.ndarray, n: int
) -> dict[str, float]:
    expected = np.asarray(expected_probs, dtype=float) * n
    stat, pvalue = chisquare(observed, expected)
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "df": float(len(observed) - 1),
    }


def geometric_chi_square(samples: np.ndarray, p: float) -> dict[str, float]:
    from exercises.day2.exercise2.discrete import geometric_pmf

    n = len(samples)
    # group tail: all k > max_k into one bin
    max_k = max(int(np.percentile(samples, 99)), 10)

    observed = np.zeros(max_k + 1, dtype=float)
    for x in samples:
        idx = min(int(x) - 1, max_k)
        observed[idx] += 1

    exp_probs = np.array([geometric_pmf(p, k) for k in range(1, max_k + 1)])
    tail_prob = max(0.0, 1.0 - exp_probs.sum())
    exp_probs = np.append(exp_probs, tail_prob)

    stat, pvalue = chisquare(observed, exp_probs * n)
    return {"statistic": float(stat), "pvalue": float(pvalue)}
