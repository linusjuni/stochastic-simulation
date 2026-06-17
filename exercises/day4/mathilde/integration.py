from __future__ import annotations

import numpy as np


def crude_mc(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.exp(rng.uniform(size=n))


def antithetic(n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    U = rng.uniform(size=n_pairs)
    return (np.exp(U) + np.exp(1.0 - U)) / 2.0


def control_variate(n: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    U = rng.uniform(size=n)
    X = np.exp(U)
    Z = U  # E[Z] = 0.5
    c = -np.cov(X, Z, ddof=1)[0, 1] / np.var(Z, ddof=1)
    return X + c * (Z - 0.5), c


def stratified(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    # n/k composite estimators; each pools k stratified draws
    n_reps = n // k
    j = np.arange(k)
    Y = np.empty(n_reps)
    for i in range(n_reps):
        V = rng.uniform(size=k)
        U = (j + V) / k  # one draw per stratum [j/k, (j+1)/k)
        Y[i] = np.exp(U).mean()
    return Y
