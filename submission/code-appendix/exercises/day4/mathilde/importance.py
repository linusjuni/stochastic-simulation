from __future__ import annotations

import numpy as np
import scipy.optimize


def crude_tail(n: int, a: float, rng: np.random.Generator) -> np.ndarray:
    Z = rng.standard_normal(n)
    return (Z > a).astype(float)


def is_tail(n: int, a: float, sigma2: float, rng: np.random.Generator) -> np.ndarray:
    X = rng.normal(loc=a, scale=np.sqrt(sigma2), size=n)
    # importance weight f(X)/g(X) where f=N(0,1), g=N(a,sigma2)
    w = np.exp(-X**2 / 2.0 + (X - a)**2 / (2.0 * sigma2)) * np.sqrt(sigma2)
    return (X > a).astype(float) * w


def var_W(lam: float) -> float:
    return (np.e ** (2.0 + lam) - 1.0) / (lam * (2.0 + lam)) - (np.e - 1.0) ** 2


def optimal_lambda() -> tuple[float, float]:
    res = scipy.optimize.minimize_scalar(var_W, bounds=(0.01, 20.0), method="bounded")
    return float(res.x), float(res.fun)


def is_exponential(n: int, lam: float, rng: np.random.Generator) -> np.ndarray:
    X = rng.exponential(1.0 / lam, size=n)
    # W = (1/lambda) * exp((1+lambda)*X) * 1{X <= 1}
    return (X <= 1.0).astype(float) * np.exp((1.0 + lam) * X) / lam


def is_pareto_mean(n: int, k: float, rng: np.random.Generator) -> np.ndarray:
    # sample from size-biased density g1 ~ Pareto(k-1, beta) via inversion
    beta = 8.0 * (k - 1.0) / k
    u = rng.uniform(size=n)
    x = beta / u ** (1.0 / (k - 1.0))
    f_x = k * beta**k * x ** (-(k + 1.0))
    g1_x = (k - 1.0) * beta ** (k - 1.0) * x ** (-k)
    return x * f_x / g1_x  # = k*beta/(k-1) exactly: constant IS weights
