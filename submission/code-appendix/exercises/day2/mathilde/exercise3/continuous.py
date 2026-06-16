from __future__ import annotations

import numpy as np


def exponential_sample(lam: float, size: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(size=size)
    return -np.log(u) / lam


def normal_box_muller(size: int, rng: np.random.Generator) -> np.ndarray:
    # generate in pairs; produce exactly `size` samples
    n_pairs = (size + 1) // 2
    u1 = rng.uniform(size=n_pairs)
    u2 = rng.uniform(size=n_pairs)
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    z = np.concatenate([r * np.cos(theta), r * np.sin(theta)])
    return z[:size]


def pareto_sample(k: float, beta: float, size: int, rng: np.random.Generator) -> np.ndarray:
    # inversion: X = beta / U^(1/k), support [beta, inf)
    u = rng.uniform(size=size)
    return beta / u ** (1.0 / k)


def pareto_composition(k: float, beta: float, size: int, rng: np.random.Generator) -> np.ndarray:
    # Pareto as Gamma-Exponential mixture (Lomax representation):
    # draw rate Λ ~ Gamma(shape=k, scale=1/beta), then X = beta + Exp(rate=Λ)
    lam = rng.gamma(shape=k, scale=1.0 / beta, size=size)
    return beta + rng.exponential(scale=1.0 / lam)
