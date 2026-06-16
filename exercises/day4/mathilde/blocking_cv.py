from __future__ import annotations

import numpy as np

from exercises.day3.mathilde.blocking_system import (
    erlang_b,
    exp_interarrival,
    exp_service,
    simulate,
)

M = 10
A = 8.0
ERLANG_B = erlang_b(A, M)


def run_once(n: int, rng: np.random.Generator) -> tuple[float, float]:
    inter = exp_interarrival(n, rng)
    svc = exp_service(n, rng)
    B = simulate(inter, svc, M)
    S = float(inter.sum())
    return B, S


def estimate_c(pilot_Bs: np.ndarray, pilot_Ss: np.ndarray) -> float:
    return np.cov(pilot_Bs, pilot_Ss, ddof=1)[0, 1] / np.var(pilot_Ss, ddof=1)
