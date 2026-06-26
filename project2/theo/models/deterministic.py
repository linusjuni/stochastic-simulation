# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for synthesizing, code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def sir_ode(
    *,
    N: float,
    I0: float,
    beta: float,
    gamma: float,
    t_max: float,
    R0_init: float = 0.0,
    n_points: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    def rhs(t, y):
        S, I, R = y
        infection = beta * S * I / N
        recovery = gamma * I
        return [-infection, infection - recovery, recovery]

    t_eval = np.linspace(0.0, t_max, n_points)
    sol = solve_ivp(
        rhs, (0.0, t_max), [N - I0 - R0_init, I0, R0_init],
        t_eval=t_eval, rtol=1e-9, atol=1e-9,
    )
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def sir_metrics(
    *, N: float, I0: float, beta: float, gamma: float, t_max: float, n_points: int = 4000
) -> dict[str, float]:
    t, S, I, R = sir_ode(N=N, I0=I0, beta=beta, gamma=gamma, t_max=t_max, n_points=n_points)
    k = int(np.argmax(I))
    return {
        "final_size_frac": float(R[-1] / N),
        "peak_frac": float(I[k] / N),
        "peak_time": float(t[k]),
    }


def final_size_fraction(R0: float) -> float:
    if R0 <= 1.0:
        return 0.0
    z = brentq(lambda z: z - np.exp(-R0 * (1.0 - z)), 1e-12, 1.0 - 1e-9)
    return 1.0 - z
