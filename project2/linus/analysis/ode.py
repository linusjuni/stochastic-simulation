from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def solve_sirs(
    *,
    N: int,
    I0: int,
    beta: float,
    gamma: float,
    omega: float,
    t_max: float,
    n_points: int = 1000,
    S0: float | None = None,
    R0_count: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic mean-field SIRS model, solved on a dense uniform grid.

    Solves the ODE system

        dS/dt = -beta * S * I / N + omega * R
        dI/dt =  beta * S * I / N - gamma * I
        dR/dt =  gamma * I        - omega * R

    Setting ``omega = 0`` recovers the classic SIR ODE, so this also serves as
    the deterministic baseline for the population-size comparison (Part I-d).

    Parameters
    ----------
    N : int
        Total population (held constant by the equations).
    I0 : int
        Initial infectives. Defaults of ``S0`` make the start match a stochastic
        run seeded with the same ``S = N - I0``.
    beta, gamma, omega : float
        Infection, recovery, and waning rates.
    t_max : float
        Time horizon.
    n_points : int
        Number of evenly spaced output times in ``[0, t_max]``.
    S0, R0_count : float, optional
        Override the initial susceptible / recovered counts. By default
        ``S0 = N - I0`` and ``R0_count = 0``.

    Returns
    -------
    (t, S, I, R) : tuple of np.ndarray
        Each array has length ``n_points``.
    """
    if S0 is None:
        S0 = N - I0

    def rhs(_t: float, y: np.ndarray) -> list[float]:
        S, I, R = y
        infection = beta * S * I / N
        return [
            -infection + omega * R,
            infection - gamma * I,
            gamma * I - omega * R,
        ]

    t_eval = np.linspace(0.0, t_max, n_points)
    sol = solve_ivp(
        rhs,
        (0.0, t_max),
        [S0, float(I0), R0_count],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )
    S, I, R = sol.y
    return sol.t, S, I, R
