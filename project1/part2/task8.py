"""Task 8 — compare the empirical lifetime CDF to the theoretical one.

The continuous-time lifetime follows a phase-type distribution with CDF
    F_T(t) = 1 - p0 exp(Q_s t) 1.
We simulate 1000 women and test the fit with a Kolmogorov-Smirnov test.

Run with:
    uv run python -m project1.part2.task8
"""
import numpy as np
import scipy.stats as st
from scipy.linalg import expm

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

DEATH = 4
N_WOMEN = 1000
P0 = np.array([1.0, 0.0, 0.0, 0.0])


def rate_matrix() -> np.ndarray:
    """Task 7/8 transition-rate matrix Q. Off-diagonals given; diagonal = -row sum."""
    Q = np.array(
        [
            [0.0, 0.005, 0.0025, 0.0, 0.001],
            [0.0, 0.0, 0.005, 0.004, 0.005],
            [0.0, 0.0, 0.0, 0.003, 0.005],
            [0.0, 0.0, 0.0, 0.0, 0.009],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def simulate_ctmc(Q: np.ndarray, start: int, rng: np.random.Generator) -> float:
    """Event-driven CTMC until death; returns the lifetime."""
    t, s = 0.0, start
    while s != DEATH:
        rate = -Q[s, s]
        t += rng.exponential(1.0 / rate)
        probs = Q[s].copy()
        probs[s] = 0.0
        s = int(rng.choice(len(Q), p=probs / rate))
    return t


def theoretical_cdf(t: np.ndarray, Q_sub: np.ndarray) -> np.ndarray:
    """F_T(t) = 1 - p0 exp(Q_s t) 1, evaluated elementwise over an array of times."""
    ones = np.ones(Q_sub.shape[0])
    return np.array([1.0 - P0 @ expm(Q_sub * ti) @ ones for ti in np.atleast_1d(t)])


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    Q = rate_matrix()
    Q_sub = Q[:4, :4]
    logger.info("Simulating CTMC", n=N_WOMEN, seed=settings.SEED)

    lifetimes = np.array([simulate_ctmc(Q, 0, rng) for _ in range(N_WOMEN)])

    D, p = st.kstest(lifetimes, lambda t: theoretical_cdf(t, Q_sub))
    logger.success(
        "KS test (empirical vs phase-type CDF)",
        D=round(float(D), 4),
        p=round(float(p), 4),
        verdict="consistent" if p > 0.05 else "inconsistent",
    )

    grid = np.linspace(0, lifetimes.max(), 400)
    with figure(figsize=(8, 5), save="project1/plots/task8/cdf.png") as fig:
        ax = fig.add_subplot(111)
        ax.step(
            np.sort(lifetimes),
            np.arange(1, N_WOMEN + 1) / N_WOMEN,
            where="post",
            label="empirical",
        )
        ax.plot(grid, theoretical_cdf(grid, Q_sub), "k--", label="theoretical")
        ax.set(
            title="Lifetime CDF: empirical vs phase-type",
            xlabel="Lifetime (months)",
            ylabel="F(t)",
        )
        ax.legend()
