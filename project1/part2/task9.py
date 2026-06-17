"""Task 9 — Kaplan-Meier survival curves: treatment vs no treatment.

Simulate 1000 women under the original rate matrix and 1000 under the
preventive-treatment matrix, then plot both Kaplan-Meier survival estimates
S(t) = (N - d(t)) / N on the same axis.

Run with:
    uv run python -m project1.part2.task9
"""
import numpy as np

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

DEATH = 4
N_WOMEN = 1000


def _fill_diagonal(Q: np.ndarray) -> np.ndarray:
    """Set the diagonal to the negative off-diagonal row-sum (eq. 1)."""
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def rate_matrix() -> np.ndarray:
    """Original transition-rate matrix Q (no treatment)."""
    return _fill_diagonal(
        np.array(
            [
                [0.0, 0.005, 0.0025, 0.0, 0.001],
                [0.0, 0.0, 0.005, 0.004, 0.005],
                [0.0, 0.0, 0.0, 0.003, 0.005],
                [0.0, 0.0, 0.0, 0.0, 0.009],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )


def treatment_rate_matrix() -> np.ndarray:
    """Preventive-treatment rate matrix (the '*' diagonals are filled by eq. 1)."""
    return _fill_diagonal(
        np.array(
            [
                [0.0, 0.0025, 0.00125, 0.0, 0.001],
                [0.0, 0.0, 0.002, 0.005, 0.005],
                [0.0, 0.0, 0.0, 0.003, 0.005],
                [0.0, 0.0, 0.0, 0.0, 0.009],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )


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


def kaplan_meier(lifetimes: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Survival estimate S(t) = (N - d(t)) / N, with d(t) = #deaths by time t."""
    deaths = np.searchsorted(np.sort(lifetimes), grid, side="right")
    return (len(lifetimes) - deaths) / len(lifetimes)


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    logger.info("Simulating both cohorts", n=N_WOMEN, seed=settings.SEED)

    untreated = np.array([simulate_ctmc(rate_matrix(), 0, rng) for _ in range(N_WOMEN)])
    treated = np.array(
        [simulate_ctmc(treatment_rate_matrix(), 0, rng) for _ in range(N_WOMEN)]
    )

    logger.success(
        "Mean lifetime (months)",
        no_treatment=round(float(untreated.mean()), 1),
        treatment=round(float(treated.mean()), 1),
    )

    grid = np.linspace(0, max(untreated.max(), treated.max()), 500)
    with figure(figsize=(8, 5), save="project1/plots/task9/survival.png") as fig:
        ax = fig.add_subplot(111)
        ax.step(grid, kaplan_meier(untreated, grid), where="post", label="no treatment")
        ax.step(grid, kaplan_meier(treated, grid), where="post", label="preventive treatment")
        ax.set(
            title="Kaplan-Meier survival: treatment vs no treatment",
            xlabel="Time after surgery (months)",
            ylabel=r"Survival $\hat{S}(t)$",
        )
        ax.legend()
