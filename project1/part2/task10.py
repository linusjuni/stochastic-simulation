"""Task 10 (optional) — log-rank test for a treatment effect on survival.

Simulate 1000 untreated and 1000 treated women, then test whether their
survival functions differ using the log-rank test. With no censoring, the
statistic is (O_A - E_A)^2 / V ~ chi^2(1).

Run with:
    uv run python -m project1.part2.task10
"""
import numpy as np
import scipy.stats as st

from utils.logger import get_logger
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


def logrank_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Log-rank test of two uncensored survival samples. Returns (chi2, p)."""
    obs_a = exp_a = var = 0.0
    for t in np.unique(np.concatenate([a, b])):
        n_a, n_b = int(np.sum(a >= t)), int(np.sum(b >= t))
        n = n_a + n_b
        d_a = int(np.sum(a == t))
        d = d_a + int(np.sum(b == t))
        if n < 2 or d == 0:
            continue
        obs_a += d_a
        exp_a += d * n_a / n
        var += d * (n_a / n) * (n_b / n) * (n - d) / (n - 1)
    chi2 = (obs_a - exp_a) ** 2 / var
    return float(chi2), float(st.chi2.sf(chi2, df=1))


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    logger.info("Simulating both cohorts", n=N_WOMEN, seed=settings.SEED)

    untreated = np.array([simulate_ctmc(rate_matrix(), 0, rng) for _ in range(N_WOMEN)])
    treated = np.array(
        [simulate_ctmc(treatment_rate_matrix(), 0, rng) for _ in range(N_WOMEN)]
    )

    chi2, p = logrank_test(untreated, treated)
    logger.success(
        "Log-rank test",
        chi2=round(chi2, 2),
        p=f"{p:.2e}",
        verdict="significant" if p < 0.05 else "not significant",
    )
