"""Task 7 — continuous-time model: lifetime distribution.

Simulate 1000 women in a CTMC until death. Report the lifetime distribution
(histogram), the mean and std each with a confidence interval, and the
proportion of women whose cancer has reappeared distantly by 30.5 months.

Run with:
    uv run python -m project1.part2.task7
"""
import numpy as np
import scipy.stats as st

from utils.logger import get_logger
from utils.plotting import figure, histogram
from utils.settings import settings

logger = get_logger(__name__)

# States: 0=post-surgery, 1=local recurrence, 2=distant metastasis, 3=both, 4=death.
DEATH = 4
DISTANT_STATES = {2, 3}  # distant metastasis, or both (which includes distant)
N_WOMEN = 1000
T_DISTANT = 30.5
N_BOOT = 10_000


def rate_matrix() -> np.ndarray:
    """Task 7 transition-rate matrix Q. Off-diagonals given; diagonal = -row sum."""
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


def simulate_ctmc(
    Q: np.ndarray, start: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Event-driven CTMC until absorption in DEATH.

    In state i: stay an Exp(rate=-q_ii) sojourn, then jump to j != i with
    probability q_ij / (-q_ii). Returns (states, entry_times) with the woman's
    lifetime = entry_times[-1].
    """
    states, times = [start], [0.0]
    t, s = 0.0, start
    while s != DEATH:
        rate = -Q[s, s]
        t += rng.exponential(1.0 / rate)
        probs = Q[s].copy()
        probs[s] = 0.0
        s = int(rng.choice(len(Q), p=probs / rate))
        states.append(s)
        times.append(t)
    return np.array(states), np.array(times)


def entered_distant_by(states: np.ndarray, times: np.ndarray, t: float) -> bool:
    """True if the woman entered a distant-recurrence state at or before time t."""
    return any(s in DISTANT_STATES and te <= t for s, te in zip(states, times))


def analytic_mean_lifetime(Q: np.ndarray) -> float:
    """E[T] = p0 (-Q_s)^{-1} 1, with Q_s the transient sub-matrix and p0 = e_1."""
    Q_sub = Q[:4, :4]
    p0 = np.array([1.0, 0.0, 0.0, 0.0])
    return float(p0 @ np.linalg.solve(-Q_sub, np.ones(4)))


def t_ci(samples: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-based confidence interval. Returns (mean, low, high)."""
    n = len(samples)
    mean = float(np.mean(samples))
    se = float(np.std(samples, ddof=1)) / np.sqrt(n)
    hw = float(st.t.ppf((1.0 + conf) / 2.0, df=n - 1) * se)
    return mean, mean - hw, mean + hw


def bootstrap_std_ci(
    samples: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT, conf: float = 0.95
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the standard deviation. Returns (std, low, high)."""
    idx = rng.integers(0, len(samples), size=(n_boot, len(samples)))
    boot = np.std(samples[idx], axis=1, ddof=1)
    lo, hi = np.quantile(boot, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(np.std(samples, ddof=1)), float(lo), float(hi)


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    Q = rate_matrix()
    logger.info("Simulating CTMC", n=N_WOMEN, seed=settings.SEED)

    trajectories = [simulate_ctmc(Q, 0, rng) for _ in range(N_WOMEN)]
    lifetimes = np.array([times[-1] for _, times in trajectories])

    mean, m_lo, m_hi = t_ci(lifetimes)
    std, s_lo, s_hi = bootstrap_std_ci(lifetimes, rng)
    logger.success(
        "Mean lifetime (months)",
        mean=round(mean, 2),
        ci95=(round(m_lo, 2), round(m_hi, 2)),
        analytic=round(analytic_mean_lifetime(Q), 2),
    )
    logger.success(
        "Std lifetime (months)", std=round(std, 2), ci95=(round(s_lo, 2), round(s_hi, 2))
    )

    distant = np.mean(
        [entered_distant_by(states, times, T_DISTANT) for states, times in trajectories]
    )
    logger.success(
        "Distant recurrence by 30.5 months", proportion=round(float(distant), 4)
    )

    with figure(figsize=(8, 5), save="project1/plots/task7/lifetimes.png") as fig:
        ax = fig.add_subplot(111)
        histogram(
            lifetimes,
            ax=ax,
            bins=40,
            title="Lifetime distribution after surgery (1000 women)",
            xlabel="Lifetime (months)",
        )
        ax.axvline(mean, color="black", linestyle="--", label=f"mean = {mean:.0f}")
        ax.legend()
