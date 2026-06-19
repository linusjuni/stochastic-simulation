import numpy as np
from scipy.stats import chisquare

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

# States: 0=NED, 1=local recurrence, 2=distant metastasis, 3=both, 4=death
P = np.array(
    [
        [0.9915, 0.005, 0.0025, 0.0, 0.001],
        [0.0, 0.986, 0.005, 0.004, 0.005],
        [0.0, 0.0, 0.992, 0.003, 0.005],
        [0.0, 0.0, 0.0, 0.991, 0.009],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)

DEATH = 4
N_WOMEN = 1000
N_BINS = 10  # deciles of the theoretical lifetime distribution

P_S = P[:4, :4]  # transient sub-matrix (states 1-4)
PI = np.array([1.0, 0.0, 0.0, 0.0])  # everyone starts in state 1 (NED)


def _sample_next_state(rng: np.random.Generator, probs: np.ndarray) -> int:
    """Draw one state index via the inverse-CDF method, given a row of P."""
    u = rng.random()
    cumulative = np.cumsum(probs)
    return int(np.searchsorted(cumulative, u))


def simulate_lifetime(rng: np.random.Generator) -> int:
    """Simulate one woman from state 1 until death, return only her lifetime."""
    state = 0
    t = 0
    while state != DEATH:
        state = _sample_next_state(rng, P[state])
        t += 1
    return t


def simulate_population_lifetimes(n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate n independent women, return their lifetimes."""
    return np.array([simulate_lifetime(rng) for _ in range(n)])


def survival_function(t: int) -> float:
    """P(T > t): probability of still being in a transient state after t months."""
    return float(PI @ np.linalg.matrix_power(P_S, t) @ np.ones(4))


def lifetime_cdf(t: int) -> float:
    """P(T <= t), the discrete phase-type distribution function for the lifetime."""
    return 1.0 - survival_function(t)


def theoretical_mean_lifetime() -> float:
    """E(T) = pi (I - P_s)^-1 1, the expected lifetime under the model."""
    expected_steps = np.linalg.solve(np.eye(4) - P_S, np.ones(4))
    return float(PI @ expected_steps)


def decile_bin_edges() -> list[int]:
    """Lifetime thresholds splitting the theoretical CDF into 10 equal-probability bins."""
    edges = []
    for k in range(1, N_BINS):
        target = k / N_BINS
        t = 0
        while lifetime_cdf(t) < target:
            t += 1
        edges.append(t)
    return edges


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    logger.info("Simulating population", n=N_WOMEN, seed=settings.SEED)

    lifetimes = simulate_population_lifetimes(N_WOMEN, rng)

    logger.success(
        "Mean lifetime",
        simulated=round(lifetimes.mean(), 1),
        theoretical=round(theoretical_mean_lifetime(), 1),
    )

    edges = decile_bin_edges()
    # +0.5 boundaries so the integer lifetime at each edge lands in the
    # bin whose theoretical mass it was selected to complete.
    hist_edges = [-np.inf] + [e + 0.5 for e in edges] + [np.inf]
    observed, _ = np.histogram(lifetimes, bins=hist_edges)

    cdf_at_edges = [0.0] + [lifetime_cdf(e) for e in edges] + [1.0]
    expected = N_WOMEN * np.diff(cdf_at_edges)

    logger.success(
        "Lifetime distribution, by decile",
        observed=observed.tolist(),
        expected=np.round(expected, 1).tolist(),
    )

    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    logger.success(
        "Chi-square goodness-of-fit test",
        chi2=round(chi2_stat, 3),
        df=N_BINS - 1,
        p_value=round(p_value, 4),
    )

    with figure(
        figsize=(8, 5), save="project1/plots/task3/observed_vs_expected.png"
    ) as fig:
        ax = fig.add_subplot(111)
        x = np.arange(N_BINS)
        width = 0.35
        ax.bar(x - width / 2, observed, width, label="Observed")
        ax.bar(x + width / 2, expected, width, label="Expected")
        ax.set_xticks(x)
        ax.set_xticklabels([f"D{i + 1}" for i in range(N_BINS)])
        ax.set_xlabel("Decile of the theoretical lifetime distribution")
        ax.set_ylabel("Number of women")
        ax.set_title("Observed vs. expected lifetimes, by theoretical decile")
        ax.legend()
