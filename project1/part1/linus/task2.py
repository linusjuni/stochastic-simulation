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
T = 120
STATE_LABELS = ["NED", "Local", "Distant", "Both", "Death"]


def _sample_next_state(rng: np.random.Generator, probs: np.ndarray) -> int:
    """Draw one state index via the inverse-CDF method, given a row of P."""
    u = rng.random()
    cumulative = np.cumsum(probs)
    return int(np.searchsorted(cumulative, u))


def simulate_woman_path(rng: np.random.Generator) -> list[int]:
    """Simulate one woman from state 1 until death, return her full path."""
    state = 0
    path = [state]
    while state != DEATH:
        state = _sample_next_state(rng, P[state])
        path.append(state)
    return path


def simulate_population_paths(n: int, rng: np.random.Generator) -> list[list[int]]:
    """Simulate n independent women, return their full paths."""
    return [simulate_woman_path(rng) for _ in range(n)]


def state_at_t(path: list[int], t: int) -> int:
    """State of a woman at month t. Once she has died, she stays dead."""
    return path[t] if t < len(path) else DEATH


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    logger.info("Simulating population", n=N_WOMEN, seed=settings.SEED)

    paths = simulate_population_paths(N_WOMEN, rng)

    observed = np.bincount([state_at_t(path, T) for path in paths], minlength=5)

    p0 = np.zeros(5)
    p0[0] = 1.0
    p_t = p0 @ np.linalg.matrix_power(P, T)
    expected = p_t * N_WOMEN

    logger.success(
        "State distribution at t=120",
        observed=observed.tolist(),
        expected=np.round(expected, 1).tolist(),
    )

    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    logger.success(
        "Chi-square goodness-of-fit test",
        chi2=round(chi2_stat, 3),
        df=4,
        p_value=round(p_value, 4),
    )

    with figure(
        figsize=(8, 5), save="project1/plots/task2/observed_vs_expected.png"
    ) as fig:
        ax = fig.add_subplot(111)
        x = np.arange(5)
        width = 0.35
        ax.bar(x - width / 2, observed, width, label="Observed")
        ax.bar(x + width / 2, expected, width, label="Expected")
        ax.set_xticks(x)
        ax.set_xticklabels(STATE_LABELS)
        ax.set_ylabel("Number of women")
        ax.set_title("Observed vs. expected state distribution at $t=120$")
        ax.legend()
