import numpy as np

from utils.logger import get_logger
from utils.plotting import figure, histogram
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
LOCAL_ONLY = 1
BOTH = 3
LOCAL_RECURRENCE_STATES = {LOCAL_ONLY, BOTH}  # either route counts as "local recurrence"
N_WOMEN = 1000


def _sample_next_state(rng: np.random.Generator, probs: np.ndarray) -> int:
    """Draw one state index via the inverse-CDF method, given a row of P."""
    u = rng.random()
    cumulative = np.cumsum(probs)
    return int(np.searchsorted(cumulative, u))


def simulate_woman(rng: np.random.Generator) -> tuple[int, list[int]]:
    """Simulate one woman from state 1 until death, return (lifetime, path)."""
    state = 0
    path = [state]
    t = 0
    while state != DEATH:
        state = _sample_next_state(rng, P[state])
        t += 1
        path.append(state)
    return t, path


def simulate_population(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[list[int]]]:
    """Simulate n independent women, return their lifetimes and full paths."""
    lifetimes = np.empty(n, dtype=int)
    paths = []
    for i in range(n):
        lifetime, path = simulate_woman(rng)
        lifetimes[i] = lifetime
        paths.append(path)
    return lifetimes, paths


def local_recurrence_fractions(paths: list[list[int]]) -> tuple[float, float]:
    """Fraction with local-only recurrence, and fraction with local recurrence in any form.

    "Any form" also counts paths that reach the both-state via distant
    metastasis first (NED -> distant -> both), since that woman also ends
    up with local recurrence even though she never visited the local-only
    state.
    """
    local_only_count = 0
    any_local_count = 0
    for path in paths:
        if LOCAL_ONLY in path:
            local_only_count += 1
        if any(state in LOCAL_RECURRENCE_STATES for state in path):
            any_local_count += 1
    return local_only_count / len(paths), any_local_count / len(paths)


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    logger.info("Simulating population", n=N_WOMEN, seed=settings.SEED)

    lifetimes, paths = simulate_population(N_WOMEN, rng)

    logger.success(
        "Simulation complete",
        mean_lifetime=round(lifetimes.mean(), 1),
        min_lifetime=lifetimes.min(),
        max_lifetime=lifetimes.max(),
    )

    with figure(figsize=(8, 5), save="project1/plots/task1/lifetime_histogram.png") as fig:
        ax = fig.add_subplot(111)
        histogram(
            lifetimes,
            ax=ax,
            bins=30,
            title="Lifetime distribution (1000 women)",
            xlabel="Lifetime (months)",
        )

    local_only_fraction, any_local_fraction = local_recurrence_fractions(paths)
    logger.success(
        "Local recurrence fractions",
        local_only=round(local_only_fraction, 3),
        local_or_local_and_distant=round(any_local_fraction, 3),
    )
