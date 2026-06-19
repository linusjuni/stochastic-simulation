"""Task 12 — build the coarsely-observed time series.

In practice the state of each woman is only seen at screenings every 4th year
(48 months). We simulate 1000 women in the continuous-time model (same Q as in
Part 2) until death and, for each, record the observed time series
    Y = (X(0), X(48), X(96), ..., 5),
which always ends in death (state 5). For the rest of Part 3 these 1000 series
are all we assume to have observed.

Run with:
    uv run python -m project1.part3.task12
"""
import numpy as np
from matplotlib.ticker import MultipleLocator

from utils.logger import get_logger
from utils.plotting import figure, histogram
from utils.settings import settings

logger = get_logger(__name__)

# States: 0=post-surgery, 1=local, 2=distant, 3=both, 4=death (absorbing).
DEATH = 4
N_WOMEN = 1000
DT = 48  # months between screenings


def rate_matrix() -> np.ndarray:
    """Part 2 transition-rate matrix Q. Off-diagonals given; diagonal = -row sum."""
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

    Returns (states, entry_times): the jump chain and the time each state was
    entered, so states[k] holds from entry_times[k] until entry_times[k+1].
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


def observe(states: np.ndarray, entry_times: np.ndarray, dt: int = DT) -> np.ndarray:
    """Sample the state on the screening grid t = 0, dt, 2*dt, ...

    The state in effect at time t is the last one entered at or before t. The
    series stops at the first grid point where death is observed, so its last
    value is always DEATH.
    """
    observations = []
    t = 0
    while True:
        idx = np.searchsorted(entry_times, t, side="right") - 1
        s = int(states[idx])
        observations.append(s)
        if s == DEATH:
            break
        t += dt
    return np.array(observations)


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    Q = rate_matrix()
    logger.info("Simulating CTMC and observing every 48 months", n=N_WOMEN, seed=settings.SEED)

    observed = [observe(*simulate_ctmc(Q, 0, rng)) for _ in range(N_WOMEN)]
    lengths = np.array([len(y) for y in observed])

    assert all(y[-1] == DEATH for y in observed), "every series must end in death"
    logger.success(
        "Observed-series length (#screenings)",
        min=int(lengths.min()),
        mean=round(float(lengths.mean()), 2),
        max=int(lengths.max()),
    )
    for y in observed[:3]:
        logger.info("Example series (states 1-5)", Y=list((y + 1).tolist()))

    with figure(figsize=(8, 5), save="project1/plots/task12/series_lengths.png") as fig:
        ax = fig.add_subplot(111)
        histogram(
            lengths,
            ax=ax,
            discrete=True,
            title="Number of screenings observed per woman (1000 women)",
            xlabel="Series length (number of 48-month screenings)",
        )
        ax.xaxis.set_major_locator(MultipleLocator(2))
