import numpy as np

from utils.logger import get_logger
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

NED = 0
DEATH = 4
CHECK_MONTH = 12  # the time point at which we check the recurrence/survival condition
N_ACCEPTED = 1000  # number of women required to pass the rejection filter


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


def meets_condition(path: list[int]) -> bool:
    """True if she is alive and has recurred (locally, distantly, or both) at CHECK_MONTH.

    Since the chain can never move back to NED once it has left it, the
    state at exactly CHECK_MONTH fully determines whether recurrence has
    happened by then.
    """
    if len(path) <= CHECK_MONTH:
        return False  # she died before reaching CHECK_MONTH
    return path[CHECK_MONTH] not in (NED, DEATH)


def simulate_conditional_lifetimes(
    n_accepted: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Rejection-sample lifetimes of women alive and recurred at CHECK_MONTH.

    Keeps simulating full lifetimes and discarding any that don't satisfy
    meets_condition, until n_accepted women are kept. Returns the
    accepted lifetimes and the total number of women simulated (accepted
    plus rejected), so the acceptance rate can be reported.
    """
    accepted_lifetimes = []
    n_simulated = 0
    while len(accepted_lifetimes) < n_accepted:
        lifetime, path = simulate_woman(rng)
        n_simulated += 1
        if meets_condition(path):
            accepted_lifetimes.append(lifetime)
    return np.array(accepted_lifetimes), n_simulated


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    logger.info(
        "Running rejection sampling for Task 4",
        n_accepted=N_ACCEPTED,
        check_month=CHECK_MONTH,
    )

    lifetimes, n_simulated = simulate_conditional_lifetimes(N_ACCEPTED, rng)

    acceptance_rate = N_ACCEPTED / n_simulated
    mean_lifetime = lifetimes.mean()
    std_error = lifetimes.std(ddof=1) / np.sqrt(N_ACCEPTED)
    ci_low, ci_high = (
        mean_lifetime - 1.96 * std_error,
        mean_lifetime + 1.96 * std_error,
    )

    logger.success(
        "Rejection sampling complete",
        n_simulated=n_simulated,
        acceptance_rate=round(acceptance_rate, 4),
    )
    logger.success(
        "Conditional expected lifetime",
        mean=round(mean_lifetime, 1),
        std_error=round(std_error, 2),
        ci_95=(round(float(ci_low), 1), round(float(ci_high), 1)),
    )
