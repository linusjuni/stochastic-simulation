import numpy as np
import matplotlib.pyplot as plt

from utils.logger import get_logger
from utils.plotting import figure, histogram
from utils.settings import settings

logger = get_logger(__name__)

P = np.array(
    [
        [0.9915, 0.005, 0.0025, 0.0, 0.001],
        [0.0, 0.986, 0.005, 0.004, 0.005],
        [0.0, 0.0, 0.992, 0.003, 0.005],
        [0.0, 0.0, 0.0, 0.991, 0.009],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)

STATES = {
    0: "cancer removal",
    1: "local recurrence",
    2: "distant metastasis",
    3: "both local and distant",
    4: "death",
}

def simulate(rng: np.random.Generator, n_women: int, P: np.ndarray):
    """Simulate the lifetime of a women aftr removal of breast cancer :)
        return the lifetimes and the paths of how they got there."""
    lifetimes = np.zeros(n_women, dtype=int)
    paths = []
    
    for women in range(n_women):
        state = 0
        path = [state]
        t = 0
        while state != 4:
            new_state = rng.choice(len(P[state]), p=P[state])
            t += 1
            path.append(new_state)
            state = new_state
        lifetimes[women] = t
        paths.append(path)
    return lifetimes, paths

def local_recurrence_fractions(paths: list[list[int]]) -> float:
    """Calculate the fraction of women that had a local recurrence so either passing state 1 or state 3."""
    local_recurrences = 0
    for path in paths:
        if 1 in path or 3 in path:
            local_recurrences += 1
    return local_recurrences / len(paths)

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    NO_OF_WOMEN = 1000
    lifetimes, paths = simulate(rng, NO_OF_WOMEN, P)
    local_recurrence_fraction = local_recurrence_fractions(paths)
    min_lifetime = np.min(lifetimes)
    mean_lifetime = np.mean(lifetimes)
    max_lifetime = np.max(lifetimes)
    logger.info(f"Fraction of women with local recurrence: {local_recurrence_fraction:.4f}")
    logger.info(f"Minimum lifetime: {min_lifetime}")
    logger.info(f"Maximum lifetime: {max_lifetime}")
    logger.info(f"Mean lifetime: {mean_lifetime:.2f}")
    
    histogram(lifetimes, bins="auto", title="Lifetime distribution of simulated women")
    plt.savefig("project1/mathias/part1/figs/lifetime_histogram.png")
    


        
    