import numpy as np
from scipy.stats import chisquare, kstest
import time as t
import matplotlib.pyplot as plt
import seaborn as sns

from project1.part1.mathias.part1.task1 import simulate, local_recurrence_fractions
from utils.logger import get_logger
from utils.plotting import figure, histogram

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

TIME_OF_INTEREST = 120

def construct_probabiltity_at_time_of_interest(paths: list[list[int]], time_of_interest: int) -> np.ndarray:
    """ Since it is a markov process, we just count the transitions from t-1 to t, and marginalize along the rows to get the probabilities"""
    
    # for all paths finished before the time of interest, we just count the last state, and for all paths that are longer than the time of interest, we count the state at the time of interest.
    count_matrix = np.zeros((5,5), dtype=int)
    for path in paths:
        if len(path) <= time_of_interest:
            count_matrix[-1, path[-1]] += 1
        else:
            count_matrix[path[time_of_interest-1], path[time_of_interest]] += 1
    # marginalize along the rows to get the probabilities
    prob_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True)
    return prob_matrix

def expected_markov_chain_marginal_probabilties(initial_distribution: np.ndarray, probability_matrix: np.ndarray, time_of_interest: int) -> np.ndarray:
    """ Since it is a markov process, we can compute the expected probabilities at the time of interest by multiplying the initial distribution with the probability matrix raised to the power of the time of interest"""
    return initial_distribution @ np.linalg.matrix_power(probability_matrix, time_of_interest)

def observed_marginal_probabilities(paths: list[list[int]], time_of_interest: int) -> np.ndarray:
    marginal_probabilities = np.zeros(5)
    dead_index = 4
    for path in paths:
        if len(path) <= time_of_interest:
            marginal_probabilities[dead_index] += 1
        else:
            marginal_probabilities[path[time_of_interest]] += 1
    marginal_probabilities /= marginal_probabilities.sum()
    return marginal_probabilities

if __name__ == "__main__":
    start = t.perf_counter()
    rng = np.random.default_rng(42)
    NO_OF_WOMEN = 1000
    _, paths = simulate(rng, NO_OF_WOMEN, P)
    initial_distribution = np.zeros(5)
    initial_distribution[0] = 1.0
    expected_marg_probabilities = expected_markov_chain_marginal_probabilties(initial_distribution, P, TIME_OF_INTEREST)
    observed_marg_probabilities = observed_marginal_probabilities(paths, TIME_OF_INTEREST) 
    logger.info("Expected marginal probabilities at time of interest", expected_marg_probabilities=expected_marg_probabilities)
    logger.info("Observed marginal probabilities at time of interest", observed_marg_probabilities=observed_marg_probabilities)
    
    states = list(STATES.values())
    x = np.arange(len(states))
    width = 0.4

    plt.bar(x - width/2, expected_marg_probabilities, width, label="Expected", color="blue", alpha=0.7)
    plt.bar(x + width/2, observed_marg_probabilities, width, label="Observed", color="orange", alpha=0.7)
    plt.xticks(x, states, rotation=30, ha="right")
    plt.legend()
    plt.title("Expected vs Observed Marginal Probabilities at t=120")
    plt.xlabel("State")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig("project1/mathias/part1/figs/expected_vs_observed_marginal_probabilities.png", dpi=300, bbox_inches="tight")

    plt.close()

    N_REPLICATIONS = 1000
    master_rng = np.random.default_rng(42)
    seeds = master_rng.integers(0, 2**31 - 1, size=N_REPLICATIONS)
    expected_counts = expected_marg_probabilities * NO_OF_WOMEN

    chi2_stats = np.empty(N_REPLICATIONS)
    p_values = np.empty(N_REPLICATIONS)
    for i, seed in enumerate(seeds):
        t_ = t.perf_counter() - start
        logger.info(f"Replication {i+1}/{N_REPLICATIONS}", elapsed_time=t_)
        rng = np.random.default_rng(int(seed))
        _, paths = simulate(rng, NO_OF_WOMEN, P)
        observed_counts = observed_marginal_probabilities(paths, TIME_OF_INTEREST) * NO_OF_WOMEN
        chi2_stats[i], p_values[i] = chisquare(f_obs=observed_counts, f_exp=expected_counts)

    ks_stat, ks_p = kstest(p_values, "uniform")
    rejection_rate = float(np.mean(p_values < 0.05))
    logger.info(
        "P-value uniformity across replications",
        n=N_REPLICATIONS,
        mean_p=float(np.mean(p_values)),
        rejection_rate_at_0_05=rejection_rate,
        ks_stat=ks_stat,
        ks_p=ks_p,
    )

    fig, ax = plt.subplots()
    ax.hist(p_values, bins=20, range=(0, 1), edgecolor="black", alpha=0.8)
    ax.axhline(N_REPLICATIONS / 20, color="red", linestyle="--", label="Uniform expectation")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Count")
    ax.set_title(f"Chi-square p-values across {N_REPLICATIONS} replications (KS p={ks_p:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig("project1/mathias/part1/figs/pvalue_uniformity.png", dpi=300, bbox_inches="tight")

    