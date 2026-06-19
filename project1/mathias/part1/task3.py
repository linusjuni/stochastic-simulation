import numpy as np
from scipy.stats import chisquare, kstest
import time as t
import matplotlib.pyplot as plt
import seaborn as sns

from task1 import simulate, local_recurrence_fractions
from utils.logger import get_logger
from utils.plotting import figure, histogram

logger = get_logger(__name__)

RANDOM_NUMBER_GENERATOR = np.random.default_rng(42)
NO_OF_WOMEN = 1000


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

def discrete_phase_type_distribution(probability_matrix: np.ndarray, initial_distribution: np.ndarray, time_of_interest: int) -> np.ndarray:
    """ First we remove the last row and column of the probability matrix,
    then we define a vector with the probabilities in state i reaching the 'last' state"""
    
    probability_matrix_sub = probability_matrix[:-1, :-1]
    probability_to_last_state = probability_matrix[:-1, -1]
    pi = initial_distribution
    
    first_term = pi @ np.linalg.matrix_power(probability_matrix_sub, time_of_interest)
    return first_term @ probability_to_last_state

def expectation_of_discrete_phase_type_distribution(probability_matrix: np.ndarray, initial_distribution: np.ndarray) -> float:
    probability_matrix_sub = probability_matrix[:-1, :-1]
    I_ = np.eye(probability_matrix_sub.shape[0])
    return float (initial_distribution @ np.linalg.inv(I_ - probability_matrix_sub) @ np.ones(probability_matrix_sub.shape[0]))

def create_cdf_interpolator(cdf_values: np.ndarray, time_range: np.ndarray):
    """Create an interpolator function for the CDF values."""
    return lambda x: np.interp(x, xp=time_range, fp=cdf_values, left=0, right=1)


if __name__ == "__main__":
    lifetimes, _ = simulate(RANDOM_NUMBER_GENERATOR, NO_OF_WOMEN, P)
    initial_distribution = np.array([1,0,0,0])
    time_range = np.arange(0, max(lifetimes) + 1)
    pdf = [discrete_phase_type_distribution(P, initial_distribution, t) for t in time_range]
    expected_lifetime = expectation_of_discrete_phase_type_distribution(P, initial_distribution)
    logger.info("Expected lifetime", expected_lifetime=expected_lifetime)

    with figure(figsize=(10, 6), save="project1/mathias/part1/figs/lifetime_vs_phase_type.png") as fig:
        ax = fig.add_subplot(111)
        histogram(
            lifetimes,
            bins="auto",
            stat="density",
            title="Simulated lifetimes vs discrete phase-type PMF",
            xlabel="Lifetime (months)",
            ylabel="Density",
            label="Simulated",
            ax=ax,
        )
        ax.plot(time_range, pdf, color="black", linewidth=2.5, label="Theoretical PMF")
        ax.axvline(expected_lifetime, color="red", linestyle="dashdot",linewidth=2.5,  label=f"Expected lifetime: {expected_lifetime:.2f} months")
        ax.legend()

    # chi square test for goodness of fit
    cdf_theoretical = np.cumsum(np.array(pdf))
    cdf_function = create_cdf_interpolator(cdf_theoretical, time_range)
    ks_statistic, p_value = kstest(lifetimes, cdf_function)
    logger.info("KS test for goodness of fit", ks_statistic=ks_statistic, p_value=p_value)
    
    