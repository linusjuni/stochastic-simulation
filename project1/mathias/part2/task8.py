import numpy as np
from scipy.stats import kstest
import scipy.linalg as la

from utils.logger import get_logger
from utils.plotting import figure

from project1.mathias.part2.task7 import simulate_continuous_time_markov_chain

logger = get_logger(__name__)

RANDOM_NUMBER_GENERATOR = np.random.default_rng(42)
NO_OF_WOMEN = 1000
NUMBER_OF_CDF_GRID_POINTS = 200

Q = np.array([
    [-0.0085, 0.005, 0.0025, 0.0, 0.001],
    [0.0, -0.014, 0.005, 0.004, 0.005],
    [0.0, 0.0, -0.008, 0.003, 0.005],
    [0.0, 0.0, 0.0, -0.009, 0.009],
    [0.0, 0.0, 0.0, 0.0, 0.0],
])

STATES = {
    0: "cancer removal",
    1: "local recurrence",
    2: "distant metastasis",
    3: "both local and distant",
    4: "death",
}


def phase_type_cdf(transient_generator, initial_transient_distribution, evaluation_time):
    ones_column = np.ones(transient_generator.shape[0])
    survival_probability = (
        initial_transient_distribution
        @ la.expm(transient_generator * evaluation_time)
        @ ones_column
    )
    return 1 - survival_probability


def make_phase_type_cdf_callable(transition_rate_matrix, initial_distribution_over_all_states):
    transient_generator = transition_rate_matrix[:-1, :-1]
    initial_transient_distribution = initial_distribution_over_all_states[:-1]

    def cdf(evaluation_time):
        evaluation_time_array = np.atleast_1d(evaluation_time)
        cdf_values = np.array([
            phase_type_cdf(transient_generator, initial_transient_distribution, t)
            for t in evaluation_time_array
        ])
        return cdf_values if np.ndim(evaluation_time) else cdf_values[0]

    return cdf


def empirical_cdf(sample, evaluation_times):
    return np.array([np.mean(sample <= t) for t in evaluation_times])


if __name__ == "__main__":
    initial_state = 0
    initial_distribution_over_all_states = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    lifetimes, _ = simulate_continuous_time_markov_chain(
        RANDOM_NUMBER_GENERATOR, initial_state, NO_OF_WOMEN, Q
    )

    theoretical_cdf = make_phase_type_cdf_callable(Q, initial_distribution_over_all_states)

    evaluation_times = np.linspace(0, lifetimes.max(), NUMBER_OF_CDF_GRID_POINTS)
    observed_cdf_values = empirical_cdf(lifetimes, evaluation_times)
    theoretical_cdf_values = np.array([theoretical_cdf(t) for t in evaluation_times])

    with figure(figsize=(10, 6), save="project1/mathias/part2/figs/ctmc_cdf_comparison.png") as fig:
        ax = fig.add_subplot(111)
        ax.plot(evaluation_times, observed_cdf_values, label="Observed CDF", linewidth=2.5)
        ax.plot(
            evaluation_times,
            theoretical_cdf_values,
            label="Theoretical phase-type CDF",
            linestyle="dashed",
            linewidth=2.5,
        )
        ax.set_xlabel("Time (months)")
        ax.set_ylabel("CDF")
        ax.set_title("Empirical vs theoretical lifetime CDF (CTMC)")
        ax.legend()

    ks_statistic, ks_p_value = kstest(lifetimes, theoretical_cdf)
    logger.info(
        "Kolmogorov-Smirnov test against phase-type CDF",
        ks_statistic=ks_statistic,
        p_value=ks_p_value,
    )
