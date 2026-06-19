import numpy as np
from scipy.stats import chisquare, kstest
import time as t
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, chi2

from utils.logger import get_logger
from utils.plotting import figure, histogram



logger = get_logger(__name__)

RANDOM_NUMBER_GENERATOR = np.random.default_rng(42)
NO_OF_WOMEN = 1000


Q = np.array([
    [-0.0085, 0.005, 0.0025, 0.0, 0.001],
    [0.0, -0.014, 0.005, 0.004, 0.005],
    [0.0, 0.0, -0.008, 0.003, 0.005],
    [0.0, 0.0, 0.0, -0.009, 0.009],
    [0.0, 0.0, 0.0, 0.0, 0.0],
]
)

STATES = {
    0: "cancer removal",
    1: "local recurrence",
    2: "distant metastasis",
    3: "both local and distant",
    4: "death",
}

ABSORBING_STATE = 4
DISTANT_METASTASIS_STATES = {2, 3}
DISTANT_METASTASIS_CHECKPOINT_MONTHS = 30.5


def simulate_continuous_time_markov_chain(rng, initial_state, no_of_women, Q):
    lifetimes = np.zeros(no_of_women)
    trajectories = []
    for i in range(no_of_women):
        states_visited = [initial_state]
        entry_times = [0.0]
        current_time = 0.0
        state = initial_state
        while state != ABSORBING_STATE:
            exit_rate = -Q[state, state]
            current_time += rng.exponential(1 / exit_rate)
            jump_probabilities = Q[state] / exit_rate
            jump_probabilities[state] = 0
            state = rng.choice(len(jump_probabilities), p=jump_probabilities)
            states_visited.append(state)
            entry_times.append(current_time)
        lifetimes[i] = current_time
        trajectories.append((states_visited, entry_times))

    return lifetimes, trajectories


def state_at_time(trajectory, query_time):
    states_visited, entry_times = trajectory
    last_index_before_query = 0
    for index, entry_time in enumerate(entry_times):
        if entry_time <= query_time:
            last_index_before_query = index
        else:
            break
    return states_visited[last_index_before_query]


def proportion_in_states_at_time(trajectories, target_states, query_time):
    hits = sum(
        1 for trajectory in trajectories
        if state_at_time(trajectory, query_time) in target_states
    )
    return hits / len(trajectories)


def normal_mean_confidence_interval(sample, confidence=0.95):
    sample_mean = np.mean(sample)
    sample_variance = np.var(sample, ddof=1)
    alpha = 1 - confidence
    standard_normal_quantile = norm.ppf(1 - alpha / 2)
    margin_of_error = standard_normal_quantile * np.sqrt(sample_variance / len(sample))
    return sample_mean - margin_of_error, sample_mean + margin_of_error


def bootstrap_std_confidence_interval(rng, sample, num_resamples=10000, confidence=0.95):
    sample_size = len(sample)
    resampled_stds = np.empty(num_resamples)
    for resample_index in range(num_resamples):
        resample = rng.choice(sample, size=sample_size, replace=True)
        resampled_stds[resample_index] = np.std(resample, ddof=1)
    alpha = 1 - confidence
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    return tuple(np.percentile(resampled_stds, [lower_percentile, upper_percentile]))

if __name__ == "__main__":
    initial_state = 0
    lifetimes, trajectories = simulate_continuous_time_markov_chain(
        RANDOM_NUMBER_GENERATOR, initial_state, NO_OF_WOMEN, Q
    )

    mean_lifetime = float(np.mean(lifetimes))
    std_lifetime = float(np.std(lifetimes, ddof=1))

    mean_ci_lower, mean_ci_upper = normal_mean_confidence_interval(lifetimes)
    std_ci_lower, std_ci_upper = bootstrap_std_confidence_interval(
        RANDOM_NUMBER_GENERATOR, lifetimes
    )

    distant_metastasis_proportion = proportion_in_states_at_time(
        trajectories, DISTANT_METASTASIS_STATES, DISTANT_METASTASIS_CHECKPOINT_MONTHS
    )

    with figure(figsize=(10, 6), save="project1/mathias/part2/figs/lifetime_for_CTMC.png") as fig:
        ax = fig.add_subplot(111)
        histogram(
            lifetimes,
            bins="auto",
            stat="density",
            title="Simulated lifetimes vs CONTINOUS RATE",
            xlabel="Lifetime (months)",
            ylabel="Density",
            label="Simulated",
            ax=ax,
        )
        ax.axvline(
            mean_lifetime,
            color="red",
            linestyle="dashdot",
            linewidth=2.5,
            label=f"Expected lifetime: {mean_lifetime:.2f} months",
        )
        ax.legend()

    logger.info(
        "Lifetime summary",
        mean_lifetime=mean_lifetime,
        std_lifetime=std_lifetime,
    )
    logger.info(
        "95% confidence interval for the mean lifetime",
        lower_bound=mean_ci_lower,
        upper_bound=mean_ci_upper,
    )
    logger.info(
        "95% bootstrap confidence interval for the standard deviation of lifetime",
        lower_bound=std_ci_lower,
        upper_bound=std_ci_upper,
    )
    logger.info(
        "Proportion of women with distant metastasis at checkpoint",
        checkpoint_months=DISTANT_METASTASIS_CHECKPOINT_MONTHS,
        proportion=distant_metastasis_proportion,
    )
    
    
