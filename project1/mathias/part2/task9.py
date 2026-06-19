import numpy as np
from scipy.stats import kstest, logrank, CensoredData
import scipy.linalg as la

from utils.logger import get_logger
from utils.plotting import figure

from project1.mathias.part2.task7 import simulate_continuous_time_markov_chain
from project1.mathias.part2.task8 import empirical_cdf

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
Q_preventive_treatment = np.array([
    [0, 0.0025, 0.00125, 0.0, 0.001],
    [0.0, 0, 0.0, 0.002, 0.005],
    [0.0, 0.0, 0, 0.003, 0.005],
    [0.0, 0.0, 0.0, 0, 0.009],
    [0.0, 0.0, 0.0, 0.0, 0.0],
])
for i in range(Q_preventive_treatment.shape[0]):
    Q_preventive_treatment[i, i] = -np.sum(Q_preventive_treatment[i])
    
STATES = {
    0: "cancer removal",
    1: "local recurrence",
    2: "distant metastasis",
    3: "both local and distant",
    4: "death",
}

def survival_function(lifetimes, no_of_women):
    times = np.arange(0, max(lifetimes) + 1)
    N = no_of_women
    dead_at_time_t = np.array([np.sum(lifetimes <= t) for t in times])
    survival_probabilities = (N - dead_at_time_t) / N
    return times, survival_probabilities
    
    
    
if __name__ == "__main__":
    lifetimes, _ = simulate_continuous_time_markov_chain(
        RANDOM_NUMBER_GENERATOR, initial_state=0, no_of_women=NO_OF_WOMEN, Q=Q
    )
    preventive_treatment_lifetimes,_ = simulate_continuous_time_markov_chain(
        RANDOM_NUMBER_GENERATOR, initial_state=0, no_of_women=NO_OF_WOMEN, Q=Q_preventive_treatment
    )
    
    evaluation_times = np.linspace(0, max(max(lifetimes), max(preventive_treatment_lifetimes)), NUMBER_OF_CDF_GRID_POINTS)
    empirical_cdf_values = empirical_cdf(lifetimes, evaluation_times)
    preventive_empirical_cdf_values = empirical_cdf(preventive_treatment_lifetimes, evaluation_times)
    
    with figure(figsize=(10, 6), save="project1/mathias/part2/figs/preventive_vs_non_preventive.png") as fig:
        ax = fig.add_subplot(111)
        ax.plot(evaluation_times, empirical_cdf_values, label="Empirical CDF (Original)", linewidth=2)
        ax.plot(evaluation_times, preventive_empirical_cdf_values, label="Empirical CDF (Preventive Treatment)", linewidth=2)
        ax.set_xlabel("Time")
        ax.set_ylabel("CDF")
        ax.set_title("Empirical CDF Comparison")
        ax.legend()
        ax.grid()   

    survival_times, survival_probabilities = survival_function(lifetimes, NO_OF_WOMEN)
    survial_times_preventive, survival_probabilities_preventive = survival_function(preventive_treatment_lifetimes, NO_OF_WOMEN)
    
    with figure(figsize=(10, 6), save="project1/mathias/part2/figs/survival_comparison.png") as fig:
        ax = fig.add_subplot(111)
        ax.step(survival_times, survival_probabilities, where='post', label="Survival Function (Original)", linewidth=2)
        ax.step(survial_times_preventive, survival_probabilities_preventive, where='post', label="Survival Function (Preventive Treatment)", linewidth=2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.set_title("Survival Function Comparison")
        ax.legend()
        ax.grid()

    control_data = CensoredData(lifetimes)
    treatment_data = CensoredData(preventive_treatment_lifetimes)
    result = logrank(control_data, treatment_data)
    logger.info(
        "Log-rank test result comparing survival curves",
        statistic=result.statistic,
        p_value=result.pvalue
    )
    
        