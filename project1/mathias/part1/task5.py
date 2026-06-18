import numpy as np
from scipy.stats import chisquare, kstest
import time as t
import matplotlib.pyplot as plt
import seaborn as sns

from task1 import simulate, local_recurrence_fractions
from task3 import discrete_phase_type_distribution, expectation_of_discrete_phase_type_distribution
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

N_WOMEN_PER_REP = 200
N_REPLICATIONS = 100
THRESHOLD = 350


def run_replications(rng, n_replications, n_women, P, threshold, start_time=t.perf_counter()):
    fractions = np.zeros(n_replications)
    mean_lifetimes = np.zeros(n_replications)
    for r in range(n_replications):
        print(f"Running replication {r+1}/{n_replications}... Time elapsed: {t.perf_counter() - start_time:.2f} seconds")
        lifetimes, _ = simulate(rng, n_women, P)
        fractions[r] = np.mean(lifetimes < threshold)
        mean_lifetimes[r] = np.mean(lifetimes)
    return fractions, mean_lifetimes


def control_variate_estimate(X, Z, mu_z):
    c = -np.cov(X, Z, ddof=1)[0, 1] / np.var(Z, ddof=1)
    Y = X + c * (Z - mu_z)
    return Y, c


if __name__ == "__main__":
    initial_distribution = np.array([1, 0, 0, 0])
    mu_z = expectation_of_discrete_phase_type_distribution(P, initial_distribution)
    X, Z = run_replications(
        RANDOM_NUMBER_GENERATOR, N_REPLICATIONS, N_WOMEN_PER_REP, P, THRESHOLD
    )
    Y, c = control_variate_estimate(X, Z, mu_z)

    crude_mean = np.mean(X)
    crude_var = np.var(X, ddof=1)
    cv_mean = np.mean(Y)
    cv_var = np.var(Y, ddof=1)

    logger.info(
        "Crude Monte Carlo",
        estimate=crude_mean,
        variance=crude_var,
        std_error=np.sqrt(crude_var / N_REPLICATIONS),
    )
    logger.info(
        "Control variate",
        estimate=cv_mean,
        variance=cv_var,
        std_error=np.sqrt(cv_var / N_REPLICATIONS),
        c=c,
        mu_z=mu_z,
    )
    logger.info(
        "Variance reduction",
        ratio=crude_var / cv_var,
        percent_reduction=100 * (1 - cv_var / crude_var),
    )
