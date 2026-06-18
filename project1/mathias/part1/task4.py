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

def women_with_cancer_within_12_months(path):
     """Given a path of states, determine if the women had cancer reappering within 12 months"""
     return len(path) > 12 and (path[12] in (1,2,3))
 
def simulate_with_condition(rng, size, P, condition_fn):
    """Simulate until we have no of women = size given they satisfy the condition function"""
    lifetimes = []
    paths = []
    while len(lifetimes) < size:
        lifetime, path = simulate(rng, 1, P)
        if condition_fn(path[0]):
            lifetimes.append(lifetime[0])
            paths.append(path[0])
    return np.array(lifetimes), paths

if __name__ == "__main__":
    lifetimes, _ = simulate_with_condition(RANDOM_NUMBER_GENERATOR, NO_OF_WOMEN, P, women_with_cancer_within_12_months)
    initial_distribution = np.array([1,0,0,0])
    expected_lifetime = np.mean(lifetimes)
    
    logger.info("Expected lifetime given that cancer reappeared within 12 months", expected_lifetime=expected_lifetime)
    