import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    NO_OF_WOMEN = 1000
    lifetimes, paths = simulate(rng, NO_OF_WOMEN, P)
    local_recurrence_fraction = local_recurrence_fractions(paths)
    
    # 

    prob_matrix = construct_probabiltity_at_time_of_interest(paths, TIME_OF_INTEREST)
    print(prob_matrix)


        
    