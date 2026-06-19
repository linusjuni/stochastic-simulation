import numpy as np
from scipy.stats import kstest, logrank, CensoredData
import scipy.linalg as la

from utils.logger import get_logger
from utils.plotting import figure

from project1.mathias.part2.task7 import simulate_continuous_time_markov_chain, state_at_time
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

def create_fake_time_series_data(lifetimes: np.ndarray, trajectories: list[tuple[np.ndarray, np.ndarray]], lag : int = 48):
    time_series_data = []
    lags = [i for i in range(0, int(max(lifetimes) + lag), lag)]
    lags_for_series = []
    for i in range(len(trajectories)):
        series = [state_at_time(trajectories[i], t) for t in lags if t <= lifetimes[i] + lag]
        time_series_data.append(series)
        lags_for_series.append(lags[:len(series)])
    return time_series_data, lags_for_series
        

    

if __name__ == "__main__":
    lifetimes, trajectories = simulate_continuous_time_markov_chain(
        RANDOM_NUMBER_GENERATOR, initial_state=0, no_of_women=NO_OF_WOMEN, Q=Q
    )
    time_series_data, lags_for_each_series = create_fake_time_series_data(lifetimes, trajectories)

    
    