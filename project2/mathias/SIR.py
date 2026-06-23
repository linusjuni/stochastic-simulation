import numpy as np

from utils.logger import get_logger
from utils.plotting import figure, histogram
from utils.settings import settings

logger = get_logger(__name__)

class SIR:
    def __init__(self, N):
        self.N = N
        self.S0 = 0
        self.I0 = 0
        self.R0 = 0
        self.S = 0
        self.I = 0
        self.R = 0

    def set_initial_conditions(self, S0):
        """Define initial for susceptible, then I = N - S0 and R = 0."""
        self.S0 = S0
        self.I0 = self.N - S0
        self.R0 = 0
        self.reset()

    def get_initial_conditions(self):
        return self.S0, self.I0, self.R0

    def reset(self):
        self.S = self.S0
        self.I = self.I0
        self.R = self.R0

    def transition_rates(self, beta, gamma):
        rate_infection = (beta / self.N) * self.S * self.I
        rate_recovery = gamma * self.I
        return rate_infection, rate_recovery

    def step(self, beta, gamma, rng):
        rate_infection, rate_recovery = self.transition_rates(beta, gamma)
        total_rate = rate_infection + rate_recovery
        if total_rate == 0.0:
            return None
        tau = rng.exponential(1.0 / total_rate)
        if rng.random() < rate_infection / total_rate: # lambda / (lambda + k * mu)
            self.S -= 1
            self.I += 1
        else:
            self.I -= 1
            self.R += 1
        return tau

    def simulate(self, beta, gamma, rng, t_max=np.inf):
        self.reset()
        t = 0.0
        times = [t]
        S_path = [self.S]
        I_path = [self.I]
        R_path = [self.R]
        while self.I > 0 and t < t_max:
            tau = self.step(beta, gamma, rng)
            if tau is None:
                break
            t += tau
            times.append(t)
            S_path.append(self.S)
            I_path.append(self.I)
            R_path.append(self.R)
        return np.array(times), np.array(S_path), np.array(I_path), np.array(R_path)

    def final_size(self):
        return self.N - self.S

    def simulate_many(self, beta, gamma, rng, n_runs, t_max=np.inf):
        final_sizes = np.empty(n_runs, dtype=int)
        for k in range(n_runs):
            self.simulate(beta, gamma, rng, t_max)
            final_sizes[k] = self.final_size()
        return final_sizes
    
    