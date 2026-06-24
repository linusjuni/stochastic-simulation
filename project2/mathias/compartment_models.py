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

    def simulate_ensemble(self, beta, gamma, rng, n_runs, t_max=np.inf):
        return [self.simulate(beta, gamma, rng, t_max) for _ in range(n_runs)]


class SEIR(SIR):
    def __init__(self, N):
        super().__init__(N)
        self.E0 = 0
        self.E = 0

    def set_initial_conditions(self, S0):
        """Define initial for susceptible and exposed, then I = N - S0 - E0 and R = 0."""
        self.S0 = S0
        self.E0 = self.N - S0
        self.I0 = 0
        self.R0 = 0
        self.reset()

    def get_initial_conditions(self):
        return self.S0, self.E0, self.I0, self.R0

    def reset(self):
        super().reset()
        self.E = self.E0

    def transition_rates(self, beta, sigma, gamma):
        rate_infection = (beta / self.N) * self.S * self.I
        rate_exposed_to_infected = sigma * self.E
        rate_recovery = gamma * self.I
        return rate_infection, rate_exposed_to_infected, rate_recovery

    def categorical(self, transition_rates, rng):
        total_rate = sum(transition_rates)
        if total_rate == 0.0:
            return None
        u = rng.random()
        
        cdf = np.cumsum(transition_rates) / total_rate
        
        return np.searchsorted(cdf, u)    

    def step(self, beta, sigma, gamma, rng):
        rate_infection, rate_exposed_to_infected, rate_recovery = self.transition_rates(beta, sigma, gamma)
        total_rate = rate_infection + rate_exposed_to_infected + rate_recovery
        if total_rate == 0.0:
            return None
        tau = rng.exponential(1.0 / total_rate)
        
        categorical_index = self.categorical([rate_infection, rate_exposed_to_infected, rate_recovery], rng)
        
        match categorical_index:
            case 0:  # Infection event
                self.S -= 1
                self.E += 1
                self.I += 0
                self.R += 0
            case 1:  # Exposed to Infected event
                self.S += 0
                self.E -= 1
                self.I += 1
                self.R += 0
            case 2:  # Recovery event
                self.S += 0
                self.E += 0
                self.I -= 1
                self.R += 1
            
        return tau

    def simulate(self, beta, sigma, gamma, rng, t_max=np.inf):
        self.reset()
        t = 0.0
        times = [t]
        S_path = [self.S]
        E_path = [self.E]
        I_path = [self.I]
        R_path = [self.R]
        while (self.I > 0 or self.E > 0) and t < t_max:
            tau = self.step(beta, sigma, gamma, rng)
            if tau is None:
                break
            t += tau
            times.append(t)
            S_path.append(self.S)
            E_path.append(self.E)
            I_path.append(self.I)
            R_path.append(self.R)
        return np.array(times), np.array(S_path), np.array(E_path), np.array(I_path), np.array(R_path)
    
    def simulate_ensemble(self, beta, sigma, gamma, rng, n_runs, t_max=np.inf):
        return [self.simulate(beta, sigma, gamma, rng, t_max) for _ in range(n_runs)]
    
    
    
    
            
    