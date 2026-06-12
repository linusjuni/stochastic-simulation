import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import factorial
from scipy.stats import norm, expon, pareto, erlang

np.random.seed(42)
from exercises.day4.mathias.part_one_to_four.variance_reduction_methods import monte_carlo_estimator, CI
from utils.plotting import histogram

def report(title, mean, confidence_interval):
    print("=" * 40)
    print(title)
    print(f"Monte Carlo Estimate: {mean}")
    lo, hi = confidence_interval
    print(f"95% Confidence Interval: [{lo}, {hi}]")
    print("=" * 40 + "\n")

def random_walk(initial_state, lo, hi, n_steps):
    states = [initial_state]
    for _ in range(n_steps):
        step = np.random.random()
        if step < 0.5:
            new_state = states[-1] - 1
        else:
            new_state = states[-1] + 1
        if new_state < lo:
            new_state = lo
        elif new_state > hi:
            new_state = hi
        states.append(new_state)
    return states

def joint_density_rvs_part1(i):
    A = 8
    evaluation = (A**i) / factorial(i)
    return evaluation

def joint_density_rvs_part2(i, j):
    A1, A2 = 4, 4
    evaluation = ((A1**i) / factorial(i) * (A2**j) / factorial(j))
    return evaluation

def metropolis_hastings_part1(n_samples, initial_state=5, lo=0, hi=10, thinning=10):
    """Metropolis hastings algorithm defined with help of joint_density_rvs, which is proportional to the posterior
    Note the proposal distribution is a random walk that is 0.5 probabilty of i - 1 or i + 1. 
    If outside range (lo, hi) simply reject.

    Args:
        n_samples (int): number of samples to generate
        initial_state (int): Initial state of chain. Defaults to 5.
        lo (int): lowest possible value of random walk. Defaults to 0.
        hi (int): highest possible value of random walk. Defaults to 10.

    Returns:
        _type_: _description_
    """

    samples = [initial_state]
    for _ in range(n_samples):
        current_state = samples[-1]
        proposed_state = current_state + np.random.choice([-1,1])
        if proposed_state < lo or proposed_state > hi:
            samples.append(current_state)
            continue
        acceptance_prob = min(1, joint_density_rvs_part1(proposed_state) / joint_density_rvs_part1(current_state))
        if np.random.random() < acceptance_prob:
            samples.append(proposed_state)
        else:
            samples.append(current_state)
    return samples[::thinning]

def metropolis_hastings_part2(n_samples, init_state_1=5,init_state_2=5, lo=0, hi=10, thinning=10):
    samples = [(init_state_1, init_state_2)]
    for _ in range(n_samples):
        curr_state_i, curr_state_j = samples[-1]
        p = np.random.random()
        if p < 0.5:
            proposed_i, proposed_j = curr_state_i + np.random.choice([-1, 1]), curr_state_j
        else:
            proposed_i, proposed_j = curr_state_i, curr_state_j + np.random.choice([-1, 1])
        if proposed_i < lo or proposed_i > hi or proposed_j < lo or proposed_j > hi or proposed_i + proposed_j > hi:
            samples.append((curr_state_i, curr_state_j))
            continue
        acceptance_prob = min(1, joint_density_rvs_part2(proposed_i, proposed_j) / joint_density_rvs_part2(curr_state_i, curr_state_j))
        p = np.random.random()
        if p < acceptance_prob:
            samples.append((proposed_i, proposed_j))
        else:
            samples.append((curr_state_i, curr_state_j))
    return samples[::thinning]
        
def conditional_density_rvs_part2(A_, i, j):
    m = 10
    support = np.arange(0, m - j + 1)
    cond_density = np.array([(A_**k) / factorial(k) for k in support])
    cond_prob = cond_density / np.sum(cond_density)
    return np.random.choice(support, p=cond_prob) # choose new i given j or reverse
    

def gibbs_sampling_part2(n_samples, init_state_1, init_state_2, thinning=10):
   samples = [(init_state_1, init_state_2)]
   for _ in range(n_samples):
        i, j= samples[-1]
        new_i = conditional_density_rvs_part2(4, i, j)
        new_j = conditional_density_rvs_part2(4, j, new_i)
        samples.append((new_i, new_j))
   return samples[::thinning]
       
           




if __name__ == "__main__":
    n_samples = 100000
    lo, hi = 0, 10
    initial_state = 0
    samples = metropolis_hastings_part1(n_samples, initial_state, lo, hi)
    histogram(samples, discrete=True, title="Metropolis-Hastings Samples, A = 8, m=10", xlabel="State", ylabel="Frequency")
    plt.savefig("exercises/day5/mathias/metropolis_hastings_samples.png")
    plt.show()

    histogram(erlang.rvs(a=8, size=100000 ), discrete=True, title="Samples from Erlang Distribution (A=8)", xlabel="State", ylabel="Frequency")
    plt.show()

    # part 2) A1, A2 and 0<= i + j <= m
    n_samples = int(1e6)
    lo, hi = 0, 10
    init_state_1, init_state_2 = 0, 0
    A1, A2 = 4, 4
    samples = metropolis_hastings_part2(n_samples, init_state_1, init_state_2, lo, hi)
    histogram([i + j for i, j in samples], discrete=True, title="Metropolis-Hastings Samples, A1 = A2 = 4, m=10", xlabel="State", ylabel="Frequency")
    plt.savefig("exercises/day5/mathias/metropolis_hastings_samples_part2.png")
    plt.show()
    
    # part 3) Gibbs sampling
    n_samples = int(1e6)
    init_state_1, init_state_2 = 0, 0
    samples = gibbs_sampling_part2(n_samples, init_state_1, init_state_2)
    histogram([i + j for i, j in samples], discrete=True, title="Gibbs Sampling Samples, A1 = A2 = 4, m=10", xlabel="State", ylabel="Frequency")
    plt.savefig("exercises/day5/mathias/gibbs_samples_part2.png")
    plt.show() 
