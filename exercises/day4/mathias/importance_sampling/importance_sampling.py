import numpy as np
from scipy.stats import norm

np.random.seed(42)
from exercises.day4.mathias.part_one_to_four.variance_reduction_methods import monte_carlo_estimator, CI

def report(title, mean, confidence_interval):
    print("=" * 40)
    print(title)
    print(f"Monte Carlo Estimate: {mean}")
    lo, hi = confidence_interval
    print(f"95% Confidence Interval: [{lo}, {hi}]")
    print("=" * 40 + "\n")
def bernoulli_mc_estiator(samples, a):
    n = len(samples)
    indicator = samples >= a
    print(f"Passedn: {np.sum(indicator)} out of {n}")
    theta_is = np.mean(indicator)
    theta_is_var = (1 / n) * np.var(samples)
    return theta_is, (CI(theta_is, theta_is_var, n))

def importance_sampling_estimator(proposal_samples, a, var, n):
    indicator = proposal_samples >= a
    print(f"Passed: {np.sum(indicator)} out of {n}")
    target_density = norm.pdf(proposal_samples, loc=0, scale=1)
    proposal_density = norm.pdf(proposal_samples, loc=a, scale=np.sqrt(var))
    weights = target_density / proposal_density
    weighted_indicator = indicator * weights
    theta_is = np.mean(weighted_indicator)
    theta_is_var = (1 / n) * np.var(weighted_indicator)
    return theta_is, (CI(theta_is, theta_is_var, n))


    

if __name__ == "__main__":
    a = [0, 1, 2, 3, 4, 5]
    n = 100
    for a_val in a:
        samples = np.random.normal(loc=0, scale=1, size=n)
        theta_is, ci_is = bernoulli_mc_estiator(samples, a_val)
        title = f"Crude MC P(Z>a) Estimator (a={a_val})"
        report(title, theta_is, ci_is)

    n_is = 100
    var = 1
    for a_val in a:
        proposal_samples = np.random.normal(loc=a_val, scale=np.sqrt(var), size=n_is)
        theta_is, ci_is = importance_sampling_estimator(proposal_samples, a_val, var, n_is)
        title = f"Importance Sampling P(Z>a) Estimator (a={a_val})"
        report(title, theta_is, ci_is)