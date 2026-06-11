import numpy as np
from scipy.stats import norm, expon, pareto

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

def importance_sampling_estimator(proposal_samples,gen, a, var, n):
    indicator = proposal_samples >= a
    print(f"Passed: {np.sum(indicator)} out of {n}")
    target_density = gen.pdf(proposal_samples, loc=0, scale=1)
    proposal_density = gen.pdf(proposal_samples, loc=a, scale=np.sqrt(var))
    weights = target_density / proposal_density
    weighted_indicator = indicator * weights
    theta_is = np.mean(weighted_indicator)
    theta_is_var = (1 / n) * np.var(weighted_indicator)
    return theta_is, (CI(theta_is, theta_is_var, n))

def is_exponentials(n, lam_):
    
    samples = np.random.exponential(scale=1/lam_, size=n)
    indicator = [(s <= 1) for s in samples]
    target_evals = np.exp(samples)
    proposal_evals = expon.pdf(samples, scale=1/lam_)
    weights = (indicator * target_evals) / proposal_evals
    theta_is = np.mean(weights)
    theta_is_var = (1 / n) * np.var(weights)    
    
    return theta_is, (CI(theta_is, theta_is_var, n))

def is_pareto(n, k):
    samples = np.random.pareto(a=k, size=n)
    proposal_evals = n * [1 / (k - 1)]
    target_evals = pareto.pdf(samples, b=k)
    indicator = samples >= 0
    weights = (indicator * target_evals) / proposal_evals
    theta_is = np.mean(weights)
    theta_is_var = (1 / n) * np.var(weights)
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
        theta_is, ci_is = importance_sampling_estimator(proposal_samples, norm,a_val, var, n_is)
        title = f"Importance Sampling P(Z>a) Estimator (a={a_val})"
        report(title, theta_is, ci_is)

    # part 8) Importance sampling for exponential distribution
    n_is = 100
    lam_ = [0.5, 1, 2, 5, 10]
    for lam in lam_:
        theta_is, ci_is = is_exponentials(n_is, lam)
        title = f"Importance Sampling for Exponential Distribution (lambda={lam})"
        report(title, theta_is, ci_is)

    # part 9) Pareto IS 
    
    n_is = 100
    k_values = [1.05, 1.5, 2, 3, 5, 10]
    for k in k_values:
        theta_is, ci_is = is_pareto(n_is, k)
        title = f"Importance Sampling for Pareto Distribution (k={k})"
        report(title, theta_is, ci_is)
    
    