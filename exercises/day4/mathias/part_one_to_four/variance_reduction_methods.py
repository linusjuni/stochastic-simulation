import numpy as np
from scipy.stats import norm

np.random.seed(42)

def integrand(U):
    return np.exp(U)

def analytical_solution():
    mean = np.exp(1) - 1
    var  = ((np.exp(1)**2 - 1) / 2) - (np.exp(1) - 1)**2
    return mean, var

def monte_carlo_estimator(samples):
    n = len(samples)
    theta_mc = np.mean(integrand(samples))
    theta_mc_var = (1 / n) * np.var(integrand(samples))
    t_quantile = norm.ppf(0.975)
    return theta_mc, (theta_mc - t_quantile * np.sqrt(theta_mc_var / n), theta_mc + t_quantile * np.sqrt(theta_mc_var / n))
    
def mc_antithetic(samples):
    n = len(samples)
    X = integrand(samples)
    Y = integrand(1 - samples)
    mc_mean = (1/n) * np.sum((X + Y) / 2)
    mc_cov  = np.mean(X * Y) - np.mean(X) * np.mean(Y)
    mc_var  = (1/4) * (np.var(X) + np.var(Y) + 2 * mc_cov)
    t_quantile = norm.ppf(0.975)
    return mc_mean, (mc_mean - t_quantile * np.sqrt(mc_var / n), mc_mean + t_quantile * np.sqrt(mc_var / n))

    
def report(title, mean, confidence_interval, analytical=analytical_solution()):
    print("=" * 40)
    print(title)
    print(f"Monte Carlo Estimate: {mean}")
    print(f"Analytical Solution: {analytical[0]} with variance {analytical[1]}")
    lo, hi = confidence_interval
    print(f"95% Confidence Interval: [{lo}, {hi}]")
    print(f"Is the true mean within the confidence interval? {'Yes' if lo <= analytical[0] <= hi else 'No'}")
    
    
    
if __name__ == "__main__":
    N = 100
    samples = np.random.random(size=N)
    estimate, confidence_interval = monte_carlo_estimator(samples)
    title = f"Monte Carlo Estimation of E[exp(U)] with N = {N} samples"
    report(title, estimate, confidence_interval)

    # Antithetic Variates
    estimate_antithetic, confidence_interval_antithetic = mc_antithetic(samples)
    title_antithetic = f"Monte Carlo Estimation with Antithetic Variates with N = {N} samples"
    report(title_antithetic, estimate_antithetic, confidence_interval_antithetic)
    
