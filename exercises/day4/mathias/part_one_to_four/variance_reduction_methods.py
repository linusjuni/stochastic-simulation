import numpy as np
from scipy.stats import norm

np.random.seed(42)

def integrand(U):
    return np.exp(U)

def analytical_solution():
    mean = np.exp(1) - 1
    var  = ((np.exp(1)**2 - 1) / 2) - (np.exp(1) - 1)**2
    return mean, var

def CI(mean, var, n, confidence=0.95):
    alpha = 1 - confidence
    t_quantile = norm.ppf(1 - alpha / 2)
    margin_of_error = t_quantile * np.sqrt(var / n)
    return mean - margin_of_error, mean + margin_of_error

def monte_carlo_estimator(samples):
    n = len(samples)
    theta_mc = np.mean(integrand(samples))
    theta_mc_var = (1 / n) * np.var(integrand(samples))
    return theta_mc, (CI(theta_mc, theta_mc_var, n))
    
def cov(x,y):
    return np.mean(x * y) - np.mean(x) * np.mean(y)
def mc_antithetic(samples):
    n = len(samples)
    X = integrand(samples)
    Y = integrand(1 - samples)
    mc_mean = (1/n) * np.sum((X + Y) / 2)
    mc_cov  = cov(X, Y)
    mc_var  = (1/4) * (np.var(X) + np.var(Y) + 2 * mc_cov)
    return mc_mean, (CI(mc_mean, mc_var, n))

def mc_control_variate(samples):
    n = len(samples)
    X = integrand(samples)
    Z = samples 
    c = -cov(X, Z) / np.var(Z)
    Y = X + c * (Z - 0.5)
    mc_mean = np.mean(Y)
    mc_var = np.var(Y) / n
    return mc_mean, (CI(mc_mean, mc_var, n))

def stratified_sampling(strata_samples):
    n_strata = strata_samples.shape[0]
    estimates = []
    
    for i in range(n_strata):
        stratum_mean = np.mean(integrand(strata_samples[i]))
        estimates.append(stratum_mean)
    
    overall_mean = np.mean(estimates)
    overall_var = np.var(estimates) / n_strata
    return overall_mean, (CI(overall_mean, overall_var, n_strata))


    
def report(title, mean, confidence_interval, analytical=analytical_solution()):
    print("=" * 40)
    print(title)
    print(f"Analytical Solution: {analytical[0]} with variance {analytical[1]}")
    print(f"Monte Carlo Estimate: {mean}")
    lo, hi = confidence_interval
    print(f"95% Confidence Interval: [{lo}, {hi}]")
    print(f"Is the true mean within the confidence interval? {'Yes' if lo <= analytical[0] <= hi else 'No'}")
    print("=" * 40 + "\n")
    
    
    
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
    
    # Control Variates
    estimate_cv, cont_int_cv = mc_control_variate(samples)
    title_cv = f"Monte Carlo Estimation with Control Variates with N = {N} samples"
    report(title_cv, estimate_cv, cont_int_cv)
    
    # Stratified Sampling
    x = 10
    strata_samples = np.array(
        [
        np.concatenate([np.random.uniform(i/10, (i+1)/10, size=1) for i in range(10)])
        for _ in range(x)
        ]
    )

    estimate_stratified, confidence_interval_stratified = stratified_sampling(strata_samples)
    title_stratified = f"Monte Carlo Estimation with Stratified Sampling with N = {N} samples"
    report(title_stratified, estimate_stratified, confidence_interval_stratified)

