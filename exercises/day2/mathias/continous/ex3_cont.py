import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.plotting import histogram

np.random.seed(42)  # For reproducibility

def exponential_dist(lambda_, size):
    U = np.random.random(size)
    return -np.log(1 - U) / lambda_

def gaussian_box_mueller(size):
    U1 = np.random.random(size)
    U2 = np.random.random(size)
    Z0 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    Z1 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
    return Z0, Z1

def log_normal_dist(alpha, beta, size):
    Z = gaussian_box_mueller(size)[0]  # Use only one of the two generated normal variables
    return np.exp(alpha + beta * Z)

def multivariate_normal_cholesky(mu, C, size):
    Z = gaussian_box_mueller(size)[0]  # Use only one of the two generated normal variables
    second_term = 0
    for i in range(len(C)):
        for j in range(0, i + 1):
            second_term += C[i, j] * Z[j]
    return mu + second_term

def one_dimensional_gaussian(size):
    U = np.random.random(size)
    n = len(U)
    return (np.sum(U) - n / 2) * np.sqrt(12 / n)

def pareto_dist(k, beta, size):
    U = np.random.random(size)
    dist = beta / (U ** (1 / k))
    return dist

def gaussian_conf_intervals_mean(obs):
    mean = np.mean(obs)
    std = np.std(obs, ddof=1)
    n = len(obs)
    z_score = 1.96  # for 95% confidence
    margin_of_error = z_score * (std / np.sqrt(n))
    return mean - margin_of_error, mean + margin_of_error

def gaussian_conf_intervals_variance(obs):
    n = len(obs)
    var = np.var(obs, ddof=1)
    chi2_lower = np.percentile(np.random.chisquare(n - 1, size=10000), 2.5)
    chi2_upper = np.percentile(np.random.chisquare(n - 1, size=10000), 97.5)
    lower_bound = (n - 1) * var / chi2_upper
    upper_bound = (n - 1) * var / chi2_lower
    return lower_bound, upper_bound

if __name__ == "__main__":
    # a) exponential dist
    SIZE = 10000
    samples = exponential_dist(lambda_=1.0, size=SIZE)
    histogram(samples,discrete=False, title="Exponential Distribution (lambda=1.0)")
    plt.savefig("exercises/day2/mathias/continous/exponential_dist.png")
    
    # b) Gaussian Box-Mueller and contour plot
    Z0, Z1 = gaussian_box_mueller(size=SIZE)
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=Z0, y=Z1, fill=True, cmap="viridis")
    plt.title("Gaussian Distribution (Box-Mueller)")
    plt.xlabel("Z0")
    plt.ylabel("Z1")
    plt.savefig("exercises/day2/mathias/continous/gaussian_box_mueller.png")
    
    # c) pareto distribution
    beta = 1.0
    k = np.array([2.05, 2.5, 3.0, 4.0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    theoretical_pareto_means = beta * k / (k - 1)
    theoretical_pareto_variances = (beta ** 2) * k / ((k - 1) ** 2 * (k - 2))
    sampled_pareto_means = []
    sampled_pareto_variances = []
    for i, k_val in enumerate(k):
        samples = pareto_dist(k=k_val, beta=beta, size=SIZE)
        sampled_pareto_means.append(float(np.mean(samples)))
        sampled_pareto_variances.append(float(np.var(samples)))
        ax = axes[i // 2, i % 2]
        histogram(samples, bins="auto", discrete=False, title=f"Pareto Distribution (k={k_val}, beta={beta})", ax=ax)
        ax.set_title(f"Pareto Distribution (k={k_val}, beta={beta})")
    plt.tight_layout()
    plt.savefig("exercises/day2/mathias/continous/pareto_dist.png") 
    
    # 2) Analysis of the Pareto distribution
    print("Theoretical Pareto Means:", theoretical_pareto_means)
    print("Sampled Pareto Means:    ", sampled_pareto_means)
    print("Theoretical Pareto Variances:", theoretical_pareto_variances)
    print("Sampled Pareto Variances:    ", sampled_pareto_variances)
    # note variance of pareto is not defined for k <= 2, and mean is not defined for k <= 1, which is reflected in the theoretical values.
    
    # 3) 100 95% confidence intervals from 10 samples for the
    # mean and variance of a Gaussian distribution
    gaussian_samples = gaussian_box_mueller(size=SIZE)[0] # Use only one
    confidence_intervals = [gaussian_conf_intervals_mean(np.random.choice(gaussian_samples, size=10, replace=False)) for _ in range(100)]
    print("Sample Confidence Intervals for the Mean of Gaussian Distribution:")
    # plot spread of confidence intervals
    lower_bounds = [ci[0] for ci in confidence_intervals]
    upper_bounds = [ci[1] for ci in confidence_intervals]
    plt.figure(figsize=(10, 6))
    plt.plot(lower_bounds, label="Lower Bound", marker='o', linestyle='None')
    plt.plot(upper_bounds, label="Upper Bound", marker='o', linestyle='None')
    plt.axhline(y=0, color='r', linestyle='--', label="True Mean (0)")
    plt.title("95% Confidence Intervals for the Mean of Gaussian Distribution")
    plt.xlabel("Interval Index")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("exercises/day2/mathias/continous/confidence_intervals.png")

    # confidence intervals for variance
    confidence_intervals_var = [gaussian_conf_intervals_variance(np.random.choice(gaussian_samples, size=10, replace=False)) for _ in range(100)]
    print("Sample Confidence Intervals for the Variance of Gaussian Distribution:")
    lower_bounds_var = [ci[0] for ci in confidence_intervals_var]
    upper_bounds_var = [ci[1] for ci in confidence_intervals_var]
    plt.figure(figsize=(10, 6))
    plt.plot(lower_bounds_var, label="Lower Bound", marker='o', linestyle='None')
    plt.plot(upper_bounds_var, label="Upper Bound", marker='o', linestyle='None')
    plt.axhline(y=1, color='r', linestyle='--', label="True Variance (1)")
    plt.title("95% Confidence Intervals for the Variance of Gaussian Distribution")
    plt.xlabel("Interval Index")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("exercises/day2/mathias/continous/confidence_intervals_variance.png")
    
    

    

    
