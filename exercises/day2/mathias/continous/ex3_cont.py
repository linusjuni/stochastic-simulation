import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.plotting import histogram

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
    k = [2.05, 2.5, 3.0, 4.0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, k_val in enumerate(k):
        samples = pareto_dist(k=k_val, beta=beta, size=SIZE)
        ax = axes[i // 2, i % 2]
        histogram(samples, bins="auto", discrete=False, title=f"Pareto Distribution (k={k_val}, beta={beta})", ax=ax)
        ax.set_title(f"Pareto Distribution (k={k_val}, beta={beta})")
    plt.tight_layout()
    plt.savefig("exercises/day2/mathias/continous/pareto_dist.png") 
    
    # 
