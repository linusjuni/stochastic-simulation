# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for synthesizing, code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

import numpy as np

# Part 1: Ross Ch. 8, Exercise 13
def bootstrap_mean_prob(data, a, b, n_boot=100_000):
    data = np.asarray(data, dtype=float)
    n = len(data)
    x_bar = data.mean()
    # Each row is one bootstrap sample (size n, drawn with replacement).
    resamples = np.random.default_rng().choice(data, size=(n_boot, n), replace=True)
    centred = resamples.mean(axis=1) - x_bar
    inside = (a < centred) & (centred < b)
    return inside.mean(), x_bar


# Part 2: Ross Ch. 8, Exercise 15
# We estimate sigma^2 by the sample variance S^2 = sum((X_i - X_bar)^2)/(n-1) and
# want the bootstrap estimate of Var(S^2): draw n_boot resamples (size n, with
# replacement), compute S^2 on each, then take the sample variance of those S^2.
def bootstrap_var_of_s2(data, n_boot=100_000):
    data = np.asarray(data, dtype=float)
    n = len(data)
    resamples = np.random.default_rng().choice(data, size=(n_boot, n), replace=True)
    s2 = resamples.var(axis=1, ddof=1)            # S^2 of each bootstrap sample
    return s2.var(ddof=1)                          # Var(S^2) across bootstraps


# Part 3: bootstrap variance of the mean vs the median for a heavy-tailed sample
# Pareto (course day-2 parametrisation): F(x) = 1 - (beta/x)^k, x >= beta.
# With shape k = 1.05 the mean k/(k-1) = 21 is finite but the variance is INFINITE
# (needs k > 2), so the sample mean is a very imprecise estimator while the median
# stays well-behaved.
def sample_pareto(n, k, beta=1.0):
    U = np.random.default_rng().uniform(size=n)
    return beta * (1 - U) ** (-1 / k)

def bootstrap_variance(data, estimator, n_boot=100_000):
    """Bootstrap estimate of Var(estimator). estimator must accept an axis kwarg
    (e.g. np.mean, np.median): resample with replacement, apply it per resample,
    then take the variance across the n_boot replicates."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    resamples = np.random.default_rng().choice(data, size=(n_boot, n), replace=True)
    reps = estimator(resamples, axis=1)
    return reps.var(ddof=1)


if __name__ == "__main__":
    # Part 1 (b): n = 10, a = -5, b = 5.
    data1 = np.array([56, 101, 78, 67, 93, 87, 64, 72, 80, 69], dtype=float)
    p_hat, x_bar = bootstrap_mean_prob(data1, a=-5, b=5)
    print("── Part 1: Ross Ex. 13 ──")
    print(f"sample mean x_bar = {x_bar:.2f}")
    print(f"bootstrap estimate of p = P(-5 < X_bar - mu < 5) = {p_hat:.4f}")

    # Part 2: n = 15.
    data2 = np.array([5, 4, 9, 6, 21, 17, 11, 20, 7, 10, 21, 15, 13, 16, 8], dtype=float)
    var_s2 = bootstrap_var_of_s2(data2)
    print("\n── Part 2: Ross Ex. 15 ──")
    print(f"sample variance S^2 = {data2.var(ddof=1):.2f}")
    print(f"bootstrap estimate of Var(S^2) = {var_s2:.2f}")

    # Part 3: n = 200 from Pareto(beta = 1, k = 1.05).
    sample = sample_pareto(200, k=1.05, beta=1.0)
    print("\n── Part 3: bootstrap variance of mean vs median (Pareto k=1.05) ──")
    print(f"(1) sample mean   = {sample.mean():.3f}   (theoretical mean = 21)")
    print(f"(1) sample median = {np.median(sample):.3f}   (theoretical median = {2 ** (1 / 1.05):.3f})")
    var_mean = bootstrap_variance(sample, np.mean)
    var_median = bootstrap_variance(sample, np.median)
    print(f"(2) bootstrap Var(mean)   = {var_mean:.3f}   (std = {np.sqrt(var_mean):.3f})")
    print(f"(3) bootstrap Var(median) = {var_median:.4f}   (std = {np.sqrt(var_median):.4f})")
    print(f"(4) Var(mean) / Var(median) = {var_mean / var_median:.1f}x")

