# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

import numpy as np
import scipy

# Part 1 - Crude Monte Carlo estimate of integral_0^1 e^x dx
def crude(n):
    U = np.random.random(n)
    samples = np.exp(U)
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    half_width = scipy.stats.t.ppf(0.975, df=n-1) * std / np.sqrt(n)
    return mean, (mean - half_width, mean + half_width)

mean, ci = crude(n=100)
print('-'*34)
print(f"Part 1 - Crude MC: mean = {mean:.6f}, 95% CI = ({ci[0]:.6f}, {ci[1]:.6f})")
print(f"  True value: {np.e - 1:.6f}")

# Part 2
def antithetic(n):
    U = np.random.random(n)
    samples = (np.exp(U) + np.exp(1-U)) / 2
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    half_width = scipy.stats.t.ppf(0.975, df=n-1) * std / np.sqrt(n)
    return mean, (mean - half_width, mean + half_width)

mean, ci = antithetic(n=50)
print('-'*34)
print(f"Part 2 - antithetic MC: mean = {mean:.6f}, 95% CI = ({ci[0]:.6f}, {ci[1]:.6f})")
print(f"  True value: {np.e - 1:.6f}")

# Part 3
def control(n):
    U = np.random.random(size=n)
    X = np.exp(U)
    Z = U

    c = -np.cov(X, Z, ddof=1)[0, 1] / np.var(Z, ddof=1)
    Y = X + c * (Z - 0.5)

    mean = np.mean(Y)
    std = np.std(Y, ddof=1)
    half_width = scipy.stats.t.ppf(0.975, df=n-1) * std / np.sqrt(n)
    return mean, (mean - half_width, mean + half_width)

mean, ci = control(n=100)
print('-'*34)
print(f"Part 3 - Control MC: mean = {mean:.6f}, 95% CI = ({ci[0]:.6f}, {ci[1]:.6f})")
print(f"  True value: {np.e - 1:.6f}")

# Part 4
def stratified(n, k=10):
    Ys = []
    for i in range(n//k):
        Xs = []
        for j in range(k):
            U = np.random.uniform(j/k,(j+1)/k)
            Xs.append(np.exp(U))
        Ys.append(np.mean(Xs))
    mean = np.mean(Ys)
    std = np.std(Ys, ddof=1)
    half_width = scipy.stats.t.ppf(0.975, df=(n//k)-1) * std / np.sqrt(n//k)
    return mean, (mean - half_width, mean + half_width)

mean, ci = stratified(n=100)
print('-'*34)
print(f"Part 4 - Stratified MC: mean = {mean:.6f}, 95% CI = ({ci[0]:.6f}, {ci[1]:.6f})")
print(f"  True value: {np.e - 1:.6f}")

# Part 5


# Part 6 has been removed from pensum this year

# Part 7


# Part 8

# Part 9