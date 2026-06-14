# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from math import factorial

# Parameters (from exercise 4)
A = 8
m = 10

# Part 1
def g(i, A):
    return A**i / factorial(i)

def metropolis_hastings_erlang(A, m, n_samples, burn_in=5000, thin=10):
    """
    Random-walk M-H on {0,...,m} with proposal y = x ± 1.
    Thinning keeps every `thin`-th post-burn-in step to reduce autocorrelation.
    """
    x = m // 2
    samples = []
    total_steps = burn_in + n_samples * thin

    for step in range(total_steps):
        y = x + np.random.choice([-1, 1])

        if 0 <= y <= m:
            ratio = (A / (x + 1)) if y == x + 1 else (x / A)
            if np.random.uniform() < min(1.0, ratio):
                x = y

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(x)

    return np.array(samples)

def theoretical_probs(A, m):
    raw = np.array([g(i, A) for i in range(m + 1)])
    return raw / raw.sum()

def chi_squared_test(samples, A, m):
    n = len(samples)
    probs = theoretical_probs(A, m)
    observed = np.array([np.sum(samples == i) for i in range(m + 1)])
    expected = n * probs

    # merge bins with expected count < 5
    chi2_stat, p_value = stats.chisquare(observed, f_exp=expected)
    return chi2_stat, p_value

# ── Run ───────────────────────────────────────────────────────────────────────
np.random.seed(42)
n_samples = 20_000          # thinned samples (total chain: 5000 + 20000*10 = 205,000 steps)
samples = metropolis_hastings_erlang(A, m, n_samples, burn_in=5000, thin=10)

chi2, p = chi_squared_test(samples, A, m)
print(f"Thinned samples:       {len(samples)}")
print(f"Chi-squared statistic: {chi2:.4f}")
print(f"P-value:               {p:.4f}")
print(f"Degrees of freedom:    {m}")
print(f"Reject H0 (α=0.05):   {p < 0.05}")

# ── Plot ──────────────────────────────────────────────────────────────────────
probs = theoretical_probs(A, m)
states = np.arange(m + 1)
empirical = np.array([np.mean(samples == i) for i in states])

plt.figure(figsize=(8, 4))
plt.bar(states - 0.2, probs, width=0.4, label="Theoretical", alpha=0.8)
plt.bar(states + 0.2, empirical, width=0.4, label="MCMC (M-H)", alpha=0.8)
plt.xlabel("i (number of busy lines)")
plt.ylabel("Probability")
plt.title(f"Erlang distribution — M-H sampler  (χ²={chi2:.2f}, p={p:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig("part1_erlang_mh.png", dpi=150)
plt.show()
