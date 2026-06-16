# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for synthesizing, code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

import numpy as np
import matplotlib.pyplot as plt

def x_geometric(p, n=10000):
    xs = []
    for _ in range(n):
        xs.append(np.floor(np.log(np.random.random())/np.log(1-p))+1)
    return xs

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, p in zip(axes, [0.001, 0.1, 0.9]):
    samples = x_geometric(p)
    print(f"p={p}: sample mean={np.mean(samples):.2f}, theoretical mean={1/p:.2f}")

    x_max = int(np.percentile(samples, 99)) + 1
    x_vals = np.arange(1, x_max + 1)

    ax.hist(samples, bins=np.arange(1, x_max + 2) - 0.5, density=True,
            label='Simulated', alpha=0.7, color='steelblue')
    ax.plot(x_vals, (1 - p) ** (x_vals - 1) * p,
            'r-o', markersize=3, linewidth=1, label='Theoretical')
    ax.set_title(f'Geometric(p={p})')
    ax.set_xlabel('X')
    ax.set_ylabel('Probability')
    ax.legend()

plt.suptitle('Geometric distribution (n=10,000)')
plt.tight_layout()
plt.show()

# Part 2a
def sample_crude(n):
    xs = []
    for _ in range(n):
        U = np.random.random()
        if U <= 7/48:
            X = 1
        elif U <= 7/48+5/48:
            X = 2
        elif U <= 7/48+5/48+1/8:
            X = 3
        elif U <= 7/48+5/48+1/8+1/16:
            X = 4
        elif U <= 7/48+5/48+1/8+1/16+1/4:
            X = 5
        else:
            X = 6
        xs.append(X)
    return xs

probs  = np.array([7/48, 5/48, 1/8, 1/16, 1/4, 5/16])
values = np.arange(1, 7)

samples_crude = np.array(sample_crude(10000))
counts = np.array([np.sum(samples_crude == v) for v in values]) / 10000

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(values - 0.2, counts, width=0.35, label='Simulated', alpha=0.8, color='steelblue')
ax.bar(values + 0.2, probs,  width=0.35, label='Theoretical', alpha=0.8, color='salmon')
ax.set_title('Six-point distribution – crude method (n=10,000)')
ax.set_xlabel('X')
ax.set_ylabel('Probability')
ax.set_xticks(values)
ax.legend()
plt.tight_layout()
plt.show()

# Part 2b
def rejection(n):
    probs  = np.array([7/48, 5/48, 1/8, 1/16, 1/4, 5/16])
    xs = []
    c = 5/16
    counter = 0
    while counter < n:
        U1 = np.random.random()
        U2 = np.random.random()
        I = int(np.floor(6 * U1) + 1)
        if U2 <= probs[I-1]/c:
            xs.append(I)
            counter += 1

    return xs

samples_rej = np.array(rejection(10000))
counts_rej = np.array([np.sum(samples_rej == v) for v in values]) / 10000

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(values - 0.2, counts_rej, width=0.35, label='Simulated', alpha=0.8, color='steelblue')
ax.bar(values + 0.2, probs,      width=0.35, label='Theoretical', alpha=0.8, color='salmon')
ax.set_title('Six-point distribution – rejection method (n=10,000)')
ax.set_xlabel('X')
ax.set_ylabel('Probability')
ax.set_xticks(values)
ax.legend()
plt.tight_layout()
plt.show()

# Part 2c
def build_alias(probs):
    n = len(probs)
    q = list(np.array(probs) * n)   # scaled probabilities
    prob_table  = [0.0] * n
    alias_table = [0]   * n

    small = [i for i in range(n) if q[i] < 1.0]
    large = [i for i in range(n) if q[i] >= 1.0]

    while small and large:
        s = small.pop()
        l = large.pop()
        prob_table[s]  = q[s]
        alias_table[s] = l
        q[l] = q[l] + q[s] - 1.0      # l donates (1 - q[s]) to fill s's column
        if q[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    for i in large + small:            # any leftovers are exactly full
        prob_table[i] = 1.0

    return prob_table, alias_table

def sample_alias(n, prob_table, alias_table):
    xs = []
    for _ in range(n):
        V    = 6 * np.random.random()
        I    = int(np.floor(V))
        frac = V - I
        if frac <= prob_table[I]:
            xs.append(I + 1)           # +1 because values are 1-indexed
        else:
            xs.append(alias_table[I] + 1)
    return xs

prob_table, alias_table = build_alias(probs)
samples_alias = np.array(sample_alias(10000, prob_table, alias_table))
counts_alias  = np.array([np.sum(samples_alias == v) for v in values]) / 10000

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(values - 0.2, counts_alias, width=0.35, label='Simulated',   alpha=0.8, color='steelblue')
ax.bar(values + 0.2, probs,        width=0.35, label='Theoretical', alpha=0.8, color='salmon')
ax.set_title('Six-point distribution – alias method (n=10,000)')
ax.set_xlabel('X')
ax.set_ylabel('Probability')
ax.set_xticks(values)
ax.legend()
plt.tight_layout()
plt.show()

# Part 3 – compare methods across sample sizes
from scipy.stats import chi2 as chi2_dist
import time

def chi2_stat(samples, probs, values):
    n = len(samples)
    observed = np.array([np.sum(samples == v) for v in values])
    expected = probs * n
    return float(np.sum((observed - expected) ** 2 / expected))

n_values = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
methods  = {'Crude': sample_crude, 'Rejection': rejection,
            'Alias': lambda n: sample_alias(n, prob_table, alias_table)}

times  = {m: [] for m in methods}
chi2s  = {m: [] for m in methods}

for n in n_values:
    print(f"n={n:,}")
    for name, fn in methods.items():
        t0 = time.time()
        samples = np.array(fn(n))
        elapsed = time.time() - t0
        times[name].append(elapsed)
        chi2s[name].append(chi2_stat(samples, probs, values))
        print(f"  {name}: {elapsed:.3f}s,  chi2={chi2s[name][-1]:.2f}")

# Plot timing
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for name in methods:
    axes[0].plot(n_values, times[name], marker='o', label=name)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel('n (sample size)')
axes[0].set_ylabel('Time (seconds)')
axes[0].set_title('Computational time vs sample size')
axes[0].legend()

# Plot chi-squared statistic — expected value is df=5 for a correct sampler
for name in methods:
    axes[1].plot(n_values, chi2s[name], marker='o', label=name)
axes[1].axhline(5, color='black', linestyle='--', linewidth=1, label='E[χ²] = 5 (df=5)')
axes[1].set_xscale('log')
axes[1].set_xlabel('n (sample size)')
axes[1].set_ylabel('χ² statistic')
axes[1].set_title('Chi-squared goodness-of-fit vs sample size')
axes[1].legend()

plt.tight_layout()
plt.show()