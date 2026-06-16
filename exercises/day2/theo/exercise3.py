# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for synthesizing, code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

n = 10000

# Part 1

# Exponential
def sample_exponential(n, lam=1.0):
    U = np.random.uniform(size=n)
    return -np.log(U) / lam

# Normal via Box-Muller
def sample_normal(n, mu=0.0, sigma=1.0):
    U1 = np.random.uniform(size=n)
    U2 = np.random.uniform(size=n)
    Z = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    # I just keep the first (Z_1) of the two coordinates
    return mu + sigma * Z

# Pareto
def sample_pareto(n, k, beta=1.0):
    U = np.random.uniform(size=n)
    return beta * (1 - U) ** (-1 / k)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Part 1 - Sampling from continuous distributions")

# Exponential
exp_samples = sample_exponential(n)
ax = axes[0, 0]
x = np.linspace(0, np.percentile(exp_samples, 99), 300)
ax.hist(exp_samples, bins=50, density=True, alpha=0.6, label="Simulated")
ax.plot(x, stats.expon.pdf(x), 'r-', label="Theoretical")
ax.set_title("Exponential (λ=1)")
ax.legend()
ks_stat, ks_p = stats.kstest(exp_samples, 'expon')
print(f"Exponential: KS stat={ks_stat:.4f}, p={ks_p:.4f} -> {'REJECT' if ks_p < 0.05 else 'ACCEPT'}")

# Normal
norm_samples = sample_normal(n)
ax = axes[0, 1]
x = np.linspace(-4, 4, 300)
ax.hist(norm_samples, bins=50, density=True, alpha=0.6, label="Simulated")
ax.plot(x, stats.norm.pdf(x), 'r-', label="Theoretical")
ax.set_title("Normal (Box-Muller, μ=0, σ=1)")
ax.legend()
ks_stat, ks_p = stats.kstest(norm_samples, 'norm')
print(f"Normal:      KS stat={ks_stat:.4f}, p={ks_p:.4f} -> {'REJECT' if ks_p < 0.05 else 'ACCEPT'}")

# Pareto for k = 2.05, 2.5, 3, 4
pareto_ks = []
for idx, k in enumerate([2.05, 2.5, 3, 4]):
    samples = sample_pareto(n, k)
    ax = axes[idx // 2, (idx % 2) + (1 if idx // 2 == 0 else 1)]
    row, col = (0, 2) if idx == 0 else (1, idx - 1)
    ax = axes[row, col]
    x = np.linspace(1, np.percentile(samples, 98), 300)
    ax.hist(samples, bins=80, density=True, alpha=0.6, label="Simulated",
            range=(1, np.percentile(samples, 98)))
    ax.plot(x, stats.pareto.pdf(x, b=k), 'r-', label="Theoretical")
    ax.set_title(f"Pareto (β=1, k={k})")
    ax.legend()
    ks_stat, ks_p = stats.kstest(samples, 'pareto', args=(k,))
    pareto_ks.append((k, ks_stat, ks_p))
    print(f"Pareto k={k}: KS stat={ks_stat:.4f}, p={ks_p:.4f} -> {'REJECT' if ks_p < 0.05 else 'ACCEPT'}")

plt.tight_layout()
plt.savefig("part1_distributions.png", dpi=150)
plt.show()

# Part 2
print("\nPart 2 - Pareto mean and variance (n=10,000)")
print(f"{'k':>6}  {'E[X] theo':>12}  {'E[X] sim':>10}  {'Var[X] theo':>14}  {'Var[X] sim':>12}")
for k in [2.05, 2.5, 3, 4]:
    samples = sample_pareto(n, k, beta=1.0)
    theoretical_mean = k / (k - 1)
    theoretical_var  = k / ((k - 1)**2 * (k - 2))
    sim_mean = np.mean(samples)
    sim_var  = np.var(samples, ddof=1)
    print(f"{k:>6}  {theoretical_mean:>12.4f}  {sim_mean:>10.4f}  {theoretical_var:>14.4f}  {sim_var:>12.4f}")

# Part 3
# 100 CIs for mean and variance of N(0,1), each from n=10 samples, 95% level
n = 10
n_rep = 100
true_mean, true_var = 0.0, 1.0

t_crit   = stats.t.ppf(0.975, df=n - 1)
chi2_lo  = stats.chi2.ppf(0.025, df=n - 1)
chi2_hi  = stats.chi2.ppf(0.975, df=n - 1)

covered_mean = 0 # Counter variables, initialized before the loop
covered_var  = 0 # Counter variables, initialized before the loop
for _ in range(n_rep):
    s = sample_normal(n)
    s_mean = np.mean(s)
    s_var   = np.var(s, ddof=1)

    # CI for mean: x̄ ± t * s/√n
    hw = t_crit * np.sqrt(s_var) / np.sqrt(n)
    if s_mean - hw <= true_mean <= s_mean + hw:
        covered_mean += 1

    # CI for variance: [(n-1)s²/χ²_hi, (n-1)s²/χ²_lo]
    ci_var_lo = (n - 1) * s_var / chi2_hi
    ci_var_hi = (n - 1) * s_var / chi2_lo
    if ci_var_lo <= true_var <= ci_var_hi:
        covered_var += 1

print(f"\nPart 3 - Coverage of 95% CIs over {n_rep} replications (n={n})")
print(f"  Mean coverage:     {covered_mean}/{n_rep} = {covered_mean/n_rep:.2%}  (nominal: 95%)")
print(f"  Variance coverage: {covered_var}/{n_rep} = {covered_var/n_rep:.2%}  (nominal: 95%)")

# Part 4
# Composition method for Pareto
def sample_pareto_composition(n, k, beta=1.0):
    Y = np.random.gamma(shape=k, scale=1.0, size=n)
    Z = np.random.exponential(scale=1.0 / Y)
    return Z + beta

n_samples = 10000
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Part 4 - Pareto: composition vs direct inversion")

print("\nPart 4 - Composition vs inversion KS tests")
for ax, k in zip(axes.flat, [2.05, 2.5, 3, 4]):
    s_inv  = sample_pareto(n_samples, k)
    s_comp = sample_pareto_composition(n_samples, k)

    clip = np.percentile(s_inv, 98)
    x = np.linspace(1, clip, 300)
    ax.hist(s_inv,  bins=60, density=True, alpha=0.5, range=(1, clip), label="Inversion")
    ax.hist(s_comp, bins=60, density=True, alpha=0.5, range=(1, clip), label="Composition")
    ax.plot(x, stats.pareto.pdf(x, b=k), 'k-', linewidth=1.5, label="Theoretical")
    ax.set_title(f"Pareto (k={k})")
    ax.legend(fontsize=7)

    ks_inv,  p_inv  = stats.kstest(s_inv,  'pareto', args=(k,))
    ks_comp, p_comp = stats.kstest(s_comp, 'pareto', args=(k,))
    print(f"  k={k}: Inversion  KS={ks_inv:.4f} p={p_inv:.4f} -> {'REJECT' if p_inv  < 0.05 else 'ACCEPT'}")
    print(f"  k={k}: Composition KS={ks_comp:.4f} p={p_comp:.4f} -> {'REJECT' if p_comp < 0.05 else 'ACCEPT'}")

plt.tight_layout()
plt.savefig("part4_pareto_composition.png", dpi=150)
plt.show()
