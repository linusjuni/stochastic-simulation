# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for synthesizing, code structering, coding, and verification.
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
print(f"\n--- Part 1")
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

# Part 2
A1, A2 = 4, 4

def g2(i, j):
    return (A1**i / factorial(i)) * (A2**j / factorial(j))

def valid_states(m):
    return [(i, j) for i in range(m + 1) for j in range(m + 1 - i)]

def theoretical_probs_2d(m):
    states = valid_states(m)
    raw = np.array([g2(i, j) for i, j in states])
    return states, raw / raw.sum()

def chi_squared_2d(observed, expected):
    mask = expected >= 5
    obs = np.append(observed[mask], observed[~mask].sum())
    exp = np.append(expected[mask], expected[~mask].sum())
    return stats.chisquare(obs, f_exp=exp)

# Part 2(a) – Direct 2D random-walk M-H
def mh_2d_direct(m, n_samples, burn_in=5000, thin=10):
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    i, j = 2, 2
    samples = []
    total_steps = burn_in + n_samples * thin

    for step in range(total_steps):
        di, dj = moves[np.random.randint(4)]
        ni, nj = i + di, j + dj

        if ni >= 0 and nj >= 0 and ni + nj <= m:
            if   di ==  1: ratio = A1 / ni   # A1 / (i+1)
            elif di == -1: ratio = i  / A1   # i  / A1
            elif dj ==  1: ratio = A2 / nj   # A2 / (j+1)
            else:          ratio = j  / A2   # j  / A2

            if np.random.uniform() < min(1.0, ratio):
                i, j = ni, nj

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append((i, j))

    return samples

np.random.seed(42)
n_samples_2 = 20_000
samples_2a = mh_2d_direct(m, n_samples_2)

states_2d, probs_2d = theoretical_probs_2d(m)
state_idx = {s: k for k, s in enumerate(states_2d)}
observed_2a = np.zeros(len(states_2d))
for s in samples_2a:
    observed_2a[state_idx[s]] += 1
expected_2a = n_samples_2 * probs_2d

chi2_2a, p_2a = chi_squared_2d(observed_2a, expected_2a)
print(f"\n--- Part 2(a): Direct M-H ---")
print(f"Chi-squared: {chi2_2a:.4f}, p-value: {p_2a:.4f}")
print(f"Reject H0 (α=0.05): {p_2a < 0.05}")

# Part 2(b) – Coordinate-wise M-H
def mh_2d_coord(m, n_samples, burn_in=5000, thin=10):
    i, j = 2, 2
    samples = []
    total_steps = burn_in + n_samples * thin

    for step in range(total_steps):
        if step % 2 == 0:  # update i
            ni = i + np.random.choice([-1, 1])
            if 0 <= ni and ni + j <= m:
                ratio = A1 / ni if ni > i else i / A1
                if np.random.uniform() < min(1.0, ratio):
                    i = ni
        else:              # update j
            nj = j + np.random.choice([-1, 1])
            if 0 <= nj and i + nj <= m:
                ratio = A2 / nj if nj > j else j / A2
                if np.random.uniform() < min(1.0, ratio):
                    j = nj

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append((i, j))

    return samples

samples_2b = mh_2d_coord(m, n_samples_2)

observed_2b = np.zeros(len(states_2d))
for s in samples_2b:
    observed_2b[state_idx[s]] += 1
expected_2b = n_samples_2 * probs_2d

chi2_2b, p_2b = chi_squared_2d(observed_2b, expected_2b)
print(f"\n--- Part 2(b): Coordinate-wise M-H ---")
print(f"Chi-squared: {chi2_2b:.4f}, p-value: {p_2b:.4f}")
print(f"Reject H0 (α=0.05): {p_2b < 0.05}")

# Part 2(c) – Gibbs sampling with exact conditionals
# P(i|j) ∝ A1^i/i! for i=0,...,m-j  (truncated Poisson(A1))
# P(j|i) ∝ A2^j/j! for j=0,...,m-i  (truncated Poisson(A2))
def sample_truncated_poisson(A, max_val):
    vals = np.arange(max_val + 1)
    raw = np.array([A**k / factorial(k) for k in vals])
    return np.random.choice(vals, p=raw / raw.sum())

def gibbs_2d(m, n_samples, burn_in=5000, thin=10):
    i, j = 2, 2
    samples = []
    total_steps = burn_in + n_samples * thin

    for step in range(total_steps):
        if step % 2 == 0:
            i = sample_truncated_poisson(A1, m - j)
        else:
            j = sample_truncated_poisson(A2, m - i)

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append((i, j))

    return samples

samples_2c = gibbs_2d(m, n_samples_2)

observed_2c = np.zeros(len(states_2d))
for s in samples_2c:
    observed_2c[state_idx[s]] += 1
expected_2c = n_samples_2 * probs_2d

chi2_2c, p_2c = chi_squared_2d(observed_2c, expected_2c)
print(f"\n--- Part 2(c): Gibbs sampling ---")
print(f"Chi-squared: {chi2_2c:.4f}, p-value: {p_2c:.4f}")
print(f"Reject H0 (α=0.05): {p_2c < 0.05}")

# Part 3
rho = 0.5
cov = np.array([[1, rho], [rho, 1]])

# Part 3(a) – sample (theta, psi) from the prior
np.random.seed(0)
xi, gamma = np.random.multivariate_normal([0, 0], cov)
theta_true, psi_true = np.exp(xi), np.exp(gamma)
print(f"\n--- Part 3(a): Prior sample ---")
print(f"theta = {theta_true:.4f}, psi = {psi_true:.4f}")

# Part 3(b) – generate observations Xi ~ N(theta, psi), n=10
n = 10
X = np.random.normal(theta_true, np.sqrt(psi_true), size=n)
print(f"\n--- Part 3(b): Observations (n={n}) ---")
print(f"X = {np.round(X, 4)}")

# Part 3(c) – log-posterior up to proportionality
# log pi(theta,psi|x) ∝ log L + log f(theta,psi)
# We work in log-space u=log(theta), v=log(psi) for the M-H proposal.
# The Jacobian +u+v cancels with -log(theta)-log(psi) from the prior.
def log_posterior_uv(u, v, X):
    n = len(X)
    theta, psi = np.exp(u), np.exp(v)
    log_prior = -(u**2 - 2*rho*u*v + v**2) / (2*(1 - rho**2))
    log_lik   = -(n/2)*v - np.sum((X - theta)**2) / (2*psi)
    return log_prior + log_lik

# Part 3(d) – M-H sampler for posterior of (theta, psi), n=10
def mh_posterior(X, n_samples=20_000, burn_in=5000, thin=10, sigma=0.5):
    u, v = 0.0, 0.0  # start at theta=1, psi=1 (prior mean in log-space)
    samples = []
    total_steps = burn_in + n_samples * thin

    for step in range(total_steps):
        u_prop = u + np.random.normal(0, sigma)
        v_prop = v + np.random.normal(0, sigma)

        log_ratio = log_posterior_uv(u_prop, v_prop, X) - log_posterior_uv(u, v, X)
        if np.log(np.random.uniform()) < log_ratio:
            u, v = u_prop, v_prop

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append((np.exp(u), np.exp(v)))

    return np.array(samples)

samples_3d = mh_posterior(X)
theta_est, psi_est = samples_3d[:, 0].mean(), samples_3d[:, 1].mean()
print(f"\n--- Part 3(d): Posterior M-H (n={n}) ---")
print(f"True:      theta={theta_true:.4f}, psi={psi_true:.4f}")
print(f"Posterior mean: theta={theta_est:.4f}, psi={psi_est:.4f}")

# Part 3(e) – repeat for n=100, 1000, 10000
print(f"\n--- Part 3(e) ---")
print(f"True: theta={theta_true:.4f}, psi={psi_true:.4f}")
print(f"{'n':>6}  {'θ mean':>8}  {'θ 95% CI':>19}  {'width':>6}  {'ψ mean':>8}  {'ψ 95% CI':>19}  {'width':>6}")
for n_e in [100, 1000, 10000]:
    X_e = np.random.normal(theta_true, np.sqrt(psi_true), size=n_e)
    s = mh_posterior(X_e)
    t_lo, t_hi = np.percentile(s[:, 0], [2.5, 97.5])
    p_lo, p_hi = np.percentile(s[:, 1], [2.5, 97.5])
    t_ci = f"[{t_lo:.4f}, {t_hi:.4f}]"
    p_ci = f"[{p_lo:.4f}, {p_hi:.4f}]"
    print(f"{n_e:>6}  {s[:,0].mean():>8.4f}  {t_ci:>19}  {t_hi-t_lo:>6.4f}  {s[:,1].mean():>8.4f}  {p_ci:>19}  {p_hi-p_lo:>6.4f}")
