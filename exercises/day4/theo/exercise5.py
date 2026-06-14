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
print(f"Part 1 - Crude MC\nmean = {mean:.6f}, 95% CI = ({ci[0]:.6f}, {ci[1]:.6f})")
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
print(f"Part 2 - antithetic MC\nmean = {mean:.6f}, 95% CI = ({ci[0]:.6f}, {ci[1]:.6f})")
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
print(f"Part 3 - Control MC\nmean = {mean:.6f}, 95% CI = ({ci[0]:.6f}, {ci[1]:.6f})")
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
print(f"Part 4 - Stratified MC\nmean = {mean:.6f}, 95% CI = ({ci[0]:.6f}, {ci[1]:.6f})")
print(f"  True value: {np.e - 1:.6f}")

# Part 5
# Modified simulate that also returns sum of inter-arrival times
def simulate_cv(arrival_fn, service_fn, m, n):
    server_free_at = np.zeros(m)
    t = 0.0
    blocked = 0
    sum_interarrivals = 0.0
    for _ in range(n):
        dt = arrival_fn()
        sum_interarrivals += dt
        t += dt
        if min(server_free_at) <= t:
            index = np.argmin(server_free_at)
            server_free_at[index] = t + service_fn()
        else:
            blocked += 1
    return blocked / n, sum_interarrivals

# First we do 100 runs to calculate c
Bs, Ss = [], []
n_customers = 10000
for i in range(100):
    arrival_fn = lambda: np.random.exponential(1.0)
    service_fn = lambda: np.random.exponential(8.0)
    B_i, S_i = simulate_cv(arrival_fn=arrival_fn, service_fn=service_fn, m=10, n=n_customers)
    Bs.append(B_i)
    Ss.append(S_i)
c = np.cov(Bs, Ss)[0,1] / np.var(Ss, ddof=1)

Bs, Ss, Ys = [], [], []
n = 10000
n_reps = 100
for i in range(n_reps):
    arrival_fn = lambda: np.random.exponential(1.0)
    service_fn = lambda: np.random.exponential(8.0)
    B_i, S_i = simulate_cv(arrival_fn=arrival_fn, service_fn=service_fn, m=10, n=n)
    Bs.append(B_i)
    Ss.append(S_i)
    Ys.append(B_i - c * (S_i - n))
mean = np.mean(Ys)
std = np.std(Ys, ddof=1)
half_width = scipy.stats.t.ppf(0.975, df=n_reps-1) * std / np.sqrt(n_reps)

from math import factorial
def erlang_b(A, m):
    num = A**m / factorial(m)
    return num / sum(A**i / factorial(i) for i in range(m+1))

crude_mean = np.mean(Bs)
crude_std = np.std(Bs, ddof=1)
crude_hw = scipy.stats.t.ppf(0.975, df=n_reps-1) * crude_std / np.sqrt(n_reps)

print('-'*34)
print(f"Part 5 - Blocking system variance reduction (n={n_customers} customers, {n_reps} reps)")
print(f"  Erlang B reference:  {erlang_b(8, 10):.6f}")
print(f"  Crude (no CV):       mean = {crude_mean:.6f}, 95% CI = ({crude_mean - crude_hw:.6f}, {crude_mean + crude_hw:.6f}), width = {2*crude_hw:.6f}")
print(f"  Control variates:    mean = {mean:.6f}, 95% CI = ({mean - half_width:.6f}, {mean + half_width:.6f}), width = {2*half_width:.6f}")

# Part 6 has been removed from pensum this year

# Part 7
def crude_tail(n, a):
    X = np.random.normal(size=n)
    samples = (X > a).astype(float)
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    hw = scipy.stats.t.ppf(0.975, df=n-1) * std / np.sqrt(n)
    return mean, (mean - hw, mean + hw)

def importance(n, a, sigma2=1):
    X = np.random.normal(loc=a, scale=np.sqrt(sigma2), size=n)
    w = scipy.stats.norm.pdf(X, 0, 1) / scipy.stats.norm.pdf(X, a, np.sqrt(sigma2))
    samples = (X > a) * w
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    hw = scipy.stats.t.ppf(0.975, df=n-1) * std / np.sqrt(n)
    return mean, (mean - hw, mean + hw)

print('-'*34)
print("Part 7 - Importance sampling for P(Z > a)")
for n in [100, 1000, 10000]:
    print(f"  n={n}")
    for a in [2, 4]:
        true_p = scipy.stats.norm.sf(a)
        crude_mean, crude_ci = crude_tail(n=n, a=a)
        print(f"    a={a}, true P(Z>a) = {true_p:.2e}")
        print(f"      Crude:        mean = {crude_mean:.2e}, CI width = {crude_ci[1]-crude_ci[0]:.2e}")
        for sigma2 in [1, 2, 3]:
            is_mean, is_ci = importance(n=n, a=a, sigma2=sigma2)
            print(f"      IS (Var={sigma2}):    mean = {is_mean:.2e}, CI width = {is_ci[1]-is_ci[0]:.2e}")

# Part 8
var_W = lambda lam: (np.e**(2+lam) - 1) / (lam * (2+lam)) - (np.e-1)**2
res = scipy.optimize.minimize_scalar(var_W, bounds=(0.01, 20), method='bounded')
lam_opt = res.x
print('-'*34)
print(f"Part 8 - IS with Exp(lambda) for integral_0^1 e^x dx")
print(f"  Optimal lambda: {lam_opt:.4f}")
print(f"  Var(W) at lambda*: {res.fun:.4f}")

def importance2(n, lam):
    X = np.random.exponential(1/lam, size=n)
    w = np.exp(X * (1 + lam)) / lam
    samples = (X <= 1) * w
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    hw = scipy.stats.t.ppf(0.975, df=n-1) * std / np.sqrt(n)
    return mean, (mean - hw, mean + hw)

n = 10000
crude_mean, crude_ci = crude(n)
print(f"  Crude MC:          mean = {crude_mean:.6f}, CI width = {crude_ci[1]-crude_ci[0]:.6f}")
for lam in [0.5, 1.0, lam_opt, 2.0, 5.0]:
    is_mean, is_ci = importance2(n, lam)
    print(f"  IS lambda={lam:.3f}:   mean = {is_mean:.6f}, CI width = {is_ci[1]-is_ci[0]:.6f}")
print(f"  True value: {np.e - 1:.6f}")

# Part 9
# This one is no-code, so just see the report