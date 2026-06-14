# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

import numpy as np
from math import factorial
from scipy.stats import t as t_dist

m, n_customers, n_runs = 10, 10000, 10
service_fn = lambda: np.random.exponential(8.0)

def simulate(arrival_fn, service_fn, m, n_customers):
    server_free_at = np.zeros(m)
    t = 0.0
    blocked = 0
    for _ in range(n_customers):
        t += arrival_fn()
        if min(server_free_at) <= t:
            index = np.argmin(server_free_at)
            server_free_at[index] = t + service_fn()
        else:
            blocked += 1
    return blocked / n_customers

def erlang_b(A, m):
    numerator = A**m / factorial(m)
    denominator = sum(A**i / factorial(i) for i in range(m+1))
    return numerator / denominator

def confidence_interval(fractions):
    n = len(fractions)
    mean = np.mean(fractions)
    std = np.std(fractions, ddof=1)
    t_val = t_dist.ppf(0.975, df=n-1)
    return mean, mean - t_val * std / np.sqrt(n), mean + t_val * std / np.sqrt(n)

# Part 1 - Poisson arrivals, exponential service
arrival_fn = lambda: np.random.exponential(1.0)
fractions = [simulate(arrival_fn, service_fn, m, n_customers) for _ in range(n_runs)]
mean, lo, hi = confidence_interval(fractions)
print("-- Part 1: Poisson arrivals, Exp service --")
print(f"Simulated:  {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
print(f"Erlang B:   {erlang_b(A=8, m=10):.4f}")

# Part 2 - Renewal process arrivals
def erlang_arrival(k):
    return np.sum(np.random.exponential(scale=1/k, size=k))

def hyperexp_arrival(p1, lam1, lam2):
    if np.random.random() < p1:
        return np.random.exponential(scale=1/lam1)
    else:
        return np.random.exponential(scale=1/lam2)

print("\n-- Part 2a: Erlang arrivals --")
for k in [2, 5]:
    fractions = [simulate(lambda: erlang_arrival(k), service_fn, m, n_customers) for _ in range(n_runs)]
    mean, lo, hi = confidence_interval(fractions)
    print(f"Erlang k={k}:  {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

print("\n-- Part 2b: Hyper-exponential arrivals --")
fractions = [simulate(lambda: hyperexp_arrival(0.8, 0.8333, 5.0), service_fn, m, n_customers) for _ in range(n_runs)]
mean, lo, hi = confidence_interval(fractions)
print(f"Hyper-exp:  {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

# Part 3
def constant_service(k=8):
    return k
def pareto_service(k, mean=8):
    return np.random.pareto(k) * mean*(k-1)
def normal_service(mean=8):
    return np.random.normal(mean)
def randomuniform_service(mean):
    return np.random.uniform(0,16)

print("\n-- Part 3: Poisson arrivals, varying service distributions --")
fractions = [simulate(arrival_fn, constant_service, m, n_customers) for _ in range(n_runs)]
mean, lo, hi = confidence_interval(fractions)
print(f"Constant:      {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

for k in [1.05, 2.05, 3.00]:
    fractions = [simulate(arrival_fn, lambda: pareto_service(k), m, n_customers) for _ in range(n_runs)]
    mean, lo, hi = confidence_interval(fractions)
    print(f"Pareto k={k:.2f}: {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

fractions = [simulate(arrival_fn, normal_service, m, n_customers) for _ in range(n_runs)]
mean, lo, hi = confidence_interval(fractions)
print(f"Normal:        {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

fractions = [simulate(arrival_fn, lambda: randomuniform_service(8), m, n_customers) for _ in range(n_runs)]
mean, lo, hi = confidence_interval(fractions)
print(f"Uniform(0,16): {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

# Part 4 - Summary comparison
erlang_b_ref = erlang_b(A=8, m=10)
results = {}

arrival_fn = lambda: np.random.exponential(1.0)
service_fn = lambda: np.random.exponential(8.0)

results["Part 1: Poisson+Exp"]       = confidence_interval([simulate(arrival_fn, service_fn, m, n_customers) for _ in range(n_runs)])
results["Part 2a: Erlang k=2"]       = confidence_interval([simulate(lambda: erlang_arrival(2), service_fn, m, n_customers) for _ in range(n_runs)])
results["Part 2a: Erlang k=5"]       = confidence_interval([simulate(lambda: erlang_arrival(5), service_fn, m, n_customers) for _ in range(n_runs)])
results["Part 2b: Hyper-exp"]        = confidence_interval([simulate(lambda: hyperexp_arrival(0.8, 0.8333, 5.0), service_fn, m, n_customers) for _ in range(n_runs)])
results["Part 3a: Constant"]         = confidence_interval([simulate(arrival_fn, constant_service, m, n_customers) for _ in range(n_runs)])
results["Part 3b: Pareto k=1.05"]    = confidence_interval([simulate(arrival_fn, lambda: pareto_service(1.05), m, n_customers) for _ in range(n_runs)])
results["Part 3b: Pareto k=2.05"]    = confidence_interval([simulate(arrival_fn, lambda: pareto_service(2.05), m, n_customers) for _ in range(n_runs)])
results["Part 3c: Normal"]           = confidence_interval([simulate(arrival_fn, normal_service, m, n_customers) for _ in range(n_runs)])
results["Part 3c: Uniform(0,16)"]    = confidence_interval([simulate(arrival_fn, lambda: randomuniform_service(8), m, n_customers) for _ in range(n_runs)])

print("\n" + "="*65)
print(f"{'SUMMARY — Erlang B reference: ' + f'{erlang_b_ref:.4f}':^65}")
print("="*65)
print(f"{'Experiment':<30} {'Mean':>8}  {'95% CI':>22}  {'Contains B?':>11}")
print("-"*65)
for label, (mean, lo, hi) in results.items():
    contains = "YES" if lo <= erlang_b_ref <= hi else "NO"
    print(f"{label:<30} {mean:>8.4f}  [{lo:.4f}, {hi:.4f}]  {contains:>11}")
print("="*65)