# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file. 
# They have been used for code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, kstest, norm

def lcg(x_initial, n=10000, a=42, c=69, M=67):
    x = x_initial
    xs = [x]
    seen = set()
    for _ in range(n - 1):
        x = (a * x + c) % M
        seen.add(x)
        xs.append(x)
    return np.array(xs), len(seen)

def run_tests(us, label=""):
    us = np.array(us, dtype=float)
    n = len(us)
    print(f"\n=== {label} ===")

    # Chi-squared test
    k = 10
    observed = np.histogram(us, bins=k, range=(0, 1))[0]
    chi2_stat = np.sum((observed - n / k) ** 2 / (n / k))
    chi2_p = 1 - chi2.cdf(chi2_stat, df=k - 1)
    print(f"Chi-squared: stat={chi2_stat:.4f}, p={chi2_p:.4f} -> {'REJECT' if chi2_p < 0.05 else 'ACCEPT'}")

    # KS test
    ks_stat, ks_p = kstest(us, 'uniform')
    print(f"KS test:     stat={ks_stat:.4f}, p={ks_p:.4f} -> {'REJECT' if ks_p < 0.05 else 'ACCEPT'}")

    # Runs test (above/below median, Wald-Wolfowitz)
    binary = (us >= np.median(us)).astype(int)
    n1, n2 = int(binary.sum()), int(n - binary.sum())
    runs = 1 + int(np.sum(binary[1:] != binary[:-1]))
    mu = 1 + 2 * n1 * n2 / (n1 + n2)
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mu) / var ** 0.5
    runs_p = 2 * (1 - norm.cdf(abs(z)))
    print(f"Runs test:   z={z:.4f}, p={runs_p:.4f} -> {'REJECT' if runs_p < 0.05 else 'ACCEPT'}")

    # Correlation test: c_h ~ Normal(0.25, 7/(144n))  [slide 24]
    print("Correlation test:")
    se = np.sqrt(7 / (144 * n))
    for h in [1, 2, 5, 10, 20]:
        c_h = np.mean(us[:n - h] * us[h:])
        z = (c_h - 0.25) / se
        p = 2 * (1 - norm.cdf(abs(z)))
        print(f"  h={h}: c_h={c_h:.4f}, z={z:.4f}, p={p:.4f} -> {'REJECT' if p < 0.05 else 'ACCEPT'}")

# Bad LCG: a=42, c=69, M=67 gives a period of only 67 -- clearly a bad generator
xs_bad, period_bad = lcg(x_initial=4, a=42, c=69, M=67)
us_bad = xs_bad / 67

print(f"Bad LCG period: {period_bad}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Bad LCG (a=42, c=69, M=67) -- period=67")
axes[0].hist(us_bad, bins=10, range=(0, 1), edgecolor='black')
axes[0].set_title("Histogram (10 classes)")
axes[1].scatter(us_bad[:-1], us_bad[1:], s=1, alpha=0.5)
axes[1].set_title("Scatter plot ($u_i$ vs $u_{i+1}$)")
plt.tight_layout()
plt.show()

# Good LCG: GNU C parameters (a=1103515245, c=12345, M=2^31)
M_good = 2**31
xs_good, period_good = lcg(x_initial=4, a=1103515245, c=12345, M=M_good)
us_good = xs_good / M_good

print(f"Good LCG period: {period_good}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Good LCG (a=1103515245, c=12345, M=2^31)")
axes[0].hist(us_good, bins=10, range=(0, 1), edgecolor='black')
axes[0].set_title("Histogram (10 classes)")
axes[1].scatter(us_good[:-1], us_good[1:], s=1, alpha=0.5)
axes[1].set_title("Scatter plot ($u_i$ vs $u_{i+1}$)")
plt.tight_layout()
plt.show()

# Part 1(b): run tests on both to compare
run_tests(us_bad,  label="Bad LCG  (a=42, c=69, M=67)")
run_tests(us_good, label="Good LCG (a=1103515245, c=12345, M=2^31)")

# Part 2: System generator (Mersenne Twister)
us_sys = np.random.uniform(size=10000)
run_tests(us_sys, label="System generator (np.random.uniform, Mersenne Twister)")
