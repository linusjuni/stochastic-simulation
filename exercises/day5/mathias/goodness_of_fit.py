import numpy as np
from math import factorial
from scipy.stats import chisquare

from exercises.day5.mathias.metropolis_hastings import (
    metropolis_hastings_part1,
    metropolis_hastings_part2,
    gibbs_sampling_part2,
)


def _truncated_pmf_1d(A, m):
    weights = np.array([(A ** i) / factorial(i) for i in range(m + 1)], dtype=float)
    return weights / weights.sum()


def _truncated_joint_pmf_2d(A1, A2, m):
    pmf = np.zeros((m + 1, m + 1), dtype=float)
    for i in range(m + 1):
        for j in range(m + 1 - i):
            pmf[i, j] = (A1 ** i / factorial(i)) * (A2 ** j / factorial(j))
    return pmf / pmf.sum()


def _pool_low_expected(observed, expected, min_expected=5):
    """Sweep bins left-to-right and merge any with expected count below
    ``min_expected`` into a running accumulator. Tail leftover is folded into
    the previous bin so every emitted bin has expected >= min_expected
    (except possibly the case of a single bin)."""
    pooled_obs, pooled_exp = [], []
    acc_o, acc_e = 0.0, 0.0
    for o, e in zip(observed, expected):
        acc_o += o
        acc_e += e
        if acc_e >= min_expected:
            pooled_obs.append(acc_o)
            pooled_exp.append(acc_e)
            acc_o, acc_e = 0.0, 0.0
    if acc_e > 0:
        if pooled_exp:
            pooled_obs[-1] += acc_o
            pooled_exp[-1] += acc_e
        else:
            pooled_obs.append(acc_o)
            pooled_exp.append(acc_e)
    return np.array(pooled_obs), np.array(pooled_exp)


def chi2_goodness_of_fit(samples, A=None, A1=None, A2=None, m=10, min_expected=5):
    """Chi-squared goodness-of-fit test for MCMC samples.

    Auto-detects 1D (Part 1) vs 2D (Part 2) input:
      - 1D ints: test against pmf proportional to A^i / i! on {0, ..., m}.
      - 2D tuples: test the joint against pmf proportional to
        (A1^i / i!)(A2^j / j!) on {(i, j) : i + j <= m}.

    Adjacent bins with expected count below ``min_expected`` are pooled so
    the chi-squared approximation stays valid.

    Returns dict with keys: statistic, p_value, dof, n_bins, n_samples.
    """
    samples = list(samples)
    n = len(samples)
    is_2d = isinstance(samples[0], tuple)

    if is_2d:
        if A1 is None or A2 is None:
            raise ValueError("Provide A1 and A2 for 2D samples")
        pmf = _truncated_joint_pmf_2d(A1, A2, m)
        observed = np.zeros_like(pmf)
        for i, j in samples:
            observed[i, j] += 1
        mask = pmf > 0
        obs_flat = observed[mask]
        exp_flat = n * pmf[mask]
    else:
        if A is None:
            raise ValueError("Provide A for 1D samples")
        pmf = _truncated_pmf_1d(A, m)
        observed = np.bincount(samples, minlength=m + 1).astype(float)
        obs_flat = observed
        exp_flat = n * pmf

    pooled_obs, pooled_exp = _pool_low_expected(obs_flat, exp_flat, min_expected)
    result = chisquare(f_obs=pooled_obs, f_exp=pooled_exp)

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "dof": int(len(pooled_obs) - 1),
        "n_bins": int(len(pooled_obs)),
        "n_samples": int(n),
    }


def report_chi2(title, result, alpha=0.05):
    decision = "FAIL to reject H0" if result["p_value"] >= alpha else "REJECT H0"
    print("=" * 60)
    print(title)
    print("-" * 60)
    print(f"Samples:               {result['n_samples']}")
    print(f"Bins (after pooling):  {result['n_bins']}")
    print(f"Degrees of freedom:    {result['dof']}")
    print(f"Chi-squared statistic: {result['statistic']:.4f}")
    print(f"p-value:               {result['p_value']:.4g}")
    print(f"Decision (alpha={alpha}): {decision}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    np.random.seed(42)

    # Part 1: 1D Metropolis-Hastings, truncated dist. proportional to 8^i / i!
    samples_p1 = metropolis_hastings_part1(n_samples=100000, initial_state=0)
    res_p1 = chi2_goodness_of_fit(samples_p1, A=8, m=10)
    report_chi2("Part 1: Metropolis-Hastings (A=8, m=10)", res_p1)

    # Part 2: 2D Metropolis-Hastings on joint with i+j <= 10
    samples_p2_mh = metropolis_hastings_part2(
        n_samples=int(1e6), init_state_1=0, init_state_2=0
    )
    res_p2_mh = chi2_goodness_of_fit(samples_p2_mh, A1=4, A2=4, m=10)
    report_chi2("Part 2: Metropolis-Hastings joint (A1=A2=4, m=10)", res_p2_mh)

    # Part 2: Gibbs sampling on the same joint
    samples_p2_gibbs = gibbs_sampling_part2(
        n_samples=int(1e6), init_state_1=0, init_state_2=0
    )
    res_p2_gibbs = chi2_goodness_of_fit(samples_p2_gibbs, A1=4, A2=4, m=10)
    report_chi2("Part 2: Gibbs sampling joint (A1=A2=4, m=10)", res_p2_gibbs)
