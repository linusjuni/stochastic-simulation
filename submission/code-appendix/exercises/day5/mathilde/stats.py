from __future__ import annotations

import numpy as np
import scipy.stats as st

def pool_low_expected(observed: np.ndarray, expected: np.ndarray, min_expected: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Sweep bins left-to-right and merge any with expected count below min_expected."""
    pooled_obs = []
    pooled_exp = []
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

def chi_square_gof(observed_counts: np.ndarray, expected_probs: np.ndarray, min_expected: float = 5.0) -> tuple[float, float, int]:
    """
    Perform a Chi-Square goodness-of-fit test with automatic pooling of bins 
    where expected counts < min_expected to ensure test validity.
    
    Returns (chi2_statistic, p_value, degrees_of_freedom).
    """
    n = observed_counts.sum()
    expected_counts = expected_probs * n
    
    obs_flat = observed_counts.flatten()
    exp_flat = expected_counts.flatten()
    
    pooled_obs, pooled_exp = pool_low_expected(obs_flat, exp_flat, min_expected)
    res = st.chisquare(f_obs=pooled_obs, f_exp=pooled_exp)
    
    # DoF = number of pooled bins - 1
    dof = len(pooled_obs) - 1
    return float(res.statistic), float(res.pvalue), dof
