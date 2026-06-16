import numpy as np
import scipy.stats as st
from math import factorial

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)
rng = np.random.default_rng(settings.SEED)

PLOTS = "exercises/day5/linus/plots"

# Part 1
A1 = 8.0
M1 = 10

# Part 2
A2_1 = 4.0
A2_2 = 4.0
M2 = 10

# Part 3
RHO = 0.5


# ---------------------------------------------------------------------------
# Shared chi2 helpers — sweep pooling matches Mathias's _pool_low_expected
# ---------------------------------------------------------------------------

def _pool_low_expected(observed: list, expected: list, min_expected: float = 5.0):
    """Sweep left-to-right; accumulate until expected >= min_expected, then emit."""
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


def chi2_test_1d(samples: np.ndarray, states: np.ndarray, exact_pmf: np.ndarray) -> tuple[float, float, int]:
    n = len(samples)
    observed = [float(np.sum(samples == s)) for s in states]
    expected = list(exact_pmf * n)
    obs_p, exp_p = _pool_low_expected(observed, expected)
    chi2, p = st.chisquare(obs_p, exp_p)
    return float(chi2), float(p), len(obs_p) - 1


def chi2_test_2d(samples: np.ndarray, exact_pmf: dict) -> tuple[float, float, int]:
    n = len(samples)
    counts: dict[tuple, int] = {}
    for row in samples:
        k = (int(row[0]), int(row[1]))
        counts[k] = counts.get(k, 0) + 1
    # iterate states in natural (i, j) order
    states = sorted(exact_pmf.keys())
    observed = [float(counts.get(s, 0)) for s in states]
    expected = [exact_pmf[s] * n for s in states]
    obs_p, exp_p = _pool_low_expected(observed, expected)
    chi2, p = st.chisquare(obs_p, exp_p)
    return float(chi2), float(p), len(obs_p) - 1


# ---------------------------------------------------------------------------
# Part 1: Truncated Poisson via Metropolis-Hastings
# ---------------------------------------------------------------------------

def g1(i: int) -> float:
    return A1**i / factorial(i)


def mh_truncated_poisson(n_samples: int = 100_000, thinning: int = 10) -> list:
    samples = [0]
    for _ in range(n_samples):
        current = samples[-1]
        proposed = current + rng.choice([-1, 1])
        if proposed < 0 or proposed > M1:
            samples.append(current)
            continue
        alpha = min(1, g1(proposed) / g1(current))
        if rng.random() < alpha:
            samples.append(proposed)
        else:
            samples.append(current)
    return samples[::thinning]


def part1() -> None:
    samples = mh_truncated_poisson()
    states = np.arange(M1 + 1)
    unnorm = np.array([g1(i) for i in states])
    exact = unnorm / unnorm.sum()

    chi2, p, df = chi2_test_1d(np.array(samples), states, exact)
    logger.info("Part 1 chi2", chi2=round(chi2, 3), p=round(p, 3), df=df)

    observed_freq = np.array([np.mean(np.array(samples) == s) for s in states])
    with figure(figsize=(8, 4), save=f"{PLOTS}/metropolis_hastings_samples.png") as fig:
        ax = fig.add_subplot(111)
        ax.bar(states - 0.2, observed_freq, width=0.38, label="MH samples", color="steelblue")
        ax.bar(states + 0.2, exact, width=0.38, label="Exact", color="tomato", alpha=0.8)
        ax.set_xlabel("State i")
        ax.set_ylabel("Probability")
        ax.set_title(f"Truncated Poisson MH  (A={A1}, m={M1})\nchi2={chi2:.3f}, df={df}, p={p:.3f}")
        ax.legend()


# ---------------------------------------------------------------------------
# Part 2: Joint truncated distribution
# ---------------------------------------------------------------------------

def g2(i: int, j: int) -> float:
    if i < 0 or j < 0 or i + j > M2:
        return 0.0
    return (A2_1**i / factorial(i)) * (A2_2**j / factorial(j))


def _exact_joint_pmf() -> dict[tuple[int, int], float]:
    unnorm = {(i, j): g2(i, j) for i in range(M2 + 1) for j in range(M2 + 1 - i)}
    total = sum(unnorm.values())
    return {k: v / total for k, v in unnorm.items()}


def _plot_sum_dist(samples: np.ndarray, exact_pmf: dict, title: str, save: str) -> None:
    n = len(samples)
    sums = samples[:, 0] + samples[:, 1]
    exact_sum = np.zeros(M2 + 1)
    for (i, j), p in exact_pmf.items():
        exact_sum[i + j] += p
    empirical = np.array([np.sum(sums == k) / n for k in range(M2 + 1)])
    ks = np.arange(M2 + 1)
    with figure(figsize=(8, 4), save=save) as fig:
        ax = fig.add_subplot(111)
        ax.bar(ks - 0.2, empirical, width=0.38, label="Samples", color="steelblue")
        ax.bar(ks + 0.2, exact_sum, width=0.38, label="Exact", color="tomato", alpha=0.8)
        ax.set_xlabel("i + j")
        ax.set_ylabel("Probability")
        ax.set_title(title)
        ax.legend()


def mh_joint_direct(n_samples: int = 1_000_000, thinning: int = 10) -> list:
    """Part 2a: joint proposal perturbing both coords simultaneously — mixes poorly."""
    samples = [(0, 0)]
    for _ in range(n_samples):
        ci, cj = samples[-1]
        pi = ci + rng.choice([-1, 1])
        pj = cj + rng.choice([-1, 1])
        gp = g2(pi, pj)
        gc = g2(ci, cj)
        if gp > 0 and rng.random() < min(1.0, gp / gc):
            samples.append((pi, pj))
        else:
            samples.append((ci, cj))
    return samples[::thinning]


def mh_joint_coordinatewise(n_samples: int = 1_000_000, thinning: int = 10) -> list:
    """Part 2b: update one coordinate at a time."""
    samples = [(0, 0)]
    for _ in range(n_samples):
        ci, cj = samples[-1]
        p = rng.random()
        if p < 0.5:
            pi, pj = ci + rng.choice([-1, 1]), cj
        else:
            pi, pj = ci, cj + rng.choice([-1, 1])
        if pi < 0 or pi > M2 or pj < 0 or pj > M2 or pi + pj > M2:
            samples.append((ci, cj))
            continue
        alpha = min(1, g2(pi, pj) / g2(ci, cj))
        if rng.random() < alpha:
            samples.append((pi, pj))
        else:
            samples.append((ci, cj))
    return samples[::thinning]


def gibbs_joint(n_samples: int = 1_000_000, thinning: int = 10) -> list:
    """Part 2c: exact conditional sampling — no accept/reject."""
    samples = [(0, 0)]
    for _ in range(n_samples):
        ci, cj = samples[-1]
        # sample i | j
        sup_i = np.arange(M2 - cj + 1)
        w_i = np.array([A2_1**k / factorial(k) for k in sup_i])
        new_i = int(rng.choice(sup_i, p=w_i / w_i.sum()))
        # sample j | new_i
        sup_j = np.arange(M2 - new_i + 1)
        w_j = np.array([A2_2**k / factorial(k) for k in sup_j])
        new_j = int(rng.choice(sup_j, p=w_j / w_j.sum()))
        samples.append((new_i, new_j))
    return samples[::thinning]


def part2() -> None:
    exact_pmf = _exact_joint_pmf()

    # (a) direct joint MH — both coords perturbed simultaneously
    s_a_raw = mh_joint_direct()
    s_a = np.array(s_a_raw)
    chi2_a, p_a, df_a = chi2_test_2d(s_a, exact_pmf)
    logger.info("Part 2a direct MH", chi2=round(chi2_a, 3), p=round(p_a, 3), df=df_a)

    # (b) coordinate-wise MH
    s_b_raw = mh_joint_coordinatewise()
    s_b = np.array(s_b_raw)
    chi2_b, p_b, df_b = chi2_test_2d(s_b, exact_pmf)
    logger.info("Part 2b coord-wise MH", chi2=round(chi2_b, 3), p=round(p_b, 3), df=df_b)
    _plot_sum_dist(
        s_b,
        exact_pmf,
        f"Coord-wise MH  (A1=A2={A2_1}, m={M2})\nchi2={chi2_b:.2f}, df={df_b}, p={p_b:.3f}",
        f"{PLOTS}/metropolis_hastings_samples_part2.png",
    )

    # (c) Gibbs
    s_c_raw = gibbs_joint()
    s_c = np.array(s_c_raw)
    chi2_c, p_c, df_c = chi2_test_2d(s_c, exact_pmf)
    logger.info("Part 2c Gibbs", chi2=round(chi2_c, 3), p=round(p_c, 3), df=df_c)
    _plot_sum_dist(
        s_c,
        exact_pmf,
        f"Gibbs sampling  (A1=A2={A2_1}, m={M2})\nchi2={chi2_c:.2f}, df={df_c}, p={p_c:.3f}",
        f"{PLOTS}/gibbs_samples_part2.png",
    )


# ---------------------------------------------------------------------------
# Part 3: Bayesian inference with log-normal prior
# ---------------------------------------------------------------------------

def _posterior_naive(theta: float, psi: float, obs: np.ndarray) -> float:
    """Full posterior density including (2pi*psi)^{-n/2} — underflows to 0 for n~1000.
    Uses np.float64 so overflow returns inf (not OverflowError). When both current and
    proposed posteriors are 0, acceptance ratio = nan, Python's min(1, nan) = 1, chain
    always accepts and wanders."""
    if theta <= 0 or psi <= 0:
        return 0.0
    n = len(obs)
    t, p = np.float64(theta), np.float64(psi)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        ll_const = (np.float64(2) * np.pi * p) ** np.float64(-n / 2)
        ll_exp = np.exp(-np.sum((obs - t) ** 2) / (np.float64(2) * p))
        prior_const = np.float64(1) / (np.float64(2) * np.pi * t * p * np.sqrt(np.float64(1) - RHO**2))
        prior_exp = np.exp(
            -(np.log(t) ** 2 - np.float64(2) * RHO * np.log(t) * np.log(p) + np.log(p) ** 2)
            / (np.float64(2) * (np.float64(1) - RHO**2))
        )
    return ll_const * ll_exp * prior_const * prior_exp  # np.float64: 0/0 → nan, not ZeroDivisionError


def log_posterior(theta: float, psi: float, obs: np.ndarray) -> float:
    """Log unnormalized posterior (drops constants that cancel in MH ratio)."""
    if theta <= 0 or psi <= 0:
        return -np.inf
    n = len(obs)
    ll = -n / 2 * np.log(psi) - np.sum((obs - theta) ** 2) / (2 * psi)
    lp = (
        -np.log(theta)
        - np.log(psi)
        - (np.log(theta) ** 2 - 2 * RHO * np.log(theta) * np.log(psi) + np.log(psi) ** 2)
        / (2 * (1 - RHO**2))
    )
    return ll + lp


def mh_posterior_naive(
    obs: np.ndarray,
    init_theta: float,
    init_psi: float,
    n_samples: int = 100_000,
    sigma: float = 1.0,
    thinning: int = 10,
) -> list:
    """Naive MH: computes actual density — underflows at n~1000 (0/0 → nan → wanders)."""
    samples = [(init_theta, init_psi)]
    for _ in range(n_samples):
        ct, cp = samples[-1]
        delta = rng.normal(0, sigma)
        if rng.random() < 0.5:
            pt, pp = ct + delta, cp
        else:
            pt, pp = ct, cp + delta
        if pt <= 0 or pp <= 0:
            samples.append((ct, cp))
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = _posterior_naive(pt, pp, obs) / _posterior_naive(ct, cp, obs)
        acceptance_ratio = min(1, ratio)  # min(1, nan)=1 → wanders; min(1, 0/0)=1 → wanders
        if rng.random() < acceptance_ratio:
            samples.append((pt, pp))
        else:
            samples.append((ct, cp))
    burn_in = int(0.1 * len(samples))
    return samples[burn_in::thinning]


def mh_posterior_log(
    obs: np.ndarray,
    init_theta: float,
    init_psi: float,
    n_samples: int = 100_000,
    sigma: float = 1.0,
    thinning: int = 10,
) -> list:
    """Log-stable MH: acceptance ratio via log-posterior difference."""
    samples = [(init_theta, init_psi)]
    lp_curr = log_posterior(init_theta, init_psi, obs)
    for _ in range(n_samples):
        ct, cp = samples[-1]
        delta = rng.normal(0, sigma)
        if rng.random() < 0.5:
            pt, pp = ct + delta, cp
        else:
            pt, pp = ct, cp + delta
        if pt <= 0 or pp <= 0:
            samples.append((ct, cp))
            continue
        lp_prop = log_posterior(pt, pp, obs)
        acceptance_ratio = min(1, np.exp(lp_prop - lp_curr))
        if rng.random() < acceptance_ratio:
            samples.append((pt, pp))
            lp_curr = lp_prop
        else:
            samples.append((ct, cp))
    burn_in = int(0.1 * len(samples))
    return samples[burn_in::thinning]


def _plot_posterior(samples: list, true_theta: float, true_psi: float, title: str, save: str) -> None:
    arr = np.array(samples)
    with figure(figsize=(6, 5), save=save) as fig:
        ax = fig.add_subplot(111)
        ax.scatter(arr[:, 0], arr[:, 1], alpha=0.15, s=3, color="steelblue", rasterized=True)
        ax.axvline(true_theta, color="tomato", linestyle="--", lw=1.5, label=f"true theta={true_theta:.3f}")
        ax.axhline(true_psi, color="darkorange", linestyle="--", lw=1.5, label=f"true psi={true_psi:.3f}")
        ax.set_xlabel("theta")
        ax.set_ylabel("psi")
        ax.set_title(title)
        ax.legend(fontsize=8)


def part3() -> None:
    # (a) sample from prior: (Xi, Gamma) ~ BivNorm(0,0; rho=0.5)
    n_prior = 100_000
    X = rng.normal(0, 1, size=n_prior)
    Z = rng.normal(0, 1, size=n_prior)
    Y = RHO * X + np.sqrt(1 - RHO**2) * Z
    thetas_prior = np.exp(X)
    psis_prior = np.exp(Y)

    with figure(figsize=(6, 5), save=f"{PLOTS}/log_prior_samples.png") as fig:
        ax = fig.add_subplot(111)
        ax.scatter(X[:1000], Y[:1000], alpha=0.3, s=5, color="steelblue")
        ax.set_xlabel("Xi = log theta")
        ax.set_ylabel("Gamma = log psi")
        ax.set_title(f"Prior samples in log-space  (rho={RHO})")

    with figure(figsize=(6, 5), save=f"{PLOTS}/prior_samples.png") as fig:
        ax = fig.add_subplot(111)
        ax.scatter(thetas_prior[:1000], psis_prior[:1000], alpha=0.3, s=5, color="steelblue")
        ax.set_xlabel("Theta = e^Xi")
        ax.set_ylabel("Psi = e^Gamma")
        ax.set_title("Prior samples (Theta, Psi)")

    # fix true (theta, psi) as first prior draw
    true_theta = float(thetas_prior[0])
    true_psi = float(psis_prior[0])
    logger.info("True parameters", theta=round(true_theta, 4), psi=round(true_psi, 4))

    # (b) simulate n=10 observations
    obs_10 = rng.normal(true_theta, np.sqrt(true_psi), size=10)
    with figure(figsize=(7, 4), save=f"{PLOTS}/gaussian_samples_theta_psi.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(obs_10, bins=10, color="steelblue", edgecolor="white")
        ax.axvline(true_theta, color="tomato", linestyle="--", label=f"theta={true_theta:.3f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Count")
        ax.set_title(f"n=10 observations  N(theta={true_theta:.3f}, psi={true_psi:.3f})")
        ax.legend()

    n_mcmc = 100_000

    # (d) MH posterior n=10, sigma=1, log-stable
    s10 = mh_posterior_log(obs_10, true_theta, true_psi, n_mcmc, sigma=1.0)
    _plot_posterior(s10, true_theta, true_psi,
                    f"Posterior MH  n_obs=10, n_samples={n_mcmc}, sigma=1",
                    f"{PLOTS}/posterior_samples_xobs10.png")

    # (e) n=100, naive (works fine — (2pi*psi)^{-50} is representable)
    obs_100 = rng.normal(true_theta, np.sqrt(true_psi), size=100)
    s100 = mh_posterior_naive(obs_100, true_theta, true_psi, n_mcmc, sigma=1.0)
    _plot_posterior(s100, true_theta, true_psi,
                    f"Posterior MH  n_obs=100, n_samples={n_mcmc}, sigma=1",
                    f"{PLOTS}/posterior_samples.png")

    # n=1000 naive — (2pi*psi)^{-500} underflows to 0 -> 0/0=nan -> wanders
    obs_1000 = rng.normal(true_theta, np.sqrt(true_psi), size=1000)
    s1000_naive = mh_posterior_naive(obs_1000, true_theta, true_psi, n_mcmc, sigma=1.0)
    _plot_posterior(s1000_naive, true_theta, true_psi,
                    f"Posterior MH naive  n_obs=1000  (numerical issue)",
                    f"{PLOTS}/posterior_samples_xobs1000.png")

    # n=1000 log-stable
    s1000_log = mh_posterior_log(obs_1000, true_theta, true_psi, n_mcmc, sigma=1.0)
    _plot_posterior(s1000_log, true_theta, true_psi,
                    f"Log-posterior MH  n_obs=1000, sigma=1",
                    f"{PLOTS}/log_posterior_samples.png")

    # sigma comparison: 0.01
    s10_s001 = mh_posterior_log(obs_10, true_theta, true_psi, n_mcmc, sigma=0.01)
    _plot_posterior(s10_s001, true_theta, true_psi,
                    f"Posterior MH  n_obs=10, sigma=0.01",
                    f"{PLOTS}/posterior_samples_xobs10sigma_001.png")
    s1000_s001 = mh_posterior_log(obs_1000, true_theta, true_psi, n_mcmc, sigma=0.01)
    _plot_posterior(s1000_s001, true_theta, true_psi,
                    f"Log-posterior MH  n_obs=1000, sigma=0.01",
                    f"{PLOTS}/log_posterior_samples_sigma_001.png")

    # sigma comparison: 50
    s10_s50 = mh_posterior_log(obs_10, true_theta, true_psi, n_mcmc, sigma=50.0)
    _plot_posterior(s10_s50, true_theta, true_psi,
                    f"Posterior MH  n_obs=10, sigma=50",
                    f"{PLOTS}/posterior_samples_xobs10_sigma50.png")
    s1000_s50 = mh_posterior_log(obs_1000, true_theta, true_psi, n_mcmc, sigma=50.0)
    _plot_posterior(s1000_s50, true_theta, true_psi,
                    f"Log-posterior MH  n_obs=1000, sigma=50",
                    f"{PLOTS}/log_posterior_samples_sigma50.png")


def main() -> None:
    logger.info("Exercise 6 — MCMC", seed=settings.SEED)

    logger.info("--- Part 1: Truncated Poisson via MH ---")
    part1()

    logger.info("--- Part 2: Joint truncated distribution ---")
    part2()

    logger.info("--- Part 3: Bayesian inference (log-normal prior) ---")
    part3()


if __name__ == "__main__":
    main()
