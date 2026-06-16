import numpy as np
import scipy.stats as st

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)
rng = np.random.default_rng(settings.SEED)
N = 10_000
BETA = 1.0
K_VALUES = [2.05, 2.5, 3.0, 4.0]
PLOTS = "exercises/day2/linus/continuous/plots"


# Part 1: samplers
def exponential_sample(lam: float, size: int) -> np.ndarray:
    u = rng.uniform(size=size)
    return -np.log(u) / lam


def normal_box_muller(size: int) -> np.ndarray:
    n_pairs = (size + 1) // 2
    u1 = rng.uniform(size=n_pairs)
    u2 = rng.uniform(size=n_pairs)
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    z = np.concatenate([r * np.cos(theta), r * np.sin(theta)])
    return z[:size]


def pareto_sample(k: float, beta: float, size: int) -> np.ndarray:
    u = rng.uniform(size=size)
    return beta / u ** (1.0 / k)


# Part 1 main logic
def part1_exponential() -> np.ndarray:
    lam = 1.0
    samples = exponential_sample(lam, N)
    ks_stat, ks_pval = st.kstest(samples, "expon", args=(0, 1 / lam))
    logger.info(
        "Exponential",
        lam=lam,
        mean=round(float(samples.mean()), 4),
        theory=round(1 / lam, 4),
        ks=round(float(ks_stat), 4),
        p_value=round(float(ks_pval), 4),
    )

    x = np.linspace(0, np.percentile(samples, 99), 300)
    with figure(figsize=(7, 4), save=f"{PLOTS}/exponential.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(samples, bins=50, density=True, alpha=0.7, label="simulated")
        ax.plot(x, st.expon.pdf(x, scale=1 / lam), "r-", lw=2, label="theoretical pdf")
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_title(f"Exponential(λ={lam}), n={N:,}")
        ax.legend()

    return samples


def part1_normal() -> np.ndarray:
    samples = normal_box_muller(N)
    ks_stat, ks_pval = st.kstest(samples, "norm")
    logger.info(
        "Normal (Box-Muller)",
        mean=round(float(samples.mean()), 4),
        std=round(float(samples.std()), 4),
        ks=round(float(ks_stat), 4),
        p_value=round(float(ks_pval), 4),
    )

    x = np.linspace(-4, 4, 300)
    with figure(figsize=(7, 4), save=f"{PLOTS}/normal.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(samples, bins=50, density=True, alpha=0.7, label="simulated")
        ax.plot(x, st.norm.pdf(x), "r-", lw=2, label="N(0,1) pdf")
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_title(f"Normal via Box-Muller, n={N:,}")
        ax.legend()

    return samples


def part1_pareto() -> dict[float, np.ndarray]:
    pareto_samples: dict[float, np.ndarray] = {}

    with figure(figsize=(12, 8), save=f"{PLOTS}/pareto_all.png") as fig:
        for i, k in enumerate(K_VALUES, 1):
            samples = pareto_sample(k, BETA, N)
            ks_stat, ks_pval = st.kstest(samples, st.pareto(b=k, scale=BETA).cdf)
            logger.info(
                "Pareto",
                k=k,
                beta=BETA,
                mean=round(float(samples.mean()), 4),
                theory_mean=round(BETA * k / (k - 1), 4),
                ks=round(float(ks_stat), 4),
                p_value=round(float(ks_pval), 4),
            )
            pareto_samples[k] = samples

            clip = np.percentile(samples, 99)
            x = np.linspace(BETA, clip, 300)
            ax = fig.add_subplot(2, 2, i)
            ax.hist(
                samples[samples <= clip],
                bins=60,
                density=True,
                alpha=0.7,
                label="simulated",
            )
            ax.plot(x, st.pareto.pdf(x, b=k, scale=BETA), "r-", lw=2, label="pdf")
            ax.set_title(f"Pareto(k={k}, β={BETA})")
            ax.set_xlabel("x")
            ax.legend(fontsize=8)
        fig.tight_layout()

    return pareto_samples


# Part 2: Pareto moments
def part2_pareto_moments(pareto_samples: dict[float, np.ndarray]) -> None:
    for k, samples in pareto_samples.items():
        e_theory = BETA * k / (k - 1)
        var_theory = BETA**2 * k / ((k - 1) ** 2 * (k - 2)) if k > 2 else float("inf")
        logger.info(
            "Pareto moments",
            k=k,
            mean_sim=round(float(samples.mean()), 4),
            mean_theory=round(e_theory, 4),
            var_sim=round(float(samples.var()), 4),
            var_theory=round(var_theory, 4) if var_theory != float("inf") else "inf",
        )


# Part 4: Pareto via composition
def pareto_composition(k: float, beta: float, size: int) -> np.ndarray:
    lam = rng.gamma(shape=k, scale=1.0 / beta, size=size)
    return beta + rng.exponential(scale=1.0 / lam)


def part4_pareto_composition(pareto_samples: dict[float, np.ndarray]) -> None:
    k = 3.0
    inv_samples = pareto_samples[k]
    comp_samples = pareto_composition(k, BETA, N)

    ks_stat, ks_pval = st.kstest(comp_samples, st.pareto(b=k, scale=BETA).cdf)
    ks2_stat, ks2_pval = st.ks_2samp(inv_samples, comp_samples)
    logger.info(
        "Pareto composition vs theory",
        k=k,
        ks=round(float(ks_stat), 4),
        p_value=round(float(ks_pval), 4),
    )
    logger.info(
        "Pareto composition vs inversion",
        k=k,
        ks=round(float(ks2_stat), 4),
        p_value=round(float(ks2_pval), 4),
    )

    clip = np.percentile(np.concatenate([inv_samples, comp_samples]), 99)
    x = np.linspace(BETA, clip, 300)
    with figure(figsize=(8, 4), save=f"{PLOTS}/pareto_composition.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(inv_samples[inv_samples <= clip], bins=60, density=True,
                alpha=0.5, label="inversion")
        ax.hist(comp_samples[comp_samples <= clip], bins=60, density=True,
                alpha=0.5, label="composition")
        ax.plot(x, st.pareto.pdf(x, b=k, scale=BETA), "r-", lw=2, label="pdf")
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_title(f"Pareto(k={k}, β={BETA}): inversion vs composition")
        ax.legend()


# Part 3: Normal confidence intervals
def part3_normal_ci() -> None:
    n_obs = 10
    n_ci = 100
    alpha = 0.05
    t_crit = st.t.ppf(1 - alpha / 2, df=n_obs - 1)
    chi2_lo = st.chi2.ppf(alpha / 2, df=n_obs - 1)
    chi2_hi = st.chi2.ppf(1 - alpha / 2, df=n_obs - 1)

    cover_mean = 0
    cover_var = 0
    lo_means, hi_means = [], []

    for _ in range(n_ci):
        x = normal_box_muller(n_obs)
        xbar, s2 = x.mean(), x.var(ddof=1)

        half = t_crit * np.sqrt(s2 / n_obs)
        lo_m, hi_m = xbar - half, xbar + half
        lo_means.append(lo_m)
        hi_means.append(hi_m)
        if lo_m <= 0.0 <= hi_m:
            cover_mean += 1

        lo_v = (n_obs - 1) * s2 / chi2_hi
        hi_v = (n_obs - 1) * s2 / chi2_lo
        if lo_v <= 1.0 <= hi_v:
            cover_var += 1

    logger.info(
        "Normal CI coverage",
        n_obs=n_obs,
        n_ci=n_ci,
        cover_mean=cover_mean,
        cover_var=cover_var,
        expected=95,
    )

    with figure(figsize=(10, 5), save=f"{PLOTS}/ci_mean.png") as fig:
        ax = fig.add_subplot(111)
        for j, (lo, hi) in enumerate(zip(lo_means, hi_means)):
            color = "steelblue" if lo <= 0.0 <= hi else "tomato"
            ax.plot([lo, hi], [j, j], color=color, lw=0.8, alpha=0.7)
        ax.axvline(0, color="black", lw=1.5, label="true μ=0")
        ax.set_xlabel("μ")
        ax.set_ylabel("interval index")
        ax.set_title(f"100 × 95% CI for mean (coverage {cover_mean}/100)")
        ax.legend()


def main() -> None:
    logger.info("Exercise 3 — continuous distributions", seed=settings.SEED, n=N)

    logger.info("Part 1: Sampling and KS tests")
    part1_exponential()
    part1_normal()
    pareto_samples = part1_pareto()

    logger.info("Part 2: Pareto moments")
    part2_pareto_moments(pareto_samples)

    logger.info("Part 3: Normal confidence intervals")
    part3_normal_ci()

    logger.info("Part 4: Pareto via composition")
    part4_pareto_composition(pareto_samples)


if __name__ == "__main__":
    main()
