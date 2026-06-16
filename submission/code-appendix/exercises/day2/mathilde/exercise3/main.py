"""
Run with:
uv run -m exercises.day2.mathilde.exercise3.main
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.stats as st

from exercises.day2.mathilde.exercise3.continuous import (
    exponential_sample,
    normal_box_muller,
    pareto_composition,
    pareto_sample,
)
from utils.plotting import figure

N = 10_000
SEED = 42
BETA = 1.0
K_VALUES = [2.05, 2.5, 3.0, 4.0]
OUT = Path(__file__).resolve().parents[4] / "report" / "plots" / "day2"


# --- Part 1a: Exponential ---
def part1a_exponential(rng: np.random.Generator) -> None:
    lam = 1.0
    samples = exponential_sample(lam, N, rng)
    ks_stat, ks_pval = st.kstest(samples, "expon", args=(0, 1 / lam))
    print(
        f"Exponential(λ={lam}): mean={samples.mean():.4f} (theory={1 / lam:.4f}), "
        f"KS={ks_stat:.4f}, p={ks_pval:.4f}"
    )

    x = np.linspace(0, np.percentile(samples, 99), 300)
    with figure(figsize=(7, 4), save=OUT / "exponential.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(samples, bins=50, density=True, alpha=0.7, label="simulated")
        ax.plot(x, st.expon.pdf(x, scale=1 / lam), "r-", lw=2, label="theoretical pdf")
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_title(f"Exponential(λ={lam})")
        ax.legend()


# --- Part 1b: Normal (Box-Muller) ---
def part1b_normal(rng: np.random.Generator) -> None:
    samples = normal_box_muller(N, rng)
    ks_stat, ks_pval = st.kstest(samples, "norm")
    print(
        f"Normal (Box-Muller): mean={samples.mean():.4f}, std={samples.std():.4f}, "
        f"KS={ks_stat:.4f}, p={ks_pval:.4f}"
    )

    x = np.linspace(-4, 4, 300)
    with figure(figsize=(7, 4), save=OUT / "normal.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(samples, bins=50, density=True, alpha=0.7, label="simulated")
        ax.plot(x, st.norm.pdf(x), "r-", lw=2, label="N(0,1) pdf")
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_title("Normal distribution (Box-Muller)")
        ax.legend()


# --- Part 1c: Pareto ---
def part1c_pareto(rng: np.random.Generator) -> None:
    print("Pareto (β=1):")
    with figure(figsize=(12, 8), save=OUT / "pareto_all.png") as fig:
        for i, k in enumerate(K_VALUES, 1):
            samples = pareto_sample(k, BETA, N, rng)
            ks_stat, ks_pval = st.kstest(samples, st.pareto(b=k, scale=BETA).cdf)
            print(
                f"  k={k}: mean={samples.mean():.4f}, KS={ks_stat:.4f}, p={ks_pval:.4f}"
            )

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
            ax.set_title(f"Pareto k={k}")
            ax.set_xlabel("x")
            ax.legend(fontsize=8)
        fig.tight_layout()


# --- Part 2: Pareto moments ---
def part2_pareto_moments(rng: np.random.Generator) -> None:
    print("\nPareto moments (β=1, n=10,000):")
    print(
        f"  {'k':>5}  {'E[X] sim':>10} {'E[X] theory':>12}  {'Var sim':>12} {'Var theory':>12}"
    )
    for k in K_VALUES:
        samples = pareto_sample(k, BETA, N, rng)
        e_theory = BETA * k / (k - 1)
        var_theory = BETA**2 * k / ((k - 1) ** 2 * (k - 2)) if k > 2 else float("inf")
        print(
            f"  {k:>5}  {samples.mean():>10.4f} {e_theory:>12.4f}  "
            f"{samples.var():>12.4f} {var_theory:>12.4f}"
        )
    print(
        "  Note: k=2.05 has k-2=0.05 → huge theoretical variance; sample variance is "
        "unreliable due to the heavy tail."
    )


# --- Part 3: Normal confidence intervals ---
def part3_normal_ci(rng: np.random.Generator) -> None:
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
        x = normal_box_muller(n_obs, rng)
        xbar, s2 = x.mean(), x.var(ddof=1)
        # mean CI
        half = t_crit * np.sqrt(s2 / n_obs)
        lo_m, hi_m = xbar - half, xbar + half
        lo_means.append(lo_m)
        hi_means.append(hi_m)
        if lo_m <= 0.0 <= hi_m:
            cover_mean += 1
        # variance CI
        lo_v = (n_obs - 1) * s2 / chi2_hi
        hi_v = (n_obs - 1) * s2 / chi2_lo
        if lo_v <= 1.0 <= hi_v:
            cover_var += 1

    print(f"\nNormal CIs (n={n_obs}, 100 intervals at 95%):")
    print(f"  Mean CI coverage:     {cover_mean}/100 (expect ~95)")
    print(f"  Variance CI coverage: {cover_var}/100 (expect ~95)")

    with figure(figsize=(10, 5), save=OUT / "ci_mean.png") as fig:
        ax = fig.add_subplot(111)
        for j, (lo, hi) in enumerate(zip(lo_means, hi_means)):
            color = "steelblue" if lo <= 0.0 <= hi else "tomato"
            ax.plot([lo, hi], [j, j], color=color, lw=0.8, alpha=0.7)
        ax.axvline(0, color="black", lw=1.5, label="true μ=0")
        ax.set_xlabel("μ")
        ax.set_ylabel("interval index")
        ax.set_title(f"100 × 95% CI for mean (coverage {cover_mean}/100)")
        ax.legend()


# --- Part 4: Pareto via composition ---
def part4_pareto_composition(rng: np.random.Generator) -> None:
    k = 3.0
    inv_samples = pareto_sample(k, BETA, N, rng)
    comp_samples = pareto_composition(k, BETA, N, rng)

    ks_stat, ks_pval = st.kstest(comp_samples, st.pareto(b=k, scale=BETA).cdf)
    ks2_stat, ks2_pval = st.ks_2samp(inv_samples, comp_samples)
    print(f"\nPareto composition (k={k}, β={BETA}):")
    print(f"  KS vs theory: stat={ks_stat:.4f}, p={ks_pval:.4f}")
    print(f"  KS vs inversion: stat={ks2_stat:.4f}, p={ks2_pval:.4f}")

    clip = np.percentile(np.concatenate([inv_samples, comp_samples]), 99)
    x = np.linspace(BETA, clip, 300)
    with figure(figsize=(8, 4), save=OUT / "pareto_composition.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(
            inv_samples[inv_samples <= clip],
            bins=60,
            density=True,
            alpha=0.5,
            label="inversion",
        )
        ax.hist(
            comp_samples[comp_samples <= clip],
            bins=60,
            density=True,
            alpha=0.5,
            label="composition",
        )
        ax.plot(x, st.pareto.pdf(x, b=k, scale=BETA), "r-", lw=2, label="pdf")
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_title(f"Pareto k={k}: inversion vs composition")
        ax.legend()


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("=== Part 1: Sampling from continuous distributions ===")
    part1a_exponential(rng)
    part1b_normal(rng)
    part1c_pareto(rng)
    part2_pareto_moments(rng)
    part3_normal_ci(rng)
    part4_pareto_composition(rng)


if __name__ == "__main__":
    main()
