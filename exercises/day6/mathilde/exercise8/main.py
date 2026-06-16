"""
Run with:
uv run -m exercises.day6.mathilde.exercise8.main
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from exercises.day6.mathilde.exercise8.bootstrap import (
    bootstrap_replicates,
    bootstrap_variance,
    pareto_sample,
    sample_variance,
)
from utils.plotting import figure

SEED = 42
OUT = Path(__file__).resolve().parent


def part1_bootstrap_mean(rng: np.random.Generator) -> None:
    data = np.array([56, 101, 78, 67, 93, 87, 64, 72, 80, 69])
    a, b = -5, 5
    xbar = data.mean()

    reps = bootstrap_replicates(data, np.mean, 10_000, rng)
    diff = reps - xbar
    p_hat = np.mean((a < diff) & (diff < b))

    print("\nPart 1: bootstrap estimate of p = P{a < mean(X) - mu < b}")
    print(f"xbar = {xbar:.4f}")
    print(f"p_hat = {p_hat:.4f}")

    with figure(figsize=(8, 5), save=OUT / "ex13_bootstrap_mean.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(diff, bins=40, edgecolor="black", linewidth=0.6)
        ax.axvspan(a, b, alpha=0.2, color="orange", label=f"[{a}, {b}]")
        ax.set_xlabel(r"$\bar{X}^* - \bar{x}$")
        ax.set_ylabel("Count")
        ax.set_title("Bootstrap distribution of $\\bar{X}^* - \\bar{x}$")
        ax.legend()


def part2_var_s2(rng: np.random.Generator) -> None:
    data = np.array([5, 4, 9, 6, 21, 17, 11, 20, 7, 10, 21, 15, 13, 16, 8])

    s2, var_hat = bootstrap_variance(data, sample_variance, 10_000, rng)

    print("\nPart 2: bootstrap estimate of Var(S^2)")
    print(f"observed S^2 = {s2:.4f}")
    print(f"Var_hat(S^2) = {var_hat:.4f}")

    reps = bootstrap_replicates(data, sample_variance, 10_000, rng)
    with figure(figsize=(8, 5), save=OUT / "ex15_var_s2.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(reps, bins=40, edgecolor="black", linewidth=0.6)
        ax.set_xlabel(r"$S^{2*}$")
        ax.set_ylabel("Count")
        ax.set_title("Bootstrap distribution of $S^{2*}$")

    # sanity check: Ross Ex. 14, n=2, X={1,3} -> bootstrap Var(S^2) = 1
    check_rng = np.random.default_rng(0)
    _, var_check = bootstrap_variance(np.array([1.0, 3.0]), sample_variance, 50_000, check_rng)
    assert np.isclose(var_check, 1.0, atol=0.05)


def part3_mean_vs_median(rng: np.random.Generator) -> None:
    data = pareto_sample(1.0, 1.05, 200, rng)

    mean_hat, mean_var = bootstrap_variance(data, np.mean, 100, rng)
    median_hat, median_var = bootstrap_variance(data, np.median, 100, rng)

    print("\nPart 3: mean vs. median, Pareto(beta=1, k=1.05), n=200")
    print(f"{'estimator':<10} {'estimate':>12} {'boot. var':>14} {'boot. std err':>14}")
    print(f"{'mean':<10} {mean_hat:>12.4f} {mean_var:>14.4f} {np.sqrt(mean_var):>14.4f}")
    print(f"{'median':<10} {median_hat:>12.4f} {median_var:>14.4f} {np.sqrt(median_var):>14.4f}")

    mean_reps = bootstrap_replicates(data, np.mean, 100, rng)
    median_reps = bootstrap_replicates(data, np.median, 100, rng)

    with figure(figsize=(10, 5), save=OUT / "part3_mean_vs_median.png") as fig:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.hist(mean_reps, bins=20, edgecolor="black", linewidth=0.6)
        ax1.set_title("Bootstrap replicates of the mean")
        ax1.set_xlabel(r"$\bar{X}^*$")
        ax2.hist(median_reps, bins=20, edgecolor="black", linewidth=0.6)
        ax2.set_title("Bootstrap replicates of the median")
        ax2.set_xlabel(r"$\mathrm{median}^*$")


def main() -> None:
    rng = np.random.default_rng(SEED)
    part1_bootstrap_mean(rng)
    part2_var_s2(rng)
    part3_mean_vs_median(rng)


if __name__ == "__main__":
    main()
