"""
Run with:
uv run -m exercises.day2.exercise2.main
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from exercises.day2.exercise2.discrete import (
    alias_build,
    alias_sample,
    crude_sample,
    geometric_pmf,
    geometric_sample,
    rejection_sample,
)
from exercises.day2.exercise2.stats_tests import (
    chi_square_gof,
    geometric_chi_square,
)
from utils.plotting import figure

PROBS = np.array([7 / 48, 5 / 48, 6 / 48, 3 / 48, 12 / 48, 15 / 48])
VALUES = np.array([1, 2, 3, 4, 5, 6])
N = 10_000
SEED = 69
OUT = Path(__file__).resolve().parents[3] / "report" / "plots" / "day2"


def _six_point_counts(samples: np.ndarray) -> np.ndarray:
    return np.array([np.sum(samples == v) for v in VALUES], dtype=float)


# --- Part 1 ---
def part1_geometric(rng: np.random.Generator) -> None:
    print("=== Part 1: Geometric distribution ===")
    for p in [0.1, 0.3, 0.7]:
        samples = geometric_sample(p, N, rng)
        gof = geometric_chi_square(samples, p)
        print(
            f"  p={p}: mean={samples.mean():.3f} (theory={1 / p:.3f}), "
            f"chi2={gof['statistic']:.3f}, p-value={gof['pvalue']:.4f}"
        )

        max_k = max(int(np.percentile(samples, 99)), 10)
        k_vals = np.arange(1, max_k + 1)
        theory = np.array([geometric_pmf(p, k) for k in k_vals])
        counts = np.array([np.sum(samples == k) for k in k_vals], dtype=float)

        with figure(figsize=(8, 4), save=OUT / f"geometric_p{p}.png") as fig:
            ax = fig.add_subplot(111)
            ax.bar(k_vals, counts / N, label="simulated", alpha=0.7, width=0.6)
            ax.plot(
                k_vals, theory, "ro-", markersize=4, linewidth=1, label="theoretical"
            )
            ax.set_xlabel("k")
            ax.set_ylabel("probability")
            ax.set_title(f"Geometric distribution, p={p}")
            ax.legend()


# --- Part 2 ---
def part2_six_point() -> None:
    print("\n=== Part 2: Six-point distribution ===")
    prob_table, alias_table = alias_build(PROBS)

    for seed in [42, 69]:
        print(f"  -- seed {seed} --")
        seed_rng = np.random.default_rng(seed)
        methods = {
            "crude": crude_sample(PROBS, N, seed_rng),
            "rejection": rejection_sample(PROBS, N, seed_rng),
            "alias": alias_sample(prob_table, alias_table, N, seed_rng),
        }

        for name, samples in methods.items():
            counts = _six_point_counts(samples)
            gof = chi_square_gof(counts, PROBS, N)
            print(f"  {name}: chi2={gof['statistic']:.4f}, p-value={gof['pvalue']:.4f}")

            if seed == 69:
                x = np.arange(len(VALUES))
                w = 0.35
                with figure(figsize=(8, 4), save=OUT / f"sixpoint_{name}.png") as fig:
                    ax = fig.add_subplot(111)
                    ax.bar(x - w / 2, counts / N, width=w, label="simulated", alpha=0.8)
                    ax.bar(x + w / 2, PROBS, width=w, label="theoretical", alpha=0.8)
                    ax.set_xticks(x)
                    ax.set_xticklabels(VALUES)
                    ax.set_xlabel("x")
                    ax.set_ylabel("probability")
                    ax.set_title(f"Six-point distribution — {name} method")
                    ax.legend()


# --- Part 3: comparison ---
def part3_compare(rng: np.random.Generator) -> None:
    print("\n=== Part 3: Method comparison (n=1,000,000) ===")
    prob_table, alias_table = alias_build(PROBS)
    n_large = 1_000_000

    methods = {
        "crude": lambda: crude_sample(PROBS, n_large, rng),
        "rejection": lambda: rejection_sample(PROBS, n_large, rng),
        "alias": lambda: alias_sample(prob_table, alias_table, n_large, rng),
    }

    print(f"  {'Method':<12} {'chi2':>8} {'p-value':>8} {'time (s)':>10}")
    timings: dict[str, float] = {}
    for name, fn in methods.items():
        t0 = time.perf_counter()
        samples = fn()
        elapsed = time.perf_counter() - t0
        timings[name] = elapsed
        counts = _six_point_counts(samples)
        gof = chi_square_gof(counts, PROBS, n_large)
        print(
            f"  {name:<12} {gof['statistic']:>8.4f} {gof['pvalue']:>8.4f} {elapsed:>10.4f}"
        )

    names = list(timings.keys())
    times = [timings[n] for n in names]
    with figure(figsize=(6, 4), save=OUT / "method_timing.png") as fig:
        ax = fig.add_subplot(111)
        ax.bar(names, times, alpha=0.8)
        ax.set_ylabel("time (s)")
        ax.set_title("Sampling time for n=1,000,000")


def main() -> None:
    rng = np.random.default_rng(SEED)
    part1_geometric(rng)
    part2_six_point()
    part3_compare(rng)


if __name__ == "__main__":
    main()
