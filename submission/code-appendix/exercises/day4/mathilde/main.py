"""
Run with:
uv run -m exercises.day4.mathilde.main
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.stats as st

from exercises.day4.mathilde.blocking_cv import ERLANG_B, estimate_c, run_once
from exercises.day4.mathilde.importance import (
    crude_tail,
    is_exponential,
    is_pareto_mean,
    is_tail,
    optimal_lambda,
    var_W,
)
from exercises.day4.mathilde.integration import (
    antithetic,
    control_variate,
    crude_mc,
    stratified,
)
from exercises.day4.mathilde.stats import t_ci
from utils.plotting import figure

SEED = 42
N = 100  # function evaluations for parts 1–4
N_PAIRS = 50  # antithetic pairs (= 100 evaluations)
K_STRATA = 10  # strata count for stratified sampling
N_CUSTOMERS = 10_000
N_PILOT = 100
N_REPS = 100
N_IS = 10_000
OUT = Path(__file__).resolve().parents[3] / "report" / "plots" / "day4"


def part1_crude(rng: np.random.Generator) -> tuple[float, float, float, float]:
    samples = crude_mc(N, rng)
    mean, lo, hi, width = t_ci(samples)
    print("=== Part 1: Crude MC (n=100) ===")
    print(f"  mean={mean:.6f}  CI=[{lo:.6f}, {hi:.6f}]  width={width:.6f}")
    return mean, lo, hi, width


def part2_antithetic(rng: np.random.Generator) -> tuple[float, float, float, float]:
    samples = antithetic(N_PAIRS, rng)
    mean, lo, hi, width = t_ci(samples)
    print("\n=== Part 2: Antithetic Variables (n=50 pairs) ===")
    print(f"  mean={mean:.6f}  CI=[{lo:.6f}, {hi:.6f}]  width={width:.6f}")
    return mean, lo, hi, width


def part3_control(rng: np.random.Generator) -> tuple[float, float, float, float]:
    samples, c = control_variate(N, rng)
    mean, lo, hi, width = t_ci(samples)
    print("\n=== Part 3: Control Variate (n=100) ===")
    print(f"  estimated c = {c:.4f}")
    print(f"  mean={mean:.6f}  CI=[{lo:.6f}, {hi:.6f}]  width={width:.6f}")
    return mean, lo, hi, width


def part4_stratified(rng: np.random.Generator) -> tuple[float, float, float, float]:
    samples = stratified(N, K_STRATA, rng)
    mean, lo, hi, width = t_ci(samples)
    print("\n=== Part 4: Stratified Sampling (n=100, k=10) ===")
    print(f"  mean={mean:.6f}  CI=[{lo:.6f}, {hi:.6f}]  width={width:.6f}")
    return mean, lo, hi, width


def _plot_ci_comparison(results: dict[str, tuple[float, float, float, float]]) -> None:
    names = list(results.keys())
    means = np.array([r[0] for r in results.values()])
    los = np.array([r[1] for r in results.values()])
    his = np.array([r[2] for r in results.values()])
    true_val = math.e - 1.0

    with figure(figsize=(9, 4), save=OUT / "ci_comparison.png") as fig:
        ax = fig.add_subplot(111)
        y = np.arange(len(names))
        ax.axvline(
            true_val,
            color="#d62728",
            linestyle="--",
            lw=1.5,
            label=f"e−1 = {true_val:.6f}",
        )
        for yi, m, lo, hi in zip(y, means, los, his):
            ax.plot([lo, hi], [yi, yi], color="#2ca02c", lw=2.5)
            ax.plot(m, yi, "o", color="#2ca02c", markersize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("Estimate of ∫₀¹ eˣ dx")
        ax.set_title("95% CIs across variance-reduction methods (100 evaluations)")
        ax.legend(loc="lower right")


def part5_blocking(rng: np.random.Generator) -> None:
    print("\n=== Part 5: Blocking System with Control Variate ===")
    print(f"  Erlang B (A=8, m=10) = {ERLANG_B:.6f}")

    pilot_Bs = np.empty(N_PILOT)
    pilot_Ss = np.empty(N_PILOT)
    for i in range(N_PILOT):
        pilot_Bs[i], pilot_Ss[i] = run_once(N_CUSTOMERS, rng)
    c = estimate_c(pilot_Bs, pilot_Ss)
    print(f"  pilot c = {c:.6f}")

    Bs = np.empty(N_REPS)
    Ss = np.empty(N_REPS)
    for i in range(N_REPS):
        Bs[i], Ss[i] = run_once(N_CUSTOMERS, rng)
    Ys = Bs - c * (Ss - N_CUSTOMERS)

    crude_mean, _, _, crude_w = t_ci(Bs)
    cv_mean, _, _, cv_w = t_ci(Ys)

    print(f"\n  {'Method':<20} {'Mean':>10} {'CI width':>12} {'Reduction':>10}")
    print(f"  {'-' * 55}")
    print(f"  {'Crude (no CV)':<20} {crude_mean:>10.6f} {crude_w:>12.6f} {'---':>10}")
    print(
        f"  {'Control variate':<20} {cv_mean:>10.6f} {cv_w:>12.6f} {crude_w / cv_w:>9.2f}×"
    )


def part7_is_tail(rng: np.random.Generator) -> None:
    print("\n=== Part 7: Importance Sampling for P(Z > a) ===")
    for n in [100, 1000, N_IS]:
        for a in [2, 4]:
            true_p = st.norm.sf(a)
            print(f"\n  n={n}, a={a}  [true P(Z>{a}) = {true_p:.2e}]")
            samples = crude_tail(n, a, rng)
            m, _, _, w = t_ci(samples)
            print(f"    {'Crude':<18} mean={m:.2e}   width={w:.2e}")
            for sigma2 in [1, 2, 3]:
                samples_is = is_tail(n, a, sigma2, rng)
                m, _, _, w = t_ci(samples_is)
                print(f"    {f'IS σ²={sigma2}':<18} mean={m:.2e}   width={w:.2e}")


def part8_exp_is(rng: np.random.Generator) -> None:
    print("\n=== Part 8: IS with Exp(λ) for ∫₀¹ eˣ dx ===")
    lam_star, var_star = optimal_lambda()
    print(f"  λ* = {lam_star:.4f},  Var(W) at λ* = {var_star:.4f}")

    lambdas = np.linspace(0.05, 8.0, 400)
    variances = np.vectorize(var_W)(lambdas)
    with figure(figsize=(7, 4), save=OUT / "is_lambda_variance.png") as fig:
        ax = fig.add_subplot(111)
        ax.plot(lambdas, variances, lw=2)
        ax.axvline(
            lam_star, color="#d62728", linestyle="--", label=f"λ* = {lam_star:.4f}"
        )
        ax.axhline(var_star, color="#d62728", linestyle=":", alpha=0.5)
        ax.set_xlabel("λ")
        ax.set_ylabel("Var(W)")
        ax.set_title("IS variance for Exp(λ) proposal on ∫₀¹ eˣ dx")
        ax.legend()

    print(f"\n  n={N_IS},  true value e−1 = {math.e - 1:.6f}")
    print(f"  {'Estimator':<24} {'Mean':>10} {'CI width':>12}")
    print(f"  {'-' * 48}")
    samples_crude = crude_mc(N_IS, rng)
    m, _, _, w = t_ci(samples_crude)
    print(f"  {'Crude MC':<24} {m:>10.6f} {w:>12.6f}")
    for lam in [0.5, 1.0, lam_star, 2.0, 5.0]:
        samples_is = is_exponential(N_IS, lam, rng)
        m, _, _, w = t_ci(samples_is)
        label = f"IS λ*={lam:.3f}" if abs(lam - lam_star) < 0.01 else f"IS λ={lam:.3f}"
        print(f"  {label:<24} {m:>10.6f} {w:>12.6f}")


def part9_pareto(rng: np.random.Generator) -> None:
    print("\n=== Part 9: Pareto First-Moment IS ===")
    k = 3.0
    beta = 8.0 * (k - 1.0) / k
    true_mean = k * beta / (k - 1.0)  # = 8.0
    samples = is_pareto_mean(N_IS, k, rng)
    m, lo, hi, w = t_ci(samples)
    print(f"  Pareto(k={k:.0f}, β={beta:.4f}),  true mean = {true_mean:.4f}")
    print(f"  IS mean = {m:.6f}  CI=[{lo:.6f}, {hi:.6f}]  width = {w:.2e}")
    print(f"  (weights w(x) = k·β/(k−1) = {true_mean:.4f} exactly → Var = 0 in theory)")


def main() -> None:
    rng = np.random.default_rng(SEED)

    r1 = part1_crude(rng)
    r2 = part2_antithetic(rng)
    r3 = part3_control(rng)
    r4 = part4_stratified(rng)

    true_val = math.e - 1.0
    print(f"\n  True value e−1 = {true_val:.6f}")
    print(f"\n  {'Method':<24} {'CI width':>10} {'Reduction':>12}")
    print(f"  {'-' * 48}")
    for name, r in [
        ("Crude", r1),
        ("Antithetic", r2),
        ("Control variate", r3),
        ("Stratified", r4),
    ]:
        reduction = r1[3] / r[3] if name != "Crude" else 1.0
        suffix = "---" if name == "Crude" else f"{reduction:.1f}×"
        print(f"  {name:<24} {r[3]:>10.4f} {suffix:>12}")

    _plot_ci_comparison(
        {
            "Crude (n=100)": r1,
            "Antithetic (50 pairs)": r2,
            "Control variate (n=100)": r3,
            "Stratified (n=100, k=10)": r4,
        }
    )

    part5_blocking(rng)
    part7_is_tail(rng)
    part8_exp_is(rng)
    part9_pareto(rng)


if __name__ == "__main__":
    main()
