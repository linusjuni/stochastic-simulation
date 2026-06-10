"""
Run with:
uv run -m exercises.day3.mathilde.main
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.stats as st

from exercises.day3.mathilde.blocking_system import (
    constant_service,
    erlang_b,
    erlang_interarrival,
    exp_interarrival,
    exp_service,
    hyperexp_interarrival,
    lognormal_service,
    pareto_service,
    simulate,
)
from utils.plotting import figure

M = 10
N_CUSTOMERS = 10_000
N_REPS = 10
SEED = 42
A = 8.0  # offered traffic = lambda * s = 1 * 8
ERLANG_B = erlang_b(A, M)
OUT = Path(__file__).resolve().parent


def run_config(arrival_gen, service_gen, rng: np.random.Generator) -> tuple[float, float, float, np.ndarray]:
    fractions = np.empty(N_REPS)
    for i in range(N_REPS):
        interarrival = arrival_gen(N_CUSTOMERS, rng)
        service = service_gen(N_CUSTOMERS, rng)
        fractions[i] = simulate(interarrival, service, M)

    mean = fractions.mean()
    se = fractions.std(ddof=1) / np.sqrt(N_REPS)
    t_crit = st.t.ppf(0.975, df=N_REPS - 1)
    return mean, mean - t_crit * se, mean + t_crit * se, fractions


def report(name: str, mean: float, lo: float, hi: float) -> None:
    print(f"  {name:<28} fraction={mean:.4f}  CI=[{lo:.4f}, {hi:.4f}]")


def main() -> None:
    rng = np.random.default_rng(SEED)
    results: dict[str, tuple[float, float, float]] = {}

    print(f"Erlang B (A={A}, m={M}) = {ERLANG_B:.4f}\n")

    print("=== Part 1: Poisson arrivals, exponential service ===")
    mean, lo, hi, _ = run_config(exp_interarrival, exp_service, rng)
    report("Poisson / Exponential", mean, lo, hi)
    results["Poisson/Exp (Part 1)"] = (mean, lo, hi)

    print("\n=== Part 2: Renewal arrival processes (exp service) ===")
    mean, lo, hi, _ = run_config(erlang_interarrival(shape=2), exp_service, rng)
    report("Erlang(2) arrivals", mean, lo, hi)
    results["Erlang arrivals (Part 2a)"] = (mean, lo, hi)

    mean, lo, hi, _ = run_config(hyperexp_interarrival, exp_service, rng)
    report("Hyperexponential arrivals", mean, lo, hi)
    results["Hyperexp arrivals (Part 2b)"] = (mean, lo, hi)

    print("\n=== Part 3: Poisson arrivals, varied service distributions ===")
    mean, lo, hi, _ = run_config(exp_interarrival, constant_service, rng)
    report("Constant service", mean, lo, hi)
    results["Constant service (Part 3a)"] = (mean, lo, hi)

    mean, lo, hi, _ = run_config(exp_interarrival, pareto_service(k=1.05), rng)
    report("Pareto k=1.05 service", mean, lo, hi)
    results["Pareto k=1.05 (Part 3b)"] = (mean, lo, hi)

    mean, lo, hi, _ = run_config(exp_interarrival, pareto_service(k=2.05), rng)
    report("Pareto k=2.05 service", mean, lo, hi)
    results["Pareto k=2.05 (Part 3b)"] = (mean, lo, hi)

    mean, lo, hi, _ = run_config(exp_interarrival, lognormal_service, rng)
    report("Lognormal service", mean, lo, hi)
    results["Lognormal service (Part 3c)"] = (mean, lo, hi)

    print("\n=== Part 4: Comparison ===")
    print(f"  {'Configuration':<28} {'Fraction':>9} {'CI low':>9} {'CI high':>9}  {'Erlang B in CI?':>15}")
    for name, (mean, lo, hi) in results.items():
        in_ci = lo <= ERLANG_B <= hi
        print(f"  {name:<28} {mean:>9.4f} {lo:>9.4f} {hi:>9.4f}  {str(in_ci):>15}")

    names = list(results.keys())
    means = [results[n][0] for n in names]
    los = [results[n][0] - results[n][1] for n in names]
    his = [results[n][2] - results[n][0] for n in names]

    with figure(figsize=(10, 6), save=OUT / "comparison.png") as fig:
        ax = fig.add_subplot(111)
        y = np.arange(len(names))
        ax.errorbar(means, y, xerr=[los, his], fmt="o", capsize=4)
        ax.axvline(ERLANG_B, color="red", linestyle="--", label=f"Erlang B = {ERLANG_B:.4f}")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("blocked fraction")
        ax.set_title("Blocking probability: simulation vs Erlang B")
        ax.legend()
        ax.invert_yaxis()


if __name__ == "__main__":
    main()
