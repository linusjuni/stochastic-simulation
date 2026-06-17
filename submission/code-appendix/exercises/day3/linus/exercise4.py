import heapq

import numpy as np
import scipy.stats as st
from matplotlib.lines import Line2D

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)
rng = np.random.default_rng(settings.SEED)

M, S, LAM = 10, 8.0, 1.0
A = LAM * S
N_CUSTOMERS = 10_000
N_RUNS = 10
ARRIVAL, DEPARTURE = 0, 1


def erlang_b(traffic: float, m: int) -> float:
    b = 1.0
    for i in range(1, m + 1):
        b = traffic * b / (i + traffic * b)
    return b


def simulate(inter_arrivals: np.ndarray, services: np.ndarray, m: int = M) -> float:
    arrivals = np.cumsum(inter_arrivals)
    events = [(t, ARRIVAL, i) for i, t in enumerate(arrivals)]
    heapq.heapify(events)
    n_busy, blocked = 0, 0
    while events:
        t, kind, i = heapq.heappop(events)
        if kind == ARRIVAL:
            if n_busy < m:
                n_busy += 1
                heapq.heappush(events, (t + services[i], DEPARTURE, i))
            else:
                blocked += 1
        else:
            n_busy -= 1
    return blocked / len(arrivals)


def run_config(arrival_gen, service_gen) -> tuple[float, float, float]:
    fracs = np.array([
        simulate(arrival_gen(N_CUSTOMERS), service_gen(N_CUSTOMERS))
        for _ in range(N_RUNS)
    ])
    mean = float(fracs.mean())
    half = float(st.t.ppf(0.975, df=N_RUNS - 1) * fracs.std(ddof=1) / np.sqrt(N_RUNS))
    return mean, mean - half, mean + half


def poisson_arr(n: int) -> np.ndarray:
    return rng.exponential(1.0, n)


def erlang2_arr(n: int) -> np.ndarray:
    return rng.gamma(shape=2, scale=0.5, size=n)


def hyperexp_arr(n: int) -> np.ndarray:
    branch = rng.choice(2, size=n, p=[0.8, 0.2])
    rates = np.where(branch == 0, 0.8333, 5.0)
    return rng.exponential(1.0 / rates)


def exp_svc(n: int) -> np.ndarray:
    return rng.exponential(S, n)


def const_svc(n: int) -> np.ndarray:
    return np.full(n, S)


def pareto_svc(k: float):
    beta = S * (k - 1) / k

    def gen(n: int) -> np.ndarray:
        return beta / rng.uniform(size=n) ** (1.0 / k)

    return gen


def lognormal_svc(n: int) -> np.ndarray:
    sigma, mu = 1.0, np.log(S) - 0.5
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


def main() -> None:
    B = erlang_b(A, M)
    logger.info("Erlang B reference", A=A, m=M, B=round(B, 4))

    results: dict[str, tuple[float, float, float]] = {}

    configs = [
        ("Poisson / Exponential",    poisson_arr,  exp_svc),
        ("Erlang(2) arrivals",        erlang2_arr,  exp_svc),
        ("Hyperexponential arrivals", hyperexp_arr, exp_svc),
        ("Constant service",          poisson_arr,  const_svc),
        ("Pareto k=1.05",             poisson_arr,  pareto_svc(1.05)),
        ("Pareto k=2.05",             poisson_arr,  pareto_svc(2.05)),
        ("Lognormal service",         poisson_arr,  lognormal_svc),
    ]

    for label, arr_gen, svc_gen in configs:
        mean, lo, hi = run_config(arr_gen, svc_gen)
        logger.info(label, fraction=round(mean, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), B_in_ci=lo <= B <= hi)
        results[label] = (mean, lo, hi)

    names = list(results.keys())
    means = np.array([results[n][0] for n in names])
    los   = np.array([results[n][0] - results[n][1] for n in names])
    his   = np.array([results[n][2] - results[n][0] for n in names])
    in_ci = np.array([results[n][1] <= B <= results[n][2] for n in names])

    match_c, dev_c, ref_c = "#2ca02c", "#d62728", "#444444"

    with figure(figsize=(10, 6), save="exercises/day3/linus/plots/comparison.png") as fig:
        ax = fig.add_subplot(111)
        y = np.arange(len(names))

        ax.axvspan(B - 0.002, B + 0.002, color=ref_c, alpha=0.10, zorder=1)
        ax.axvline(B, color=ref_c, linestyle="--", lw=1.5, zorder=2)

        for color, mask in [(match_c, in_ci), (dev_c, ~in_ci)]:
            ax.errorbar(
                means[mask], y[mask],
                xerr=[los[mask], his[mask]],
                fmt="o", color=color, ecolor=color,
                capsize=5, markersize=8, elinewidth=2, zorder=3,
            )

        for yi, m, h, ok in zip(y, means, his, in_ci):
            ax.annotate(
                f"{m:.3f}", (m + h, yi),
                xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=match_c if ok else dev_c,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Blocked fraction")
        ax.set_title(f"Blocking probability: simulation vs. Erlang B  (A={A}, m={M})")

        handles = [
            Line2D([0], [0], marker="o", lw=0, markersize=8, color=match_c, label="CI contains Erlang B"),
            Line2D([0], [0], marker="o", lw=0, markersize=8, color=dev_c,   label="CI misses Erlang B"),
            Line2D([0], [0], color=ref_c, linestyle="--", lw=1.5,           label=f"Erlang B = {B:.4f}"),
        ]
        ax.legend(handles=handles, loc="lower right", framealpha=0.9)


if __name__ == "__main__":
    main()
