import time

import numpy as np
from scipy.stats import chisquare

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)
rng = np.random.default_rng(settings.SEED)
N = 10_000


# Part 1: Geometric distribution
def geometric_sample(p: float, size: int) -> np.ndarray:
    u = rng.uniform(size=size)
    return np.floor(np.log(u) / np.log(1 - p)).astype(int) + 1


def geometric_pmf(k: np.ndarray, p: float) -> np.ndarray:
    return (1 - p) ** (k - 1) * p


def chi_square_geometric(samples: np.ndarray, p: float) -> tuple[float, float]:
    # bin k=1..max_k, group tail k>max_k into one final bin
    max_k = max(int(np.percentile(samples, 99)), 10)
    observed = np.zeros(max_k + 1)
    for x in samples:
        observed[min(x - 1, max_k)] += 1
    expected_probs = np.array([geometric_pmf(k, p) for k in range(1, max_k + 1)])
    tail_prob = max(0.0, 1.0 - expected_probs.sum())
    expected_probs = np.append(expected_probs, tail_prob)
    stat, pval = chisquare(observed, expected_probs * len(samples))
    return stat, pval


# Part 2: Six-point distribution
PROBS = np.array([7 / 48, 5 / 48, 1 / 8, 1 / 16, 1 / 4, 5 / 16])
VALUES = np.arange(1, 7)


def crude_sample(size: int, rng: np.random.Generator) -> np.ndarray:
    cdf = np.cumsum(PROBS)
    u = rng.uniform(size=size)
    return np.searchsorted(cdf, u, side="left") + 1


def rejection_sample(size: int, rng: np.random.Generator) -> np.ndarray:
    k = len(PROBS)
    c = PROBS.max()
    results = []
    while len(results) < size:
        I = rng.integers(0, k)
        u = rng.uniform()
        if u <= PROBS[I] / c:
            results.append(I + 1)
    return np.array(results[:size])


def alias_build() -> tuple[np.ndarray, np.ndarray]:
    n = len(PROBS)
    prob = PROBS * n
    alias = np.zeros(n, dtype=int)
    small = [i for i in range(n) if prob[i] < 1.0]
    large = [i for i in range(n) if prob[i] >= 1.0]
    while small and large:
        s, g = small.pop(), large[-1]
        alias[s] = g
        prob[g] += prob[s] - 1.0
        if prob[g] < 1.0:
            large.pop()
            small.append(g)
    for i in small + large:
        prob[i] = 1.0
    return prob, alias


def alias_sample(
    prob: np.ndarray, alias: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    n = len(prob)
    I = rng.integers(0, n, size=size)
    u = rng.uniform(size=size)
    return np.where(u <= prob[I], I, alias[I]) + 1


def rejection_sample_fast(size: int, rng: np.random.Generator) -> np.ndarray:
    k = len(PROBS)
    c = PROBS.max()
    results = np.empty(size, dtype=int)
    filled = 0
    while filled < size:
        batch = min((size - filled) * 4, 500_000)
        idx = rng.integers(0, k, size=batch)
        u = rng.uniform(size=batch)
        accepted = idx[u <= PROBS[idx] / c]
        take = min(len(accepted), size - filled)
        results[filled : filled + take] = accepted[:take] + 1
        filled += take
    return results


def chi_square_sixpoint(samples: np.ndarray) -> tuple[float, float]:
    observed = np.array([np.sum(samples == v) for v in VALUES], dtype=float)
    stat, pval = chisquare(observed, PROBS * len(samples))
    return stat, pval


def main() -> None:

    # Part 1: Geometric distribution
    logger.info("Starting geometric distribution simulation", seed=settings.SEED, n=N)

    with figure(
        figsize=(14, 4), save="exercises/day2/linus/discrete/plots/geometric.png"
    ) as fig:
        axes = fig.subplots(1, 3)

        for ax, p in zip(axes, [0.1, 0.3, 0.7]):
            samples = geometric_sample(p, N)
            stat, pval = chi_square_geometric(samples, p)
            logger.info(
                "Geometric sample summary",
                p=p,
                mean=round(float(samples.mean()), 3),
                theory=round(1 / p, 3),
                chi2=round(float(stat), 2),
                p_value=round(float(pval), 4),
            )

            max_k = int(np.percentile(samples, 99)) + 1
            k_vals = np.arange(1, max_k + 1)

            ax.bar(
                k_vals,
                np.array([np.sum(samples == k) for k in k_vals]) / N,
                alpha=0.7,
                label="simulated",
                width=0.6,
            )
            ax.plot(
                k_vals,
                geometric_pmf(k_vals, p),
                "ro-",
                markersize=4,
                linewidth=1,
                label="theoretical",
            )
            ax.set_title(f"Geometric(p={p})")
            ax.set_xlabel("k")
            ax.set_ylabel("probability")
            ax.legend()

        fig.suptitle("Geometric distribution, n=10,000")

    # Part 2: Six-point distribution
    logger.info("Starting six-point distribution simulation")
    prob_table, alias_table = alias_build()

    methods = {
        "crude": lambda r: crude_sample(N, r),
        "rejection": lambda r: rejection_sample(N, r),
        "alias": lambda r: alias_sample(prob_table, alias_table, N, r),
    }

    for seed in [42, 69]:
        seed_rng = np.random.default_rng(seed)
        for name, fn in methods.items():
            samples = fn(seed_rng)
            stat, pval = chi_square_sixpoint(samples)
            logger.info(
                "Six-point chi-square",
                method=name,
                seed=seed,
                chi2=round(float(stat), 4),
                p_value=round(float(pval), 4),
            )

            if seed == 69:
                with figure(
                    figsize=(7, 4),
                    save=f"exercises/day2/linus/discrete/plots/sixpoint_{name}.png",
                ) as fig:
                    ax = fig.add_subplot(111)
                    x = np.arange(len(VALUES))
                    w = 0.35
                    ax.bar(
                        x - w / 2,
                        np.array([np.sum(samples == v) for v in VALUES]) / N,
                        width=w,
                        label="simulated",
                        alpha=0.8,
                    )
                    ax.plot(x, PROBS, "ro-", markersize=5, label="theoretical")
                    ax.set_xticks(x)
                    ax.set_xticklabels(VALUES)
                    ax.set_xlabel("x")
                    ax.set_ylabel("probability")
                    ax.set_title(f"Six-point distribution — {name} (seed {seed})")
                    ax.legend()

    # Part 3: Method comparison at n=1,000,000
    N_LARGE = 1_000_000
    logger.info("Starting method comparison", n=N_LARGE)
    comp_rng = np.random.default_rng(settings.SEED)
    prob_table, alias_table = alias_build()

    comp_methods = {
        "crude": lambda: crude_sample(N_LARGE, comp_rng),
        "rejection": lambda: rejection_sample_fast(N_LARGE, comp_rng),
        "alias": lambda: alias_sample(prob_table, alias_table, N_LARGE, comp_rng),
    }

    names, times_list, chi2s, pvals = [], [], [], []
    for name, fn in comp_methods.items():
        t0 = time.perf_counter()
        samples = fn()
        elapsed = time.perf_counter() - t0
        stat, pval = chi_square_sixpoint(samples)
        names.append(name)
        times_list.append(elapsed)
        chi2s.append(stat)
        pvals.append(pval)
        logger.info(
            "Comparison result",
            method=name,
            chi2=round(float(stat), 4),
            p_value=round(float(pval), 4),
            time_s=round(elapsed, 4),
        )

    with figure(
        figsize=(6, 4), save="exercises/day2/linus/discrete/plots/method_timing.png"
    ) as fig:
        ax = fig.add_subplot(111)
        ax.bar(names, times_list, alpha=0.8)
        ax.set_ylabel("time (s)")
        ax.set_title(f"Sampling time, n={N_LARGE:,}")


if __name__ == "__main__":
    main()
