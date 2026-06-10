import numpy as np
from scipy import stats

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)


def lcg(n: int, a: int, c: int, m: int, seed: int) -> np.ndarray:
    """Return n samples from U(0,1) using the recurrence x_i = (a*x + c) mod m."""
    x = seed
    samples = np.empty(n)
    logger.info("Starting LCG generation", n=n, a=a, c=c, m=m, seed=seed)
    for i in range(n):
        x = (a * x + c) % m
        samples[i] = x / m
    return samples


def run_test(samples: np.ndarray) -> tuple[float, float]:
    """Wald-Wolfowitz runs test above/below median. Returns (z_statistic, p_value)."""
    median = np.median(samples)
    above = samples > median
    n1 = above.sum()
    n2 = len(samples) - n1

    runs = 1 + np.sum(above[:-1] != above[1:])
    mean = (2 * n1 * n2) / (n1 + n2) + 1
    var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mean) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def correlation_test(
    samples: np.ndarray, lags: list[int]
) -> dict[int, tuple[float, float]]:
    """Serial correlation test at each lag h per lecture definition.

    c_h = mean(U_i * U_{i+h}) ~ Normal(0.25, 7/(144n)) under H0.
    Returns {h: (c_h, p_value)}.
    """
    n = len(samples)
    result = {}
    for h in lags:
        c_h = float(np.mean(samples[: n - h] * samples[h:]))
        std = np.sqrt(7 / (144 * (n - h)))
        z = (c_h - 0.25) / std
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        result[h] = (c_h, float(p))
    return result


def run_all_tests(samples: np.ndarray, lags: list[int] | None = None) -> None:
    """Run Chi-squared, KS, run test, and correlation test on samples and log results."""
    if lags is None:
        lags = [1, 2, 3, 5, 10]

    # Chi-squared
    n = len(samples)
    bins = 10
    observed, _ = np.histogram(samples, bins=bins, range=(0, 1))
    expected = np.full(bins, n / bins)
    chi2_stat, chi2_p = stats.chisquare(observed, expected)
    logger.info(
        "Chi-squared test", statistic=round(chi2_stat, 4), p_value=round(chi2_p, 4)
    )

    # Kolmogorov-Smirnov
    ks_stat, ks_p = stats.kstest(samples, "uniform")
    logger.info("KS test", statistic=round(ks_stat, 4), p_value=round(ks_p, 4))

    # Run test
    z, run_p = run_test(samples)
    logger.info("Run test (above/below median)", z=round(z, 4), p_value=round(run_p, 4))

    # Correlation test
    for h, (c_h, corr_p) in correlation_test(samples, lags).items():
        logger.info(
            "Correlation test", lag=h, c_h=round(c_h, 4), p_value=round(corr_p, 4)
        )


if __name__ == "__main__":
    # Numerical Recipes parameters — satisfy Hull-Dobell for full period 2^32
    a, c, m = 1664525, 1013904223, 2**32
    samples = lcg(10_000, a, c, m, seed=settings.SEED)
    logger.info("Generated LCG samples", n=10_000, a=a, c=c, m=m, seed=settings.SEED)

    run_all_tests(samples)

    with figure(figsize=(7, 4), save="exercises/day1/linus/lcg_histogram.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(samples, bins=10, edgecolor="white")
        ax.set_xlabel("U")
        ax.set_ylabel("Count")
        ax.set_title("LCG histogram (n=10,000)")

    with figure(figsize=(6, 6), save="exercises/day1/linus/lcg_scatter.png") as fig:
        ax = fig.add_subplot(111)
        ax.scatter(samples[:-1], samples[1:], s=1, alpha=0.3)
        ax.set_xlabel("$U_i$")
        ax.set_ylabel("$U_{i+1}$")
        ax.set_title("Consecutive pairs (n=10,000)")

    # Exercise 2: system generator
    rng = np.random.default_rng(settings.SEED)
    system_samples = rng.uniform(0, 1, 10_000)
    logger.info("Testing numpy system generator (PCG64)")
    run_all_tests(system_samples)
