import numpy as np
from scipy import stats

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

N = 10_000
LAGS = [1, 2, 5]
PLOTS = "report/plots/day1"

A_GOOD, C_GOOD, M_GOOD = 1_664_525, 1_013_904_223, 2**32
A_BAD, C_BAD, M_BAD = 5, 1, 16


def lcg(n: int, a: int, c: int, m: int, seed: int) -> np.ndarray:
    x = seed
    samples = np.empty(n)
    for i in range(n):
        x = (a * x + c) % m
        samples[i] = x / m
    return samples


def run_test(samples: np.ndarray) -> tuple[float, float]:
    """Wald-Wolfowitz runs test above/below median."""
    median = np.median(samples)
    above = samples > median
    n1 = int(above.sum())
    n2 = len(samples) - n1
    runs = 1 + int(np.sum(above[:-1] != above[1:]))
    mean = (2 * n1 * n2) / (n1 + n2) + 1
    var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mean) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def pearson_correlation(samples: np.ndarray, lags: list[int]) -> dict[int, float]:
    """Pearson serial correlation coefficient r_h at each lag."""
    return {h: float(np.corrcoef(samples[:-h], samples[h:])[0, 1]) for h in lags}


def correlation_test(
    samples: np.ndarray, lags: list[int]
) -> dict[int, tuple[float, float]]:
    """Lecture c_h test: c_h = mean(U_i * U_{i+h}) ~ Normal(1/4, 7/(144n)) under H0."""
    n = len(samples)
    result = {}
    for h in lags:
        c_h = float(np.mean(samples[: n - h] * samples[h:]))
        std = np.sqrt(7 / (144 * (n - h)))
        z = (c_h - 0.25) / std
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        result[h] = (c_h, float(p))
    return result


def run_all_tests(samples: np.ndarray, label: str) -> None:
    n = len(samples)

    bins = 10
    observed, _ = np.histogram(samples, bins=bins, range=(0, 1))
    chi2_stat, chi2_p = stats.chisquare(observed, np.full(bins, n / bins))
    logger.info(
        f"{label} chi-squared",
        statistic=round(chi2_stat, 4),
        p_value=round(chi2_p, 4),
        reject=chi2_stat > 16.9,
    )

    ks_stat, ks_p = stats.kstest(samples, "uniform")
    logger.info(
        f"{label} KS",
        statistic=round(ks_stat, 4),
        p_value=round(ks_p, 4),
        reject=ks_stat > 0.0136,
    )

    z, run_p = run_test(samples)
    logger.info(
        f"{label} runs", z=round(z, 4), p_value=round(run_p, 4), reject=abs(z) > 1.96
    )

    threshold = 1.96 / np.sqrt(n)
    for h, r in pearson_correlation(samples, LAGS).items():
        logger.info(
            f"{label} Pearson r_h", lag=h, r=round(r, 4), reject=abs(r) > threshold
        )

    for h, (c_h, p) in correlation_test(samples, LAGS).items():
        z_h = (c_h - 0.25) / np.sqrt(7 / (144 * (n - h)))
        logger.info(
            f"{label} c_h test",
            lag=h,
            c_h=round(c_h, 6),
            z=round(z_h, 4),
            p_value=round(p, 4),
            reject=abs(z_h) > 1.96,
        )


def multi_seed_rejection_rates(n_seeds: int = 1_000) -> None:
    """Rejection rates over n_seeds independent seeds for good LCG and MT."""
    keys = ["chi2", "ks", "runs", "ch1", "ch2", "ch5"]
    lcg_rej = {k: 0 for k in keys}
    mt_rej = {k: 0 for k in keys}
    exp = np.full(10, N / 10)

    for seed in range(n_seeds):
        for samp, rej in [
            (lcg(N, A_GOOD, C_GOOD, M_GOOD, seed), lcg_rej),
            (np.random.RandomState(seed).uniform(size=N), mt_rej),
        ]:
            chi2, _ = stats.chisquare(np.histogram(samp, bins=10, range=(0, 1))[0], exp)
            if chi2 > 16.9:
                rej["chi2"] += 1
            ks, _ = stats.kstest(samp, "uniform")
            if ks > 0.0136:
                rej["ks"] += 1
            z, _ = run_test(samp)
            if abs(z) > 1.96:
                rej["runs"] += 1
            for key, h in [("ch1", 1), ("ch2", 2), ("ch5", 5)]:
                c_h = float(np.mean(samp[:-h] * samp[h:]))
                z_h = (c_h - 0.25) / np.sqrt(7 / (144 * (N - h)))
                if abs(z_h) > 1.96:
                    rej[key] += 1

    def pct(d: dict, k: str) -> str:
        return f"{d[k] / n_seeds * 100:.1f}%"

    for lbl, rej in [("Good LCG", lcg_rej), ("Mersenne Twister", mt_rej)]:
        logger.info(
            f"Multi-seed rejection rates ({lbl}), n_seeds={n_seeds}",
            chi2=pct(rej, "chi2"),
            ks=pct(rej, "ks"),
            runs=pct(rej, "runs"),
            ch_h1=pct(rej, "ch1"),
            ch_h2=pct(rej, "ch2"),
            ch_h5=pct(rej, "ch5"),
        )


if __name__ == "__main__":
    # --- Part 1a: Good LCG ---
    logger.info(
        "Part 1a: Good LCG histogram", a=A_GOOD, c=C_GOOD, m=M_GOOD, seed=settings.SEED
    )
    samples = lcg(N, A_GOOD, C_GOOD, M_GOOD, seed=settings.SEED)

    with figure(figsize=(7, 4), save=f"{PLOTS}/good_lcg_histogram.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(samples, bins=10, edgecolor="white")
        ax.set_xlabel("U")
        ax.set_ylabel("Count")
        ax.set_title(f"Good LCG histogram (n={N:,})")

    # --- Part 1b: Statistical tests on good LCG ---
    logger.info("Part 1b: Good LCG tests")
    run_all_tests(samples, "Good LCG")

    with figure(figsize=(6, 6), save=f"{PLOTS}/good_lcg_scatter.png") as fig:
        ax = fig.add_subplot(111)
        ax.scatter(samples[:-1], samples[1:], s=1, alpha=0.3)
        ax.set_xlabel("$U_i$")
        ax.set_ylabel("$U_{i+1}$")
        ax.set_title(f"Good LCG consecutive pairs (n={N:,})")

    # --- Part 1c: Bad LCG comparison ---
    logger.info("Part 1c: Bad LCG", a=A_BAD, c=C_BAD, m=M_BAD, seed=settings.SEED)
    bad_samples = lcg(N, A_BAD, C_BAD, M_BAD, seed=settings.SEED)
    run_all_tests(bad_samples, "Bad LCG")

    with figure(figsize=(7, 4), save=f"{PLOTS}/bad_lcg_histogram.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(bad_samples, bins=10, edgecolor="white")
        ax.set_xlabel("U")
        ax.set_ylabel("Count")
        ax.set_title(f"Bad LCG histogram — a={A_BAD}, c={C_BAD}, M={M_BAD} (n={N:,})")

    with figure(figsize=(6, 6), save=f"{PLOTS}/bad_lcg_scatter.png") as fig:
        ax = fig.add_subplot(111)
        ax.scatter(bad_samples[:-1], bad_samples[1:], s=1, alpha=0.3)
        ax.set_xlabel("$U_i$")
        ax.set_ylabel("$U_{i+1}$")
        ax.set_title(
            f"Bad LCG consecutive pairs — a={A_BAD}, c={C_BAD}, M={M_BAD} (n={N:,})"
        )

    # --- Part 2: System generator (Mersenne Twister via numpy.RandomState) ---
    logger.info("Part 2: Mersenne Twister system generator", seed=settings.SEED)
    system_samples = np.random.RandomState(settings.SEED).uniform(size=N)
    run_all_tests(system_samples, "Mersenne Twister")

    with figure(figsize=(7, 4), save=f"{PLOTS}/system_histogram.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(system_samples, bins=10, edgecolor="white")
        ax.set_xlabel("U")
        ax.set_ylabel("Count")
        ax.set_title(f"Mersenne Twister histogram (n={N:,})")

    with figure(figsize=(6, 6), save=f"{PLOTS}/system_scatter.png") as fig:
        ax = fig.add_subplot(111)
        ax.scatter(system_samples[:-1], system_samples[1:], s=1, alpha=0.3)
        ax.set_xlabel("$U_i$")
        ax.set_ylabel("$U_{i+1}$")
        ax.set_title(f"Mersenne Twister consecutive pairs (n={N:,})")

    # --- Part 3: Multi-seed rejection rates ---
    logger.info("Part 3: Multi-seed rejection rates (1,000 seeds)")
    multi_seed_rejection_rates(n_seeds=1_000)
