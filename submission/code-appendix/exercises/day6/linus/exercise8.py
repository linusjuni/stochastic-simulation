import numpy as np

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

SEED = settings.SEED
PLOTS = "exercises/day6/linus/plots"


def bootstrap_replicates(data: np.ndarray, statistic, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(data)
    samples = rng.choice(data, size=(k, n), replace=True)
    return np.apply_along_axis(statistic, 1, samples)


def bootstrap_variance(data: np.ndarray, statistic, k: int, rng: np.random.Generator) -> tuple[float, float]:
    replicates = bootstrap_replicates(data, statistic, k, rng)
    return float(statistic(data)), float(replicates.var(ddof=1))


def pareto_sample(beta: float, k: float, size: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(size=size)
    return beta * u ** (-1.0 / k)


def part1_ross13(rng: np.random.Generator) -> None:
    data = np.array([56, 101, 78, 67, 93, 87, 64, 72, 80, 69], dtype=float)
    a, b, k = -5.0, 5.0, 10_000
    xbar = data.mean()

    reps = bootstrap_replicates(data, np.mean, k, rng)
    diff = reps - xbar
    p_hat = float(np.mean((a < diff) & (diff < b)))

    logger.info("Part 1: Ross Ex. 13 — bootstrap estimate of p = P{a < mean(X) - mu < b}")
    logger.info("Sample mean", xbar=round(float(xbar), 4))
    logger.info("Bootstrap p_hat", p_hat=round(p_hat, 4))

    with figure(figsize=(8, 5), save=f"{PLOTS}/ex13_bootstrap_mean.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(diff, bins=40, edgecolor="black", linewidth=0.6)
        ax.axvspan(a, b, alpha=0.2, color="orange", label=f"[{a:.0f}, {b:.0f}]")
        ax.set_xlabel(r"$\bar{X}^* - \bar{x}$")
        ax.set_ylabel("Count")
        ax.set_title(f"Bootstrap distribution of $\\bar{{X}}^* - \\bar{{x}}$ (k={k:,})")
        ax.legend()


def part2_ross15(rng: np.random.Generator) -> None:
    data = np.array([5, 4, 9, 6, 21, 17, 11, 20, 7, 10, 21, 15, 13, 16, 8], dtype=float)
    k = 10_000
    sample_var = lambda x: x.var(ddof=1)

    s2, var_hat = bootstrap_variance(data, sample_var, k, rng)

    logger.info("Part 2: Ross Ex. 15 — bootstrap estimate of Var(S^2)")
    logger.info("Observed S^2", s2=round(s2, 4))
    logger.info("Bootstrap Var_hat(S^2)", var_hat=round(var_hat, 4))


def part3_pareto(rng: np.random.Generator) -> None:
    data = pareto_sample(1.0, 1.05, 200, rng)
    k = 100

    mean_hat, mean_var = bootstrap_variance(data, np.mean, k, rng)
    median_hat, median_var = bootstrap_variance(data, np.median, k, rng)

    logger.info("Part 3: Pareto(beta=1, k=1.05), n=200 — sample mean vs. sample median")
    logger.info(
        "Sample mean",
        estimate=round(mean_hat, 4),
        boot_var=round(mean_var, 4),
        boot_std_err=round(float(np.sqrt(mean_var)), 4),
    )
    logger.info(
        "Sample median",
        estimate=round(median_hat, 4),
        boot_var=round(median_var, 4),
        boot_std_err=round(float(np.sqrt(median_var)), 4),
    )
    logger.info("Variance ratio mean/median", ratio=round(mean_var / median_var, 1))


def main() -> None:
    rng = np.random.default_rng(SEED)
    part1_ross13(rng)
    part2_ross15(rng)
    part3_pareto(rng)


if __name__ == "__main__":
    main()
