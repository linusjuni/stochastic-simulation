import numpy as np

from utils.logger import get_logger
from utils.plotting import figure, histogram
from utils.settings import settings

logger = get_logger(__name__)

# States: 0=NED, 1=local recurrence, 2=distant metastasis, 3=both, 4=death
P = np.array(
    [
        [0.9915, 0.005, 0.0025, 0.0, 0.001],
        [0.0, 0.986, 0.005, 0.004, 0.005],
        [0.0, 0.0, 0.992, 0.003, 0.005],
        [0.0, 0.0, 0.0, 0.991, 0.009],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)

DEATH = 4
P_S = P[:4, :4]  # transient sub-matrix (states 1-4)
PI = np.array([1.0, 0.0, 0.0, 0.0])  # everyone starts in state 1 (NED)

BATCH_SIZE = 200  # women per Monte Carlo replication
N_REPLICATIONS = 100  # number of independent replications
DEATH_WITHIN = 350  # threshold (months) for the fraction we're estimating


def _sample_next_state(rng: np.random.Generator, probs: np.ndarray) -> int:
    """Draw one state index via the inverse-CDF method, given a row of P."""
    u = rng.random()
    cumulative = np.cumsum(probs)
    return int(np.searchsorted(cumulative, u))


def simulate_lifetime(rng: np.random.Generator) -> int:
    """Simulate one woman from state 1 until death, return only her lifetime."""
    state = 0
    t = 0
    while state != DEATH:
        state = _sample_next_state(rng, P[state])
        t += 1
    return t


def simulate_population_lifetimes(n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate n independent women, return their lifetimes."""
    return np.array([simulate_lifetime(rng) for _ in range(n)])


def theoretical_mean_lifetime() -> float:
    """E(T) = pi (I - P_s)^-1 1, the expected lifetime under the model."""
    expected_steps = np.linalg.solve(np.eye(4) - P_S, np.ones(4))
    return float(PI @ expected_steps)


def crude_fraction_estimate(lifetimes: np.ndarray, threshold: int) -> float:
    """Crude Monte Carlo estimate of P(T <= threshold): the sample fraction of deaths by then."""
    return float(np.mean(lifetimes <= threshold))


def control_variate_fraction_estimate(
    lifetimes: np.ndarray, threshold: int, mu_z: float
) -> float:
    """Control-variate estimate of P(T <= threshold), using lifetime itself as the control.

    Following the lecture's Y_i = X_i + c(Z_i - mu_Z) with c = -Cov(X_i, Z_i) / Var(Z_i),
    where X_i is the death-by-threshold indicator and Z_i is the lifetime (known mean mu_z).
    Cov and Var are estimated empirically from the same batch, as the lecture notes is
    standard practice when the true covariance is unknown.
    """
    indicators = (lifetimes <= threshold).astype(float)
    cov_xz = np.cov(indicators, lifetimes, ddof=1)[0, 1]
    var_z = np.var(lifetimes, ddof=1)
    c = -cov_xz / var_z
    y = indicators + c * (lifetimes - mu_z)
    return float(np.mean(y))


def confidence_interval(estimates: np.ndarray) -> tuple[float, float, float]:
    """95% CI for the mean of a set of estimates, returned as (mean, ci_low, ci_high)."""
    mean = estimates.mean()
    std_error = estimates.std(ddof=1) / np.sqrt(len(estimates))
    return float(mean), float(mean - 1.96 * std_error), float(mean + 1.96 * std_error)


def run_replications(
    n_replications: int,
    batch_size: int,
    threshold: int,
    mu_z: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Run independent replications, each estimating P(T <= threshold) two ways.

    Returns the crude and control-variate estimates, one per replication.
    """
    crude_estimates = np.empty(n_replications)
    cv_estimates = np.empty(n_replications)
    for i in range(n_replications):
        lifetimes = simulate_population_lifetimes(batch_size, rng)
        crude_estimates[i] = crude_fraction_estimate(lifetimes, threshold)
        cv_estimates[i] = control_variate_fraction_estimate(lifetimes, threshold, mu_z)
    return crude_estimates, cv_estimates


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    mu_z = theoretical_mean_lifetime()
    logger.info(
        "Running replications for Task 5",
        n_replications=N_REPLICATIONS,
        batch_size=BATCH_SIZE,
        death_within=DEATH_WITHIN,
        mu_z=round(mu_z, 1),
    )

    crude_estimates, cv_estimates = run_replications(
        N_REPLICATIONS, BATCH_SIZE, DEATH_WITHIN, mu_z, rng
    )

    var_crude = np.var(crude_estimates, ddof=1)
    var_cv = np.var(cv_estimates, ddof=1)
    variance_reduction = 1.0 - var_cv / var_crude

    mean_crude, ci_low_crude, ci_high_crude = confidence_interval(crude_estimates)
    mean_cv, ci_low_cv, ci_high_cv = confidence_interval(cv_estimates)

    logger.success(
        "Crude Monte Carlo estimator",
        mean=round(mean_crude, 4),
        ci_95=(round(ci_low_crude, 4), round(ci_high_crude, 4)),
        variance=round(float(var_crude), 6),
    )
    logger.success(
        "Control variate estimator",
        mean=round(mean_cv, 4),
        ci_95=(round(ci_low_cv, 4), round(ci_high_cv, 4)),
        variance=round(float(var_cv), 6),
    )
    logger.success(
        "Variance reduction",
        factor=round(float(var_crude / var_cv), 2),
        percent=round(float(variance_reduction * 100), 1),
    )

    with figure(figsize=(8, 5), save="project1/plots/task5/estimator_comparison.png") as fig:
        ax = fig.add_subplot(111)
        histogram(
            crude_estimates,
            ax=ax,
            bins=15,
            title="Crude vs. control-variate estimates of $P(T \\leq 350)$",
            xlabel="Estimated fraction",
            label="Crude Monte Carlo",
        )
        histogram(
            cv_estimates,
            ax=ax,
            bins=15,
            xlabel="Estimated fraction",
            label="Control variate",
        )
        ax.legend()
