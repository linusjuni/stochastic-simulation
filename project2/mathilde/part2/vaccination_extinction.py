"""Part II (d) -- effect of vaccination on the probability the disease
disappears.

Extends the Part I branching-process result (1/R0)^I0 to a leaky vaccine: a
single infective's contacts land on S with probability (1-p) and on V with
probability p, infecting at rate beta and (1-ve)*beta respectively. Collapsing
the two infection channels into one effective rate beta_eff = beta*(1-p*ve)
reduces the problem to the plain-SIR self-consistency equation with
beta -> beta_eff, giving an effective reproduction number

    Rv = R0 * (1 - p*ve)

and the same closed forms as plain SIR with R0 -> Rv:

    P(disease disappears) = (1/Rv)^I0          (Rv > 1)
                           = 1                  (Rv <= 1)
    critical coverage:      v_c(ve) = (1 - 1/R0) / ve

Run:  uv run python -m project2.mathilde.part2.vaccination_extinction
"""

import numpy as np

from project2.linus.analysis import confidence_interval, replicate
from project2.mathilde.models import VaccinatedSIR
from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

N = 1000
I0 = 1
BETA = 0.3
GAMMA = 0.1  # R0 = beta / gamma = 3
T_MAX = 200.0
MINOR_FRACTION = 0.01  # minor outbreak if fewer than 1% of N are ever infected
R0 = BETA / GAMMA

# Sweep 1: P(disappear) vs effectiveness, fixed coverage
COVERAGE_FIXED = 0.6
VE_SWEEP = np.linspace(0.0, 1.0, 11)
N_REPS_VE = 2000

# Sweep 2: P(disappear) vs coverage, fixed effectiveness
VE_FIXED = 0.7
COVERAGE_SWEEP = np.linspace(0.0, 1.0, 11)
N_REPS_COVERAGE = 1000

logger = get_logger(__name__)


def rv(coverage: float, ve: float) -> float:
    return R0 * (1.0 - coverage * ve)


def theory_disappear(coverage: float, ve: float) -> float:
    r_v = rv(coverage, ve)
    return (1.0 / r_v) ** I0 if r_v > 1.0 else 1.0


def split_population(coverage: float) -> tuple[int, int]:
    s0 = round((1 - coverage) * (N - I0))
    v0 = (N - I0) - s0
    return s0, v0


def run_sweep(
    *, coverage_values, ve_values, n_reps: int, rng: np.random.Generator
):
    """Run one replication batch per (coverage, ve) pair (one of the two is
    a fixed scalar broadcast against the other, which is the swept array).
    Returns parallel arrays: estimates, theory, and the raw final sizes per
    point (for the attack-rate plot, at zero extra simulation cost)."""
    estimates = []
    theory = []
    final_sizes_per_point = []
    for coverage, ve in zip(coverage_values, ve_values):
        s0, v0 = split_population(coverage)

        def make_model(stream: np.random.Generator) -> VaccinatedSIR:
            return VaccinatedSIR(
                S=s0, V=v0, I=I0, beta=BETA, gamma=GAMMA, ve=ve, rng=stream
            )

        trajectories = replicate(make_model, n=n_reps, t_max=T_MAX, rng=rng)
        final_sizes = np.array([traj.final("R") for traj in trajectories])
        disappeared = (final_sizes < MINOR_FRACTION * N).astype(float)

        estimates.append(confidence_interval(disappeared))
        theory.append(theory_disappear(coverage, ve))
        final_sizes_per_point.append(final_sizes)

    return estimates, theory, final_sizes_per_point


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)

    logger.info(
        "Sweep 1: P(disappear) vs effectiveness",
        coverage=COVERAGE_FIXED,
        n_points=len(VE_SWEEP),
        n_reps=N_REPS_VE,
    )
    est_ve, theory_ve, final_sizes_ve = run_sweep(
        coverage_values=np.full_like(VE_SWEEP, COVERAGE_FIXED),
        ve_values=VE_SWEEP,
        n_reps=N_REPS_VE,
        rng=rng,
    )
    ve_c = (1 - 1 / R0) / COVERAGE_FIXED if COVERAGE_FIXED > 0 else float("inf")
    logger.success("Sweep 1 done", ve_critical=round(ve_c, 3))

    logger.info(
        "Sweep 2: P(disappear) vs coverage",
        ve=VE_FIXED,
        n_points=len(COVERAGE_SWEEP),
        n_reps=N_REPS_COVERAGE,
    )
    est_cov, theory_cov, _ = run_sweep(
        coverage_values=COVERAGE_SWEEP,
        ve_values=np.full_like(COVERAGE_SWEEP, VE_FIXED),
        n_reps=N_REPS_COVERAGE,
        rng=rng,
    )
    coverage_c = (1 - 1 / R0) / VE_FIXED
    logger.success("Sweep 2 done", coverage_critical=round(coverage_c, 3))

    means_ve = np.array([e.mean for e in est_ve])
    lo_ve = np.array([e.lower for e in est_ve])
    hi_ve = np.array([e.upper for e in est_ve])

    means_cov = np.array([e.mean for e in est_cov])
    lo_cov = np.array([e.lower for e in est_cov])
    hi_cov = np.array([e.upper for e in est_cov])

    with figure(
        figsize=(12, 5),
        save="project2/mathilde/plots/part2/vaccination_extinction_sweeps.png",
    ) as fig:
        ax_ve = fig.add_subplot(121)
        ax_ve.errorbar(
            VE_SWEEP,
            means_ve,
            yerr=[means_ve - lo_ve, hi_ve - means_ve],
            fmt="o",
            color="tab:blue",
            label="simulation (95% CI)",
            capsize=3,
        )
        ax_ve.plot(VE_SWEEP, theory_ve, "--", color="black", label=r"theory $(1/R_v)^{I_0}$")
        ax_ve.axvline(ve_c, color="red", linestyle=":", label=r"critical value ($R_v$ = 1)")
        ax_ve.set_xlabel("Vaccine effectiveness, ve")
        ax_ve.set_ylabel("P(disease disappears)")

        ax_cov = fig.add_subplot(122)
        ax_cov.errorbar(
            COVERAGE_SWEEP,
            means_cov,
            yerr=[means_cov - lo_cov, hi_cov - means_cov],
            fmt="o",
            color="tab:blue",
            label="simulation (95% CI)",
            capsize=3,
        )
        ax_cov.plot(COVERAGE_SWEEP, theory_cov, "--", color="black", label=r"theory $(1/R_v)^{I_0}$")
        ax_cov.axvline(coverage_c, color="red", linestyle=":", label=r"critical value ($R_v$ = 1)")
        ax_cov.set_xlabel("Vaccination coverage, p")
        ax_cov.set_ylabel("P(disease disappears)")

        handles, labels = ax_ve.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)

    attack_mean = np.array([fs.mean() / N for fs in final_sizes_ve])
    attack_lo = np.array(
        [confidence_interval(fs / N).lower for fs in final_sizes_ve]
    )
    attack_hi = np.array(
        [confidence_interval(fs / N).upper for fs in final_sizes_ve]
    )

    with figure(
        figsize=(8, 5),
        save="project2/mathilde/plots/part2/vaccination_attack_rate_vs_ve.png",
    ) as fig:
        ax = fig.add_subplot(111)
        ax.errorbar(
            VE_SWEEP,
            attack_mean,
            yerr=[attack_mean - attack_lo, attack_hi - attack_mean],
            fmt="o-",
            color="tab:red",
            capsize=3,
        )
        ax.set_xlabel("Vaccine effectiveness, ve")
        ax.set_ylabel("Mean attack rate, final(R)/N")
        ax.set_title(f"Outbreak size flattens with vaccine effectiveness (coverage={COVERAGE_FIXED:.0%})")
