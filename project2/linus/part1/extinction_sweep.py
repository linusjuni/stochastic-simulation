"""Part I (a) — how the probability of disappearance depends on the parameters.

`extinction.py` answers the question for a single setting (R0=3, I0=1). This
script turns it into a *study* by sweeping the two parameters that branching-
process theory says control the minor-outbreak probability:

    P(disappear) = (1 / R0) ** I0           for R0 > 1   (and = 1 for R0 <= 1)

  * R0 sweep (fixed I0=1): P(disappear) should fall from 1 (at R0<=1, the disease
    cannot sustain itself) towards 0 as the disease becomes more contagious.
  * I0 sweep (fixed R0=3): each initial infective is an independent spark that
    must die out, so P(disappear) decays geometrically in I0 -- a straight line
    on a log axis.

Each point is a Monte-Carlo estimate with a Day-3 confidence interval, overlaid
on the theoretical curve.

Run:  uv run python -m project2.linus.part1.extinction_sweep
"""

import numpy as np

from project2.linus.analysis import confidence_interval, replicate
from project2.linus.models import SIR
from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

N = 1000
GAMMA = 0.1
T_MAX = 200.0
N_REPS = 1000
MINOR_FRACTION = 0.01  # minor outbreak if fewer than 1% of N are ever infected

# R0 sweep (I0 fixed at 1): includes the sub-critical regime R0 <= 1.
R0_GRID = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
R0_SWEEP_I0 = 1

# I0 sweep (R0 fixed at 3): one extra spark at a time. Capped at 5 -- beyond
# that P(disappear) is so small that 1000 reps gives 0 successes (undersampled).
I0_GRID = np.array([1, 2, 3, 4, 5])
I0_SWEEP_R0 = 3.0

logger = get_logger(__name__)


def p_disappear(beta: float, I0: int, rng: np.random.Generator):
    """Estimate P(disease disappears) for one (beta, I0) setting, with a CI."""
    trajectories = replicate(
        lambda s: SIR(S=N - I0, I=I0, beta=beta, gamma=GAMMA, rng=s),
        n=N_REPS,
        t_max=T_MAX,
        rng=rng,
    )
    final_sizes = np.array([traj.final("R") for traj in trajectories])
    disappeared = (final_sizes < MINOR_FRACTION * N).astype(float)
    return confidence_interval(disappeared)


def theory(R0: float, I0: int) -> float:
    """Branching-process minor-outbreak probability."""
    return 1.0 if R0 <= 1.0 else (1.0 / R0) ** I0


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)

    # --- R0 sweep (I0 = 1) ------------------------------------------------
    logger.info("R0 sweep", points=len(R0_GRID), reps_each=N_REPS, I0=R0_SWEEP_I0)
    r0_est = [p_disappear(R0 * GAMMA, R0_SWEEP_I0, rng) for R0 in R0_GRID]
    r0_mean = np.array([e.mean for e in r0_est])
    r0_err = np.array([e.half_width for e in r0_est])
    r0_theory = np.array([theory(R0, R0_SWEEP_I0) for R0 in R0_GRID])
    for R0, e, th in zip(R0_GRID, r0_est, r0_theory):
        logger.success("R0 point", R0=float(R0), estimate=str(e), theory=round(th, 3))

    # --- I0 sweep (R0 = 3) ------------------------------------------------
    beta_fixed = I0_SWEEP_R0 * GAMMA
    logger.info("I0 sweep", points=len(I0_GRID), reps_each=N_REPS, R0=I0_SWEEP_R0)
    i0_est = [p_disappear(beta_fixed, int(I0), rng) for I0 in I0_GRID]
    i0_mean = np.array([e.mean for e in i0_est])
    i0_err = np.array([e.half_width for e in i0_est])
    i0_theory = np.array([theory(I0_SWEEP_R0, int(I0)) for I0 in I0_GRID])
    for I0, e, th in zip(I0_GRID, i0_est, i0_theory):
        logger.success("I0 point", I0=int(I0), estimate=str(e), theory=round(th, 4))

    # --- plot -------------------------------------------------------------
    with figure(
        figsize=(11, 4.5), save="project2/linus/plots/part1/extinction_sweep.png"
    ) as fig:
        ax_r0 = fig.add_subplot(121)
        r0_fine = np.linspace(1.0, R0_GRID.max(), 200)
        ax_r0.plot(
            np.concatenate([[R0_GRID.min()], [1.0], r0_fine]),
            np.concatenate([[1.0], [1.0], (1.0 / r0_fine) ** R0_SWEEP_I0]),
            color="black",
            linestyle="--",
            label=r"theory $(1/R_0)^{I_0}$",
        )
        ax_r0.errorbar(
            R0_GRID, r0_mean, yerr=r0_err, fmt="o", color="tab:red",
            capsize=3, label="simulation (95% CI)",
        )
        ax_r0.axvline(1.0, color="grey", linestyle=":", linewidth=1)
        ax_r0.set_title(f"P(disappear) vs $R_0$   ($I_0$={R0_SWEEP_I0})")
        ax_r0.set_xlabel("$R_0 = \\beta / \\gamma$")
        ax_r0.set_ylabel("P(disease disappears)")
        ax_r0.set_ylim(-0.05, 1.05)
        ax_r0.legend()

        ax_i0 = fig.add_subplot(122)
        i0_fine = np.linspace(I0_GRID.min(), I0_GRID.max(), 200)
        ax_i0.plot(
            i0_fine, (1.0 / I0_SWEEP_R0) ** i0_fine,
            color="black", linestyle="--", label=r"theory $(1/R_0)^{I_0}$",
        )
        ax_i0.errorbar(
            I0_GRID, i0_mean, yerr=i0_err, fmt="o", color="tab:blue",
            capsize=3, label="simulation (95% CI)",
        )
        ax_i0.set_yscale("log")
        ax_i0.set_title(f"P(disappear) vs $I_0$   ($R_0$={I0_SWEEP_R0:.0f})")
        ax_i0.set_xlabel("Initial infectives $I_0$")
        ax_i0.set_ylabel("P(disease disappears)  (log scale)")
        ax_i0.legend()

        fig.suptitle(
            f"Probability of disappearance vs parameters "
            f"({N_REPS} runs/point, N={N})"
        )
