"""Part I (a) — finite-population correction to the branching-process theory.

`a.tex` notes two places where the simulated P(disappear) deviates from the
branching prediction (1 / R0)^I0, both because the formula assumes an *endless*
supply of susceptibles:

  * Near R0 = 1 (e.g. R0 = 1.2) the simulation sits a little *below* theory:
    outbreaks are slow and the finite population makes chains die out more often.
  * For R0 < 1 (e.g. R0 = 0.8) the estimate is not exactly 1, because a few runs
    cross the 1%-of-N threshold by chance.

Both are claimed to shrink as N grows. This script demonstrates that directly:
fix R0 and I0, sweep the population size N, and watch P(disappear) converge onto
the (population-independent) theory line.

Run:  uv run python -m project2.linus.part1.extinction_finite_n
"""

import numpy as np

from project2.linus.analysis import confidence_interval, replicate
from project2.linus.models import SIR
from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

GAMMA = 0.1
I0 = 1
T_MAX = 400.0  # generous: near-critical outbreaks are slow
N_REPS = 1000
MINOR_FRACTION = 0.01  # minor outbreak if fewer than 1% of N are ever infected

# Population grid (log-spaced) and the two R0 cases a.tex calls out.
N_GRID = np.array([250, 500, 1000, 2000, 4000, 8000])
R0_CASES = (0.8, 1.2)

logger = get_logger(__name__)


def p_disappear(N: int, R0: float, rng: np.random.Generator):
    """Estimate P(disease disappears) for one (N, R0) setting, with a CI."""
    beta = R0 * GAMMA
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
    """Branching-process minor-outbreak probability (no finite-N correction)."""
    return 1.0 if R0 <= 1.0 else (1.0 / R0) ** I0


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)

    results: dict[float, list] = {}
    for R0 in R0_CASES:
        logger.info("N sweep", R0=R0, points=len(N_GRID), reps_each=N_REPS)
        ests = [p_disappear(int(N), R0, rng) for N in N_GRID]
        results[R0] = ests
        for N, e in zip(N_GRID, ests):
            logger.success(
                "point", R0=R0, N=int(N), estimate=str(e), theory=round(theory(R0, I0), 3)
            )

    with figure(
        figsize=(7, 4.5), save="project2/linus/plots/part1/extinction_finite_n.png"
    ) as fig:
        ax = fig.add_subplot(111)
        colors = {0.8: "tab:blue", 1.2: "tab:red"}
        for R0 in R0_CASES:
            ests = results[R0]
            means = np.array([e.mean for e in ests])
            errs = np.array([e.half_width for e in ests])
            ax.errorbar(
                N_GRID, means, yerr=errs, fmt="o-", color=colors[R0],
                capsize=3, label=f"simulation, $R_0$={R0}",
            )
            ax.axhline(
                theory(R0, I0), color=colors[R0], linestyle="--", linewidth=1,
                label=f"theory $R_0$={R0}: {theory(R0, I0):.3f}",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Population size $N$  (log scale)")
        ax.set_ylabel("P(disease disappears)")
        ax.set_title(
            f"Finite-$N$ convergence to branching theory "
            f"($I_0$={I0}, {N_REPS} runs/point)"
        )
        ax.set_ylim(0.55, 1.02)
        ax.legend(fontsize=8)
