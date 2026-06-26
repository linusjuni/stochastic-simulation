# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for synthesizing, code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

import numpy as np

from project2.theo.analysis import confidence_interval, replicate
from project2.theo.models import SIRD
from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

N = 1000
I0 = 10
REMOVAL_RATE = 0.1  # r = gamma + mu  (mean infectious period = 10 time units)
R0_GRID = (1.5, 2.0, 3.0, 5.0, 8.0)
CFR_GRID = (0.5, 0.8, 0.9, 0.95, 0.99)
N_REPS = 300
T_MAX = 2000.0
COLLAPSE_FRACTION = 0.95  # society has "collapsed" once >= 95% have died


def params_from(R0: float, cfr: float, r: float = REMOVAL_RATE):
    """(R0, CFR) -> (beta, gamma, mu) at fixed removal rate r."""
    mu = cfr * r
    gamma = (1.0 - cfr) * r
    beta = R0 * r
    return beta, gamma, mu


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)

    n_cfr, n_r0 = len(CFR_GRID), len(R0_GRID)
    frac_dead = np.zeros((n_cfr, n_r0))
    p_collapse = np.zeros((n_cfr, n_r0))
    p_extinct = np.zeros((n_cfr, n_r0))  # literal D == N, for the remark

    logger.info("SIRD R0xCFR collapse sweep", N=N, reps=N_REPS, grid=f"{n_cfr}x{n_r0}",
                collapse_fraction=COLLAPSE_FRACTION)

    for i, cfr in enumerate(CFR_GRID):
        for j, R0 in enumerate(R0_GRID):
            beta, gamma, mu = params_from(R0, cfr)

            def make_model(stream, beta=beta, gamma=gamma, mu=mu) -> SIRD:
                return SIRD(S=N - I0, I=I0, beta=beta, gamma=gamma, mu=mu, rng=stream)

            trajs = replicate(make_model, n=N_REPS, t_max=T_MAX, rng=rng)
            dead = np.array([t.final("D") for t in trajs])
            frac_dead[i, j] = dead.mean() / N
            p_collapse[i, j] = (dead >= COLLAPSE_FRACTION * N).mean()
            p_extinct[i, j] = (dead == N).mean()

    # Headline: most deadly corner; plus the remark that literal extinction never happens.
    R0_max, cfr_max = R0_GRID[-1], CFR_GRID[-1]
    logger.success(
        "Collapse sweep complete",
        deadliest_corner=f"R0={R0_max}, CFR={cfr_max}",
        frac_dead_corner=round(float(frac_dead[-1, -1]), 3),
        p_collapse_corner=round(float(p_collapse[-1, -1]), 3),
        max_p_literal_extinction=float(p_extinct.max()),  # remark: ~0 everywhere
    )

    with figure(figsize=(8, 5), save="project2/theo/plots/part1/sird_extinction_sweep.png") as fig:
        ax = fig.add_subplot(111)
        im = ax.imshow(frac_dead, origin="lower", aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(range(n_r0), [f"{r:g}" for r in R0_GRID])
        ax.set_yticks(range(n_cfr), [f"{c:g}" for c in CFR_GRID])
        ax.set_xlabel("Basic reproduction number $R_0$")
        ax.set_ylabel("Case fatality ratio CFR")
        ax.set_title(f"Mean death toll D(∞)/N over $R_0$ × CFR (N={N})")
        ax.grid(False)
        for i in range(n_cfr):
            for j in range(n_r0):
                ax.text(j, i, f"{frac_dead[i, j]:.2f}", ha="center", va="center",
                        color="black" if frac_dead[i, j] < 0.6 else "white", fontsize=9)
        fig.colorbar(im, ax=ax, label="fraction of population dead")
