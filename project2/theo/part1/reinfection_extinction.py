# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for synthesizing, code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

"""Part I(c), experiment 3 — reinfection collapses society at far lower lethality.

Plain SIRD caps the death toll at about CFR: the recovered class is a permanent
immune refuge, so society collapses (>= 95% dead) only for a near-totally lethal
disease (experiment 1). Letting immunity *wane* (recovered return to S, the SIRSD
model with rate omega) removes that refuge: the disease recirculates and, because
death is the only permanent sink, grinds the population down -- so even a modestly
lethal disease can collapse society.

Two figures:
  1. A single SIRSD run: sustained transmission grinding S + R into D.
  2. P(societal collapse, D >= 0.95 N) vs CFR for omega = 0 (SIRD) and omega > 0
     (SIRSD) -- waning immunity shifts collapse to much lower CFR.

Run:  uv run python -m project2.theo.part1.reinfection_extinction
"""

import numpy as np

from project2.theo.analysis import confidence_interval, replicate
from project2.theo.models import SIRSD
from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

N = 1000
I0 = 10
R0 = 5.0
REMOVAL_RATE = 0.1  # r = gamma + mu
CFR_GRID = (0.2, 0.4, 0.6, 0.8, 0.9, 0.95)
OMEGAS = (0.0, 0.02, 0.05)  # 0.0 == plain SIRD (no waning); transition is near 0.01-0.05
N_REPS = 400
T_MAX = 15000.0  # generous: slow waning grinds the population over a long horizon
COLLAPSE_FRACTION = 0.95

COLORS = {0.0: "tab:green", 0.02: "tab:orange", 0.05: "tab:red"}
TRAJ_COLORS = {"S": "tab:blue", "I": "tab:red", "R": "tab:green", "D": "black"}
N_TRAJ, OMEGA_TRAJ = 2000, 0.1
CFR_TRAJ = 0.5


def params_from(cfr: float, r: float = REMOVAL_RATE):
    """(R0, CFR) -> (beta, gamma, mu) at fixed removal rate r and R0."""
    return R0 * r, (1.0 - cfr) * r, cfr * r  # beta, gamma, mu


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)

    # --- Figure 1: a single SIRSD trajectory (sustained transmission grinds to D) ---
    beta, gamma, mu = params_from(CFR_TRAJ)
    model = SIRSD(
        S=N_TRAJ - I0, I=I0, beta=beta, gamma=gamma, mu=mu, omega=OMEGA_TRAJ, rng=rng
    )
    traj = model.run(T_MAX)
    logger.info(
        "SIRSD trajectory", N=N_TRAJ, omega=OMEGA_TRAJ, R0=model.R0, CFR=model.cfr,
        dead=traj.final("D"), survivors=traj.final("S") + traj.final("R"),
    )
    with figure(figsize=(9, 5), save="project2/theo/plots/part1/sirsd_trajectory.png") as fig:
        ax = fig.add_subplot(111)
        for c in ("S", "I", "R", "D"):
            ax.plot(traj.t, traj[c], drawstyle="steps-post", label=c, color=TRAJ_COLORS[c])
        ax.set_title(
            f"SIRSD: waning immunity sustains transmission, grinding the population into D "
            f"(N={N_TRAJ}, $R_0$={model.R0:.0f}, CFR={model.cfr:.0%}, $\\omega$={OMEGA_TRAJ})"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Individuals")
        ax.legend()

    # --- Figure 2: P(collapse) vs CFR, with and without waning immunity ---
    with figure(
        figsize=(8, 5), save="project2/theo/plots/part1/reinfection_collapse_vs_cfr.png"
    ) as fig:
        ax = fig.add_subplot(111)
        for omega in OMEGAS:
            p, lo, hi = [], [], []
            for cfr in CFR_GRID:
                beta, gamma, mu = params_from(cfr)

                def make_model(stream, beta=beta, gamma=gamma, mu=mu, omega=omega) -> SIRSD:
                    return SIRSD(
                        S=N - I0, I=I0, beta=beta, gamma=gamma, mu=mu,
                        omega=omega, rng=stream,
                    )

                trajs = replicate(make_model, n=N_REPS, t_max=T_MAX, rng=rng)
                collapsed = np.array(
                    [t.final("D") >= COLLAPSE_FRACTION * N for t in trajs], dtype=float
                )
                # Sanity: how many runs were still infectious at T_MAX (truncated)?
                truncated = np.mean([t.final("I") > 0 for t in trajs])
                est = confidence_interval(collapsed)
                p.append(est.mean)
                lo.append(est.mean - est.lower)
                hi.append(est.upper - est.mean)
                logger.info("cell", omega=omega, CFR=cfr, p_collapse=round(est.mean, 3),
                            truncated=round(float(truncated), 3))

            label = "$\\omega$=0 (SIRD)" if omega == 0 else f"$\\omega$={omega} (SIRSD)"
            ax.errorbar(CFR_GRID, p, yerr=[lo, hi], marker="o", capsize=3,
                        color=COLORS[omega], label=label)

        ax.set_xlabel("Case fatality ratio CFR")
        ax.set_ylabel(f"P(societal collapse, $D \\geq$ {COLLAPSE_FRACTION:.0%} $N$)")
        ax.set_title(f"Reinfection collapses society at far lower lethality ($R_0$={R0:g}, N={N})")
        ax.set_ylim(-0.02, 1.02)
        ax.legend()
