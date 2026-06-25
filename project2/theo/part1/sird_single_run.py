# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for synthesizing, code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

"""Part I(c) — a single stochastic SIRD outbreak of a deadly disease.

Runs one realisation of the SIRD model and plots the S/I/R/D trajectories. The
point of interest versus plain SIR is the dead compartment D: at the end of the
outbreak D(inf) is the death toll and S(inf) + R(inf) are the survivors.

Run:  uv run python -m project2.theo.part1.sird_single_run
"""

import numpy as np

from project2.theo.models import SIRD
from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

N = 1000
I0 = 10
BETA = 0.5
GAMMA = 0.05
MU = 0.05  # R0 = beta / (gamma + mu) = 5,  CFR = mu / (gamma + mu) = 0.5
T_MAX = 400.0

COLORS = {"S": "tab:blue", "I": "tab:red", "R": "tab:green", "D": "black"}


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    model = SIRD(S=N - I0, I=I0, beta=BETA, gamma=GAMMA, mu=MU, rng=rng)
    logger.info(
        "Simulating SIRD", N=N, I0=I0, beta=BETA, gamma=GAMMA, mu=MU,
        R0=model.R0, CFR=model.cfr,
    )

    traj = model.run(T_MAX)

    dead = traj.final("D")
    survivors = traj.final("S") + traj.final("R")
    logger.success(
        "Outbreak complete",
        events=len(traj.times) - 1,
        peak_infected=traj.peak("I"),
        time_of_peak=round(traj.time_of_peak("I"), 1),
        dead=dead,
        survivors=survivors,
        fraction_dead=round(dead / N, 3),
        extinct=traj.extinct(),
    )

    with figure(figsize=(9, 5), save="project2/theo/plots/part1/sird_single_run.png") as fig:
        ax = fig.add_subplot(111)
        for c in ("S", "I", "R", "D"):
            ax.plot(traj.t, traj[c], drawstyle="steps-post", label=c, color=COLORS[c])
        ax.set_title(
            f"Stochastic SIRD outbreak (N={N}, R0={model.R0:.1f}, CFR={model.cfr:.0%})"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Individuals")
        ax.legend()
