"""Part II (d) -- illustrative single run of the leaky-vaccine SIR model.

Run:  uv run python -m project2.mathilde.part2.vaccination_single_run
"""

import numpy as np

from project2.mathilde.models import VaccinatedSIR
from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

N = 1000
I0 = 10
BETA = 0.3
GAMMA = 0.1  # R0 = beta / gamma = 3
T_MAX = 160.0
COVERAGE = 0.6  # fraction of the non-initially-infected population vaccinated
VE = 0.7  # vaccine effectiveness


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)

    S0 = round((1 - COVERAGE) * (N - I0))
    V0 = (N - I0) - S0

    model = VaccinatedSIR(S=S0, V=V0, I=I0, beta=BETA, gamma=GAMMA, ve=VE, rng=rng)
    logger.info(
        "Simulating vaccinated SIR",
        N=N,
        I0=I0,
        coverage=COVERAGE,
        ve=VE,
        R0=round(model.R0, 3),
        Rv=round(model.Rv, 3),
    )

    traj = model.run(T_MAX)

    logger.success(
        "Outbreak complete",
        events=len(traj.times) - 1,
        peak_infected=traj.peak("I"),
        time_of_peak=round(traj.time_of_peak("I"), 1),
        total_infected=traj.final("R"),
        extinct=traj.extinct(),
    )

    with figure(
        figsize=(9, 5),
        save="project2/mathilde/plots/part2/vaccination_single_run.png",
    ) as fig:
        ax = fig.add_subplot(111)
        colors = {"S": "tab:blue", "V": "tab:purple", "I": "tab:red", "R": "tab:green"}
        for c in ("S", "V", "I", "R"):
            ax.plot(traj.t, traj[c], drawstyle="steps-post", label=c, color=colors[c])
        ax.set_title(
            f"Leaky-vaccine SIR (N={N}, R0={model.R0:.1f}, "
            f"coverage={COVERAGE:.0%}, ve={VE:.0%}, Rv={model.Rv:.2f})"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Individuals")
        ax.legend()
