import numpy as np

from project2.linus.models import SIR
from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

N = 1000
I0 = 10
BETA = 0.3
GAMMA = 0.1  # R0 = beta / gamma = 3
T_MAX = 160.0


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    model = SIR(S=N - I0, I=I0, beta=BETA, gamma=GAMMA, rng=rng)
    logger.info("Simulating SIR", N=N, I0=I0, beta=BETA, gamma=GAMMA, R0=model.R0)

    traj = model.run(T_MAX)

    logger.success(
        "Outbreak complete",
        events=len(traj.times) - 1,
        peak_infected=traj.peak("I"),
        time_of_peak=round(traj.time_of_peak("I"), 1),
        total_infected=traj.final("R"),
        extinct=traj.extinct(),
    )

    with figure(figsize=(9, 5), save="project2/linus/plots/part1/sir_single_run.png") as fig:
        ax = fig.add_subplot(111)
        for c, color in zip(("S", "I", "R"), ("tab:blue", "tab:red", "tab:green")):
            ax.plot(traj.t, traj[c], drawstyle="steps-post", label=c, color=color)
        ax.set_title(f"Stochastic SIR outbreak (N={N}, R0={model.R0:.1f})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Individuals")
        ax.legend()
