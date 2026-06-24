"""Part I (a) — likelihood that the disease disappears.

In a closed SIR every run eventually reaches I = 0, so "disappears" does not mean
"reaches extinction" (that is always true). It means a *minor outbreak*: the
disease fizzles out before causing a major epidemic. The final-size distribution
R(infinity) is bimodal -- a cluster of minor outbreaks near zero and a cluster of
major epidemics -- and we estimate the probability of landing in the minor mode.

Branching-process theory predicts the minor-outbreak probability is
(1 / R0)^I0 for R0 > 1, which we compare against.

Run:  uv run python -m project2.linus.part1.extinction
"""

import numpy as np

from project2.linus.analysis import confidence_interval, replicate
from project2.linus.models import SIR
from utils.logger import get_logger
from utils.plotting import figure, histogram
from utils.settings import settings

N = 1000
I0 = 1
BETA = 0.3
GAMMA = 0.1  # R0 = beta / gamma = 3
T_MAX = 200.0
N_REPS = 2000
MINOR_FRACTION = 0.01  # minor outbreak if fewer than 1% of N are ever infected

logger = get_logger(__name__)


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    R0 = BETA / GAMMA

    def make_model(stream: np.random.Generator) -> SIR:
        return SIR(S=N - I0, I=I0, beta=BETA, gamma=GAMMA, rng=stream)

    logger.info("Running replications", n=N_REPS, N=N, I0=I0, R0=R0)
    trajectories = replicate(make_model, n=N_REPS, t_max=T_MAX, rng=rng)

    final_sizes = np.array([traj.final("R") for traj in trajectories])
    threshold = MINOR_FRACTION * N
    disappeared = (final_sizes < threshold).astype(float)  # 1 if minor outbreak

    p_disappear = confidence_interval(disappeared)
    theory = (1.0 / R0) ** I0

    logger.success(
        "P(disease disappears)",
        estimate=str(p_disappear),
        theory=round(theory, 3),
        threshold=int(threshold),
    )

    # The distribution is strongly bimodal with a wide empty gap, so a single
    # linear histogram renders as a spike + void + blob. Split it into two
    # panels: a zoom on the minor-outbreak mode and one on the major-epidemic mode.
    minor_sizes = final_sizes[final_sizes < threshold]
    major_sizes = final_sizes[final_sizes >= threshold]

    with figure(
        figsize=(11, 4.5), save="project2/linus/plots/part1/final_size_distribution.png"
    ) as fig:
        ax_minor = fig.add_subplot(121)
        histogram(
            minor_sizes,
            ax=ax_minor,
            discrete=True,
            title=f"Minor outbreaks  (n={minor_sizes.size})",
            xlabel="Total ever infected, R(∞)",
        )
        ax_minor.set_xlim(0, 2 * threshold)
        ax_minor.axvline(
            threshold, color="red", linestyle="--", label=f"threshold = {int(threshold)}"
        )
        ax_minor.legend()

        ax_major = fig.add_subplot(122)
        histogram(
            major_sizes,
            ax=ax_major,
            bins=30,
            title=f"Major epidemics  (n={major_sizes.size})",
            xlabel="Total ever infected, R(∞)",
        )

        fig.suptitle(
            f"Bimodal final-size distribution ({N_REPS} runs, N={N}, R0={R0:.0f})"
        )
