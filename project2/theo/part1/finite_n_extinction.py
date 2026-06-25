# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for synthesizing, code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

"""Part I(c), experiment 2 — collapse is robust; literal extinction is a fragile tail.

Take the deadliest SIRD: gamma = 0, so CFR = 1 (no recovery). With no immune
refuge, a major outbreak destroys almost the whole population. We compare two
outcome definitions across population size N and the two force-of-infection
conventions ("fixed" = beta*S*I/N0 mass action, "living" = beta*S*I/(S+I+R)):

  * Societal COLLAPSE      D >= 0.95 N   -- the robust, meaningful measure.
  * Literal EXTINCTION     D == N        -- every last individual, a brittle tail.

Result: collapse happens with probability ~1 for *both* conventions at all N --
the population is destroyed regardless of the contact model. Literal extinction,
by contrast, is fragile: under "fixed" mass action a deterministic cushion of
N*z survivors (z = escape fraction) escapes, so P(D=N) ~ exp(-N z) -> 0 as N grows;
only the "living" convention drives it to 1. The denominator decides the last
0.7% of survivors, not whether society collapses.

Run:  uv run python -m project2.theo.part1.finite_n_extinction
"""

import numpy as np

from project2.theo.analysis import confidence_interval, replicate
from project2.theo.models import SIRD
from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

I0 = 1
REMOVAL_RATE = 0.1  # r = mu (gamma = 0), mean infectious period = 10
R0 = 5.0
N_VALUES = (10, 20, 50, 100, 200, 500, 1000, 2000)
N_REPS = 1500
T_MAX = 5000.0
MAJOR_FRACTION = 0.5   # condition on a major outbreak (>= 50% dead)
COLLAPSE_FRACTION = 0.95

STYLE = {  # foi mode -> colour
    "fixed": "tab:blue",
    "living": "tab:red",
}
LABEL = {"fixed": "fixed $N_0$", "living": "living $S{+}I{+}R$"}


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    mu = REMOVAL_RATE
    beta = R0 * mu  # gamma = 0
    logger.info("Collapse-vs-literal sweep", R0=R0, collapse_fraction=COLLAPSE_FRACTION)

    with figure(
        figsize=(8, 5), save="project2/theo/plots/part1/finite_n_extinction.png"
    ) as fig:
        ax = fig.add_subplot(111)

        for foi, color in STYLE.items():
            p_collapse, p_literal = [], []
            for N in N_VALUES:
                def make_model(stream, N=N, foi=foi) -> SIRD:
                    return SIRD(
                        S=N - I0, I=I0, beta=beta, gamma=0.0, mu=mu, rng=stream, foi=foi
                    )

                trajs = replicate(make_model, n=N_REPS, t_max=T_MAX, rng=rng)
                dead = np.array([t.final("D") for t in trajs])
                major = dead >= MAJOR_FRACTION * N
                dmaj = dead[major]
                p_collapse.append((dmaj >= COLLAPSE_FRACTION * N).mean())
                p_literal.append((dmaj == N).mean())
                logger.info("cell", foi=foi, N=N,
                            p_collapse=round(p_collapse[-1], 3),
                            p_literal=round(p_literal[-1], 3))

            ax.plot(N_VALUES, p_collapse, "-o", color=color,
                    label=f"collapse $\\geq$95%  ({LABEL[foi]})")
            ax.plot(N_VALUES, p_literal, "--s", color=color, alpha=0.6,
                    label=f"literal $D{{=}}N$  ({LABEL[foi]})")

        ax.set_xscale("log")
        ax.set_xlabel("Population size $N$")
        ax.set_ylabel("Probability (given a major outbreak)")
        ax.set_title(f"Collapse is robust; literal extinction is a fragile tail (SIRD, $R_0$={R0:g}, CFR=1)")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=8)
