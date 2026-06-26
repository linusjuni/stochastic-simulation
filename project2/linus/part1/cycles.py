"""Part I (b) — can the disease exhibit cyclical behaviour?

A closed SIR cannot: S only ever decreases, so every run is a single epidemic
wave that ends at I = 0. Adding *waning immunity* (R -> S at rate omega) gives an
SIRS model whose susceptible pool refills, creating an endemic equilibrium.

Near that equilibrium the deterministic mean-field ODE shows *damped*
oscillations -- they ring a few times and settle onto the endemic line. The
stochastic simulation instead shows *sustained* noisy cycles: demographic
stochasticity (random individual events) continually re-excites the damping. So
the cyclical behaviour is a genuinely stochastic effect the ODE misses.

Run:  uv run python -m project2.linus.part1.cycles
"""

import numpy as np

from project2.linus.analysis import solve_sirs
from project2.linus.models import SIRS
from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

# Endemic / oscillatory regime (see plan: stable focus, I* well above 0 so the
# stochastic run does not fade out, N moderate so fluctuations stay visible).
N = 20000
BETA = 0.5
GAMMA = 0.1  # R0 = beta / gamma = 5
OMEGA = 0.002  # slow waning (mean immune period 1/omega = 500) -> weak damping
T_MAX = 1500.0

# Start near the endemic equilibrium but perturbed (I above I*) so the figure
# isolates the *endemic* dynamics: the ODE rings down to I*, the stochastic run
# keeps cycling. Starting from a full invasion (S = N) instead produces a giant
# first wave that dwarfs and hides the endemic oscillations we want to show.
S_INIT = N // 5  # S* = N / R0 = 4000
I_INIT = 600  # perturbed above I* (~314) to excite the damped oscillation
R_INIT = N - S_INIT - I_INIT

logger = get_logger(__name__)


def sample_on_grid(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Piecewise-constant (steps-post) lookup of `values` at `grid` times."""
    idx = np.searchsorted(times, grid, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)
    model = SIRS(
        S=S_INIT, I=I_INIT, R=R_INIT, beta=BETA, gamma=GAMMA, omega=OMEGA, rng=rng
    )
    logger.info(
        "Simulating stochastic SIRS",
        N=N, S0=S_INIT, I0=I_INIT, beta=BETA, gamma=GAMMA, omega=OMEGA, R0=model.R0,
    )

    traj = model.run(T_MAX)
    I_star = model.endemic_I

    # Late-time mean (second half) should sit near the endemic equilibrium.
    half = traj.t >= T_MAX / 2
    late_mean_I = float(traj["I"][half].mean())

    logger.success(
        "SIRS run complete",
        events=len(traj.times) - 1,
        endemic_I_star=round(I_star, 1),
        late_mean_I=round(late_mean_I, 1),
        min_I=int(traj["I"].min()),
        extinct=traj.extinct(),
    )

    # Deterministic overlay from the same initial condition.
    t_ode, S_ode, I_ode, R_ode = solve_sirs(
        N=N, I0=I_INIT, beta=BETA, gamma=GAMMA, omega=OMEGA, t_max=T_MAX,
        n_points=2000, S0=S_INIT, R0_count=R_INIT,
    )

    # Decimate the (potentially ~1e6-event) trajectory onto a regular grid so the
    # PNG stays light; statistics above use the full trajectory.
    grid = np.linspace(0.0, T_MAX, 4000)
    I_sto = sample_on_grid(traj.t, traj["I"], grid)
    S_sto = sample_on_grid(traj.t, traj["S"], grid)

    with figure(
        figsize=(12, 4.8), save="project2/linus/plots/part1/sirs_cycles.png"
    ) as fig:
        # (a) time series: stochastic sustained cycles vs damped ODE
        ax_t = fig.add_subplot(121)
        ax_t.plot(
            grid, I_sto, drawstyle="steps-post", color="tab:red", linewidth=0.8,
            alpha=0.7, label="stochastic $I(t)$",
        )
        ax_t.plot(t_ode, I_ode, color="black", linewidth=1.8, label="ODE $I(t)$")
        ax_t.axhline(
            I_star, color="grey", linestyle="--", linewidth=1,
            label=f"endemic $I^*$={I_star:.0f}",
        )
        ax_t.set_title(f"SIRS infectives over time ($N$={N}, $R_0$={model.R0:.0f})")
        ax_t.set_xlabel("Time")
        ax_t.set_ylabel("Infected $I$")
        ax_t.legend(fontsize=8)

        # (b) phase portrait: ODE spirals into the focus, stochastic orbits it
        ax_p = fig.add_subplot(122)
        ax_p.plot(
            S_sto, I_sto, color="tab:red", linewidth=0.5, alpha=0.5,
            label="stochastic",
        )
        ax_p.plot(S_ode, I_ode, color="black", linewidth=1.5, label="ODE")
        ax_p.plot(
            N / model.R0, I_star, "o", color="tab:blue", markersize=7,
            label="endemic equilibrium",
        )
        ax_p.set_title("Phase portrait")
        ax_p.set_xlabel("Susceptible $S$")
        ax_p.set_ylabel("Infected $I$")
        ax_p.legend(fontsize=8)

        fig.suptitle(
            "Cyclical behaviour: damped (ODE) vs sustained (stochastic) "
            "with waning immunity"
        )
