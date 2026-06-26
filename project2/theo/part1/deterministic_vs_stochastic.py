# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for synthesizing, code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

import numpy as np

from project2.theo.models import SIRD
from project2.theo.models.deterministic import final_size_fraction, sir_metrics, sir_ode
from project2.theo.analysis import replicate
from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

R0, BETA, GAMMA, I0 = 3.0, 0.3, 0.1, 1
T_MAX = 500.0
MAJOR_FRAC = 0.20  # major outbreak: final size >= 20% of N

N_SWEEP = (50, 100, 300, 1000, 3000, 10000)
N_REPS = 1200
TOL = 0.05  # "sufficient precision" tolerance

N_TRAJ = (100, 10000)   # small vs large N for the overlay
N_TRAJ_REPS = 40
N_HIST, HIST_REPS = 1000, 3000


def make_factory(N: int):
    def make_model(stream) -> SIRD:  # SIRD with mu = 0 is the classical SIR
        return SIRD(S=N - I0, I=I0, beta=BETA, gamma=GAMMA, mu=0.0, rng=stream)
    return make_model


def run_metrics(N: int, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Per-run final-size fraction R(inf)/N and peak fraction max(I)/N."""
    trajs = replicate(make_factory(N), n=n, t_max=T_MAX, rng=rng)
    final = np.array([t.final("R") / N for t in trajs])
    peak = np.array([t.peak("I") / N for t in trajs])
    return final, peak


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)

    # --- Figure 1: trajectory overlay, small vs large N ---
    with figure(figsize=(11, 4.5), save="project2/theo/plots/part1/det_vs_stoch_trajectories.png") as fig:
        for idx, N in enumerate(N_TRAJ):
            ax = fig.add_subplot(1, len(N_TRAJ), idx + 1)
            trajs = replicate(make_factory(N), n=N_TRAJ_REPS, t_max=T_MAX, rng=rng)
            for tr in trajs:
                ax.plot(tr.t, tr["I"] / N, drawstyle="steps-post",
                        color="tab:blue", alpha=0.25, lw=0.8)
            td, Sd, Id, Rd = sir_ode(N=N, I0=I0, beta=BETA, gamma=GAMMA, t_max=T_MAX)
            ax.plot(td, Id / N, color="tab:red", lw=2.5, label="deterministic ODE")
            ax.set_title(f"N = {N}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Infected fraction  I/N")
            ax.set_xlim(0, 120)
            ax.legend()
        fig.suptitle(f"Stochastic SIR runs vs deterministic ODE ($R_0$={R0:g}, $I_0$={I0})")

    # --- Figure 2: precision vs N (relative error of the major-outbreak mean + CV) ---
    rel_err_final, rel_err_peak, cv_final, minor_frac = [], [], [], []
    for N in N_SWEEP:
        final, peak = run_metrics(N, N_REPS, rng)
        major = final >= MAJOR_FRAC
        minor_frac.append(float((~major).mean()))
        ode = sir_metrics(N=N, I0=I0, beta=BETA, gamma=GAMMA, t_max=T_MAX)
        fmean, pmean = final[major].mean(), peak[major].mean()
        rel_err_final.append(abs(fmean - ode["final_size_frac"]) / ode["final_size_frac"])
        rel_err_peak.append(abs(pmean - ode["peak_frac"]) / ode["peak_frac"])
        cv_final.append(final[major].std(ddof=1) / fmean)
        logger.info("sweep", N=N, minor_frac=round(minor_frac[-1], 3),
                    rel_err_final=round(rel_err_final[-1], 4),
                    rel_err_peak=round(rel_err_peak[-1], 4),
                    cv_final=round(cv_final[-1], 4))
    logger.success("minor-outbreak fraction vs theory",
                   mean_minor=round(float(np.mean(minor_frac)), 3),
                   theory=round((1.0 / R0) ** I0, 3))

    N_arr = np.array(N_SWEEP, dtype=float)
    with figure(figsize=(8, 5), save="project2/theo/plots/part1/det_vs_stoch_precision.png") as fig:
        ax = fig.add_subplot(111)
        ax.plot(N_arr, rel_err_final, "o-", label="rel. error, final size (major mean)")
        ax.plot(N_arr, rel_err_peak, "s-", label="rel. error, peak (major mean)")
        ax.plot(N_arr, cv_final, "^-", color="tab:purple", label="CV of final size (spread)")
        ref = cv_final[0] * np.sqrt(N_arr[0] / N_arr)
        ax.plot(N_arr, ref, "--", color="gray", alpha=0.7, label=r"$\propto 1/\sqrt{N}$")
        ax.axhline(TOL, color="red", ls=":", label=f"{TOL:.0%} tolerance")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Population size $N$")
        ax.set_ylabel("Relative error / CV")
        ax.set_title(f"Deterministic precision vs N ($R_0$={R0:g}, major outbreaks)")
        ax.legend(fontsize=8)

    # --- Figure 3: bimodal final-size distribution the ODE cannot capture ---
    final_hist, _ = run_metrics(N_HIST, HIST_REPS, rng)
    ode_final = sir_metrics(N=N_HIST, I0=I0, beta=BETA, gamma=GAMMA, t_max=T_MAX)["final_size_frac"]
    minor_hist = float((final_hist < MAJOR_FRAC).mean())
    logger.info("histogram", N=N_HIST, ode_final_size=round(ode_final, 3),
                minor_frac=round(minor_hist, 3), theory=round((1.0 / R0) ** I0, 3))
    with figure(figsize=(8, 5), save="project2/theo/plots/part1/det_vs_stoch_bimodal.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(final_hist, bins=50, range=(0, 1), color="tab:blue", alpha=0.85, edgecolor="black", lw=0.4)
        ax.axvline(ode_final, color="tab:red", lw=2.5, ls="--",
                   label=f"deterministic ODE = {ode_final:.2f}")
        ax.set_xlabel("Final size  R(∞)/N")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Bimodal stochastic final size vs single ODE value "
            f"(N={N_HIST}, $R_0$={R0:g}; minor fraction ≈ {minor_hist:.0%})"
        )
        ax.legend()
