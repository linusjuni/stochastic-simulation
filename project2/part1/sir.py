# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic (Claude, claude-opus-4-8) were used in the
# creation of this file. Specifically, Claude assisted with designing the
# event-driven (Gillespie) SSA engine and the Part I analysis code (fade-out
# probability and the other classical SIR questions).
# The author takes full responsibility for all content and decisions in this file.
"""Project 2, Part I — stochastic SIR: basic modeling (questions a-d).

The stochastic SIR model is a continuous-time Markov chain on the population
counts (S, I, R) with N = S + I + R held constant (closed population). Two
events drive it:

    event       state change         rate
    ---------   ------------------   -----------
    infection   S-1, I+1             beta * S * I / N
    recovery    I-1, R+1             gamma * I

with basic reproduction number R0 = beta / gamma. It is simulated with the
event-by-event principle from the Day 3 lecture (Gillespie's SSA): draw the
waiting time to the next event from an exponential whose rate is the total
event rate, pick which event fired in proportion to its rate, update the state,
repeat.

This single file answers the four Part I questions, each in its own function;
the model is extended in place as later questions require it:

    (a) likelihood the disease disappears    -> question_a
    (b) cyclical behaviour                   -> question_b   (todo)
    (c) extinction of the population         -> question_c   (todo)
    (d) when deterministic models suffice    -> question_d

Run everything, or a single question:
    uv run python -m project2.part1.sir
    uv run python -m project2.part1.sir a
"""
import sys

import numpy as np
from scipy.integrate import solve_ivp

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

FIGDIR = "project2/figures"

def simulate_sir(
    beta: float,
    gamma: float,
    N: int,
    I0: int,
    t_max: float,
    rng: np.random.Generator,
    R_init: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Event-driven (Gillespie SSA) simulation of the stochastic SIR model.

    Parameters
    ----------
    beta : infection rate (effective contacts per infected per unit time).
    gamma : recovery rate (1 / mean infectious period).
    N : total population size, constant.
    I0 : initial number of infected.
    t_max : stop simulating once the clock passes this time.
    rng : NumPy random generator.
    R_init : initial number of recovered/immune (default 0).

    Returns
    -------
    (times, S, I, R): the jump times and the compartment counts after each
    event. times[k] is when state (S[k], I[k], R[k]) was entered, so the count
    held from times[k] until times[k+1]. The first entry is the initial state
    at t = 0. The run stops when I hits 0 (no more events possible) or the
    clock passes t_max.
    """
    S, I, R = N - I0 - R_init, I0, R_init
    t = 0.0
    times, S_hist, I_hist, R_hist = [t], [S], [I], [R]

    while I > 0 and t < t_max:
        rate_infect = beta * S * I / N
        rate_recover = gamma * I
        total = rate_infect + rate_recover

        # Time to the next event: exponential with the total event rate.
        t += rng.exponential(1.0 / total)
        if t > t_max:
            break

        # Which event fired, in proportion to its rate.
        if rng.random() < rate_infect / total:
            S, I = S - 1, I + 1  # infection
        else:
            I, R = I - 1, R + 1  # recovery

        times.append(t)
        S_hist.append(S)
        I_hist.append(I)
        R_hist.append(R)

    return (
        np.array(times),
        np.array(S_hist),
        np.array(I_hist),
        np.array(R_hist),
    )


def final_sizes(
    beta: float,
    gamma: float,
    N: int,
    I0: int,
    t_max: float,
    n_reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run n_reps independent epidemics; return total ever infected for each.

    For a closed SIR started with no recovered, the total ever infected is
    N - S_final (everyone who left the susceptible pool).
    """
    out = np.empty(n_reps, dtype=int)
    for k in range(n_reps):
        _, S, _, _ = simulate_sir(beta, gamma, N, I0, t_max, rng)
        out[k] = N - S[-1]
    return out


def sir_ode(
    beta: float,
    gamma: float,
    N: int,
    I0: int,
    t_max: float,
    n_points: int = 600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic (mean-field) SIR baseline, integrated in fractions.

    In fractions s=S/N, i=I/N, r=R/N the model is N-independent:
        ds/dt = -beta * s * i
        di/dt =  beta * s * i - gamma * i
        dr/dt =  gamma * i
    The stochastic process concentrates on this curve as N grows. Returns
    (t, S, I, R) scaled back to counts for direct comparison with simulate_sir.
    """
    def deriv(t: float, y: np.ndarray) -> tuple[float, float, float]:
        s, i, _ = y
        infection = beta * s * i
        recovery = gamma * i
        return (-infection, infection - recovery, recovery)

    y0 = ((N - I0) / N, I0 / N, 0.0)
    t_eval = np.linspace(0.0, t_max, n_points)
    sol = solve_ivp(deriv, (0.0, t_max), y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)
    return sol.t, sol.y[0] * N, sol.y[1] * N, sol.y[2] * N


# --------------------------------------------------------------------------- #
# (a) Likelihood the disease disappears                                        #
# --------------------------------------------------------------------------- #
def question_a() -> None:
    """P(disease fades out) vs the branching-process prediction (1/R0)^I0.

    While S ~ N, each infective recovers at rate gamma and infects at rate
    beta, so the early outbreak is a birth-death branching process whose
    extinction probability is 1/R0 per seed -> (1/R0)^I0 for R0 > 1, and 1 for
    R0 <= 1. We confirm this by Monte Carlo: the final-size distribution is
    bimodal (a spike of fade-outs near 0, a bump of major epidemics), and a run
    counts as a fade-out if it infects fewer than 1% of the population.
    """
    rng = np.random.default_rng(settings.SEED)
    N, I0, gamma, t_max = 10000, 1, 1.0, 10000.0
    fadeout_cutoff = 0.02 * N  # < 2% infected = the disease disappeared

    # --- 1. Final-size distribution at a single R0, to show bimodality ------ #
    R0_demo, n_hist = 2.0, 5000
    sizes = final_sizes(R0_demo * gamma, gamma, N, I0, t_max, n_hist, rng)
    p_fadeout = np.mean(sizes < fadeout_cutoff)
    logger.info(
        "Final-size distribution",
        R0=R0_demo,
        empirical_P_fadeout=round(float(p_fadeout), 3),
        theory_P_fadeout=round(1.0 / R0_demo, 3),
    )

    with figure(figsize=(9, 5), save=f"{FIGDIR}/a_final_size_hist.png") as fig:
        ax = fig.add_subplot(111)
        ax.hist(sizes, bins=60, color="C0", edgecolor="black", linewidth=0.4)
        ax.axvline(fadeout_cutoff, color="C3", ls="--", label="fade-out cutoff (1% of N)")
        ax.set_xlabel("final outbreak size (total ever infected)")
        ax.set_ylabel("count")
        ax.set_title(
            f"Bimodal outbreak sizes (R0={R0_demo:.0f}, I0={I0}, N={N}, {n_hist} runs)\n"
            f"P(fade-out): empirical {p_fadeout:.2f} vs theory {1 / R0_demo:.2f}"
        )
        ax.legend()

    # --- 2. P(fade-out) vs R0, empirical (with CI) against theory ----------- #
    R0_grid = np.linspace(0.5, 4.0, 15)
    n_reps = 100
    p_emp = np.empty_like(R0_grid)
    ci = np.empty_like(R0_grid)
    for j, r0 in enumerate(R0_grid):
        sizes = final_sizes(r0 * gamma, gamma, N, I0, t_max, n_reps, rng)
        p = np.mean(sizes < fadeout_cutoff)
        p_emp[j] = p
        ci[j] = 1.96 * np.sqrt(p * (1 - p) / n_reps)  # normal-approx binomial CI

    p_theory = np.where(R0_grid > 1.0, (1.0 / R0_grid) ** I0, 1.0)

    with figure(figsize=(9, 5), save=f"{FIGDIR}/a_fadeout_vs_R0.png") as fig:
        ax = fig.add_subplot(111)
        ax.errorbar(
            R0_grid, p_emp, yerr=ci, fmt="o", capsize=3, color="C0",
            label="simulation (95% CI)",
        )
        ax.plot(R0_grid, p_theory, color="C3", lw=2, label=r"theory $(1/R_0)^{I_0}$")
        ax.axvline(1.0, color="0.6", ls=":", label="epidemic threshold R0=1")
        ax.set_xlabel(r"basic reproduction number $R_0$")
        ax.set_ylabel("P(disease disappears)")
        ax.set_title(f"Fade-out probability vs R0 (I0={I0}, N={N}, {n_reps} runs/point)")
        ax.legend()


# --------------------------------------------------------------------------- #
# (d) When are deterministic (ODE) models sufficiently precise?                #
# --------------------------------------------------------------------------- #
def question_d() -> None:
    """How large must N be before the deterministic ODE is an accurate proxy?

    In fractions the ODE is one N-independent curve; the stochastic process
    fluctuates around it with amplitude ~ 1/sqrt(N) (CLT / system-size
    expansion). So the ODE is "precise enough" only when N is large enough that
    (i) relative fluctuations are small and (ii) fade-out -- which the ODE
    cannot represent at all -- is negligible. We show the spaghetti-vs-ODE
    overlay at small and large N, and the coefficient of variation of the peak
    prevalence vs N against a 1/sqrt(N) reference.
    """
    rng = np.random.default_rng(settings.SEED)
    R0, gamma, I0, t_max = 3.0, 1.0, 5, 30.0
    beta = R0 * gamma

    # --- 1. Stochastic major outbreaks vs the ODE, small vs large N -------- #
    with figure(figsize=(12, 5), save=f"{FIGDIR}/d_overlay.png") as fig:
        for ax, N in zip(fig.subplots(1, 2), (200, 5000)):
            shown = 0
            while shown < 40:
                t, S, I, _ = simulate_sir(beta, gamma, N, I0, t_max, rng)
                if (N - S[-1]) <= 0.1 * N:
                    continue  # skip fade-outs; the ODE describes major outbreaks
                ax.plot(
                    t, I / N, color="C0", alpha=0.15, lw=0.8,
                    label="stochastic (major outbreaks)" if shown == 0 else None,
                )
                shown += 1
            t_ode, _, I_ode, _ = sir_ode(beta, gamma, N, I0, t_max)
            ax.plot(t_ode, I_ode / N, color="C3", lw=2.5, label="ODE (mean-field)")
            ax.set_title(f"N = {N}")
            ax.set_xlabel("time")
            ax.set_ylabel("infected fraction  I/N")
            ax.set_xlim(0, 20)
            ax.legend()

    # --- 2. Coefficient of variation of peak prevalence vs N -------------- #
    N_values = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])
    n_reps = 300
    cv = np.empty(len(N_values))
    rel_err = np.empty(len(N_values))
    for m, N in enumerate(N_values):
        peaks = []
        for _ in range(n_reps):
            _, S, I, _ = simulate_sir(beta, gamma, int(N), I0, t_max, rng)
            if (N - S[-1]) > 0.1 * N:
                peaks.append(I.max() / N)
        peaks = np.array(peaks)
        _, _, I_ode, _ = sir_ode(beta, gamma, int(N), I0, t_max)
        i_peak_ode = I_ode.max() / N
        cv[m] = peaks.std(ddof=1) / peaks.mean()
        rel_err[m] = abs(peaks.mean() - i_peak_ode) / i_peak_ode
        logger.info(
            "Precision vs N",
            N=int(N),
            n_major=len(peaks),
            cv_peak=round(float(cv[m]), 4),
            rel_err_mean=round(float(rel_err[m]), 4),
        )

    c = float(np.mean(cv * np.sqrt(N_values)))  # fit the 1/sqrt(N) reference line
    with figure(figsize=(9, 5), save=f"{FIGDIR}/d_precision_vs_N.png") as fig:
        ax = fig.add_subplot(111)
        ax.loglog(N_values, cv, "o-", color="C0", label="CV of peak prevalence")
        ax.loglog(N_values, rel_err, "s-", color="C2", label="rel. error of mean peak vs ODE")
        ax.loglog(N_values, c / np.sqrt(N_values), "--", color="0.5", label=r"$\propto N^{-1/2}$")
        ax.set_xlabel("population size N")
        ax.set_ylabel("relative magnitude")
        ax.set_title(f"Stochastic departure from the ODE shrinks like $1/\\sqrt{{N}}$ (R0={R0:.0f})")
        ax.legend()


# --------------------------------------------------------------------------- #
# Demo + dispatch                                                              #
# --------------------------------------------------------------------------- #
def _demo() -> None:
    """Simulate and plot a single SIR trajectory to eyeball the engine."""
    rng = np.random.default_rng(settings.SEED)
    beta, gamma, N, I0 = 0.3, 0.1, 10000, 10
    t_max = 200.0

    times, S, I, R = simulate_sir(beta, gamma, N, I0, t_max, rng)
    logger.info(
        "Single SIR run",
        R0=beta / gamma,
        peak_I=int(I.max()),
        final_S=int(S[-1]),
        total_infected=int(N - S[-1]),
        duration=round(float(times[-1]), 1),
    )

    with figure(figsize=(9, 5), save=f"{FIGDIR}/sir_single_run.png") as fig:
        ax = fig.add_subplot(111)
        ax.step(times, S, where="post", label="S (susceptible)")
        ax.step(times, I, where="post", label="I (infected)")
        ax.step(times, R, where="post", label="R (recovered)")
        ax.set_xlabel("time")
        ax.set_ylabel("individuals")
        ax.set_title(f"Stochastic SIR — single run (N={N}, R0={beta / gamma:.0f})")
        ax.legend()


QUESTIONS = {
    "demo": _demo,
    "a": question_a,
    "d": question_d,
}


if __name__ == "__main__":
    keys = sys.argv[1:] or list(QUESTIONS)
    for key in keys:
        if key not in QUESTIONS:
            logger.error("Unknown question", key=key, available=list(QUESTIONS))
            continue
        QUESTIONS[key]()
