# CHAT helped with plotting functions for SIR model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp

from utils.settings import settings
plt.style.use(settings.PLOT_STYLE)


COLORS = {"S": "#2D6CB5", "I": "#C9344A", "R": "#1F9E80"}
INK = "#26285A"
DET = "#B7791F"


def deterministic_sir(beta, gamma, N, S0, I0, R0, t_max, n_points=400):
    def rhs(t, y):
        S, I, R = y
        infection = beta / N * S * I
        recovery = gamma * I
        return [-infection, infection - recovery, recovery]

    t_eval = np.linspace(0, t_max, n_points)
    sol = solve_ivp(rhs, (0, t_max), [S0, I0, R0], t_eval=t_eval, rtol=1e-8, atol=1e-8)
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def plot_trajectory(t, S, I, R, ax=None, deterministic=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    for label, path in (("S", S), ("I", I), ("R", R)):
        ax.step(t, path, where="post", color=COLORS[label], label=label)
    if deterministic is not None:
        td, Sd, Id, Rd = deterministic
        for label, path in (("S", Sd), ("I", Id), ("R", Rd)):
            ax.plot(td, path, color=COLORS[label], alpha=0.4, linestyle="--")
    ax.set_xlabel("time")
    ax.set_ylabel("individuals")
    ax.set_xlim(left=0)
    ax.legend()
    return ax


def plot_epidemic_curves(trajectories, ax=None, deterministic=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    for t, S, I, R in trajectories:
        ax.step(t, I, where="post", color=COLORS["I"], alpha=0.15, linewidth=1)
    if deterministic is not None:
        td, Sd, Id, Rd = deterministic
        ax.plot(td, Id, color=COLORS["I"], linewidth=2.5, label="deterministic I(t)")
        ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("infected I")
    ax.set_xlim(left=0)
    return ax


def plot_final_size_histogram(final_sizes, N, ax=None, deterministic_size=None, bins=40):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(final_sizes, bins=bins, range=(0, N), color=INK, alpha=0.85)
    if deterministic_size is not None:
        ax.axvline(deterministic_size, color=DET, linestyle="--", linewidth=2, label="deterministic")
        ax.legend()
    ax.set_xlabel("final size  R(inf) = N - S(inf)")
    ax.set_ylabel("count")
    return ax

def plot_stacked(t, S, I, R, ax=None, order=("S", "I", "R")):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    paths = {"S": S, "I": I, "R": R}
    lower = np.zeros_like(t, dtype=float)
    for label in order:
        upper = lower + paths[label]
        ax.fill_between(t, lower, upper, step="post", color=COLORS[label], alpha=0.9, label=label)
        lower = upper
    ax.set_xlabel("time")
    ax.set_ylabel("individuals")
    ax.set_xlim(left=0)
    ax.set_ylim(0, S[0] + I[0] + R[0])
    ax.legend(loc="center right")
    return ax