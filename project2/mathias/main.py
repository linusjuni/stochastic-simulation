from project2.mathias.compartment_models import SIR
import numpy as np
import plot_SIR as plot
import matplotlib.pyplot as plt

from utils.settings import settings
rng = np.random.default_rng(settings.SEED)
fpath = "project2/mathias/figures/"

if __name__ == "__main__":
    N = 1000
    model = SIR(N)
    susceptible_fraction = 0.99
    model.set_initial_conditions(S0=int(N * susceptible_fraction))
    beta = 0.3
    gamma = 0.1
    t_max = 120
    t, S, I, R = model.simulate(beta, gamma, rng, t_max=t_max)
    deterministic = plot.deterministic_sir(beta, gamma, N, model.S0, model.I0, model.R0, t_max)
    plot.plot_trajectory(t, S, I, R, deterministic=deterministic)
    plt.title(f"SIR Model Simulation (N={N}, beta={beta}, gamma={gamma}), Initial I={model.I0}")
    plt.savefig(fpath + "trajectory.png", dpi=300, bbox_inches="tight")
    plot.plot_epidemic_curves([(t, S, I, R)], deterministic=deterministic)
    plt.savefig(fpath + "epidemic_curves.png", dpi=300, bbox_inches="tight")
    final_sizes = [R[-1] for _ in range(1000) for t, S, I, R in [model.simulate(beta, gamma, rng, t_max=t_max)]]
    deterministic_size = deterministic[3][-1]
    plot.plot_final_size_histogram(final_sizes, N, deterministic_size=deterministic_size)
    plt.savefig(fpath + "final_size_histogram.png", dpi=300, bbox_inches="tight")

    plot.plot_stacked(t, S, I, R, ax=None, order=["S", "I", "R"])
    plt.savefig(fpath + "stacked.png", dpi=300, bbox_inches="tight")

    #### Simulate multiple runs and plot epidemic curves
    n_runs = 50
    runs = model.simulate_ensemble(beta, gamma, rng, n_runs, t_max=t_max)
    plot.plot_epidemic_curves(runs, deterministic=deterministic)
    plt.title(f"SIR Model Simulation (N={N}, beta={beta}, gamma={gamma}), Initial I={model.I0}, {n_runs} runs")
    plt.savefig(fpath + "epidemic_curves_ensemble.png", dpi=300, bbox_inches="tight")
