from project2.mathias.compartment_models import SEIR
import numpy as np
import plot_SEIR as plot
import matplotlib.pyplot as plt

from utils.settings import settings
rng = np.random.default_rng(settings.SEED)
fpath = "project2/mathias/figures/SEIR/"

if __name__ == "__main__":
    N = 1000
    model = SEIR(N)
    susceptible_fraction = 0.99
    model.set_initial_conditions(S0=int(N * susceptible_fraction))
    gamma = 1 / (2.3)
    sigma = 1 / (5.2)
    R_0 = 2.2
    beta = R_0 * gamma
    t_max = 120
    t, S, E, I, R = model.simulate(beta, sigma, gamma, rng, t_max=t_max)

    print(f"size of stochastic epidemic: R(end) : {R[-1]}, S(end) : {S[-1]}, E(end) : {E[-1]}, I(end) : {I[-1]}")

    deterministic = plot.deterministic_seir(beta, sigma, gamma, N, model.S0, model.E0, model.I0, model.R0, t_max)
    plot.plot_trajectory(t, S, E, I, R, deterministic=deterministic)
    plt.title(f"SEIR Model Simulation (N={N}, beta={beta:.2f}, sigma={sigma:.2f}, gamma={gamma:.2f}), Initial I={model.I0}")
    plt.savefig(fpath + "trajectory_SEIR.png", dpi=300, bbox_inches="tight")

    plot.plot_epidemic_curves([(t, S, E, I, R)], deterministic=deterministic, compartments="I")
    plt.title(f"SEIR Model Simulation (N={N}, beta={beta:.2f}, sigma={sigma:.2f}, gamma={gamma:.2f}), Initial I={model.I0}")
    plt.savefig(fpath + "epidemic_curves_SEIR.png", dpi=300, bbox_inches="tight")

    final_sizes = [R[-1] for _ in range(1000) for t, S, E, I, R in [model.simulate(beta, sigma, gamma, rng, t_max=t_max)]]
    deterministic_size = deterministic[4][-1]
    plot.plot_final_size_histogram(final_sizes, N, deterministic_size=deterministic_size)
    plt.title(f"SEIR Model Simulation (N={N}, beta={beta:.2f}, sigma={sigma:.2f}, gamma={gamma:.2f}), Initial I={model.I0}")
    plt.savefig(fpath + "final_size_histogram_SEIR.png", dpi=300, bbox_inches="tight")

    runs = model.simulate_ensemble(beta, sigma, gamma, rng, n_runs=50, t_max=t_max)
    plot.plot_epidemic_curves(runs, deterministic=deterministic, compartments=("E", "I"))
    plt.title(f"SEIR Model Simulation (N={N}, beta={beta:.2f}, sigma={sigma:.2f}, gamma={gamma:.2f}), Initial I={model.I0}, 50 runs")
    plt.savefig(fpath + "epidemic_curves_ensemble_SEIR.png", dpi=300, bbox_inches="tight")
    
    
    
    
