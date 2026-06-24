from SIR import SIR
import numpy as np
import plot_SIR as plot
import matplotlib.pyplot as plt

from utils.settings import settings
rng = np.random.default_rng(settings.SEED)

if __name__ == "__main__":
    N = 100000
    model = SIR(N)
    model.set_initial_conditions(S0=N - 5)
    beta = 0.3
    gamma = 0.1
    t_max = 160
    t, S, I, R = model.simulate(beta, gamma, rng, t_max=t_max)
    deterministic = plot.deterministic_sir(beta, gamma, N, model.S0, model.I0, model.R0, t_max)
    plot.plot_trajectory(t, S, I, R, deterministic=deterministic)
    plot.plot_epidemic_curves([(t, S, I, R)], deterministic=deterministic)
    final_sizes = [R[-1] for _ in range(1000) for t, S, I, R in [model.simulate(beta, gamma, rng, t_max=t_max)]]
    deterministic_size = deterministic[3][-1]
    plot.plot_final_size_histogram(final_sizes, N, deterministic_size=deterministic_size)

    plot.plot_stacked(t, S, I, R, ax=None, order=["S", "I", "R"])
    plt.show()