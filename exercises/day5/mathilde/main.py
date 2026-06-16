"""
Run with:
uv run -m exercises.day5.mathilde.main
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from exercises.day5.mathilde.bayesian import draw_prior, mh_posterior
from exercises.day5.mathilde.joint_lines import gibbs, mh_coordinate, mh_direct, theoretical_grid
from exercises.day5.mathilde.stats import chi_square_gof
from exercises.day5.mathilde.truncated_poisson import mh_sample
from utils.plotting import figure

SEED = 42
A = 8.0
M = 10
A1 = 4.0
A2 = 4.0
RHO = 0.5
N_SAMPLES = 100_000
DATA_SIZES = (10, 100, 1000)
OUT = Path(__file__).resolve().parent

def _theoretical_pmf_1d(A: float, m: int) -> np.ndarray:
    weights = np.array([(A ** i) / math.factorial(i) for i in range(m + 1)], dtype=float)
    return weights / weights.sum()

def part1_truncated_poisson(rng: np.random.Generator) -> None:
    print("=== Part 1: Truncated Poisson M-H ===")
    samples = mh_sample(A, M, N_SAMPLES, rng)
    obs_counts = np.bincount(samples, minlength=M + 1).astype(float)
    
    expected_probs = _theoretical_pmf_1d(A, M)
    chi2, p, dof = chi_square_gof(obs_counts, expected_probs)
    
    print(f"  Chi-Square: chi2={chi2:.4f}, p-value={p:.4f}, dof={dof}")
    
    import matplotlib.pyplot as plt
    from utils.plotting import histogram
    histogram(samples, discrete=True, title="Metropolis-Hastings Samples, A = 8, m=10", xlabel="State", ylabel="Frequency")
    plt.savefig(OUT / "part1_truncated_poisson.png")
    plt.close()

def part2_methods(rng: np.random.Generator) -> None:
    print("\n=== Part 2: Joint Distribution i+j <= 10 ===")
    grid_prob = theoretical_grid(A1, A2, M)
    
    methods = {
        "Direct M-H": mh_direct,
        "Coordinate M-H": mh_coordinate,
        "Gibbs": gibbs
    }
    
    samples_dict = {}
    
    for name, func in methods.items():
        samples = func(A1, A2, M, N_SAMPLES, rng)
        samples_dict[name] = samples
        
        # Count pairs
        obs_grid = np.zeros((M + 1, M + 1))
        for i, j in samples:
            obs_grid[i, j] += 1
            
        mask = grid_prob > 0
        obs_counts = obs_grid[mask]
        expected_probs = grid_prob[mask]
        
        chi2, p, dof = chi_square_gof(obs_counts, expected_probs)
        print(f"  {name:<15}: chi2={chi2:>6.2f}, p-value={p:.4f}, dof={dof}")
        
    import matplotlib.pyplot as plt
    from utils.plotting import histogram
    
    # Mathias style: one plot for M-H, one for Gibbs
    mh_samples = samples_dict["Coordinate M-H"] # Using Coordinate M-H as it mixes well
    histogram([i + j for i, j in mh_samples], discrete=True, title="Metropolis-Hastings Samples, A1 = A2 = 4, m=10", xlabel="State", ylabel="Frequency")
    plt.savefig(OUT / "part2_metropolis_hastings.png")
    plt.close()
    
    gibbs_samples = samples_dict["Gibbs"]
    histogram([i + j for i, j in gibbs_samples], discrete=True, title="Gibbs Sampling Samples, A1 = A2 = 4, m=10", xlabel="State", ylabel="Frequency")
    plt.savefig(OUT / "part2_gibbs.png")
    plt.close()

def part3_bayesian(rng: np.random.Generator) -> None:
    print("\n=== Part 3: Bayesian Model ===")
    true_theta, true_psi = draw_prior(RHO, rng)
    print(f"  True (theta, psi) drawn from prior: ({true_theta:.4f}, {true_psi:.4f})")

    import seaborn as sns
    import matplotlib.pyplot as plt

    for n in DATA_SIZES:
        X_obs = rng.normal(true_theta, np.sqrt(true_psi), size=n)
        samples = mh_posterior(X_obs, RHO, n_samples=N_SAMPLES, rng=rng, step=1.0, burn_in=10000, thin=10)

        theta_s, psi_s = samples[:, 0], samples[:, 1]
        print(f"  n={n:<4}: posterior mean (theta, psi) = ({theta_s.mean():.4f}, {psi_s.mean():.4f})")

        g = sns.jointplot(x=theta_s, y=psi_s, kind="scatter", alpha=0.3)
        g.plot_joint(sns.kdeplot, color="red", levels=8, zorder=2)
        g.figure.suptitle(f"Posterior Samples with rho={RHO}, data n={n}", y=1.02)
        g.set_axis_labels(r"$\theta$", r"$\psi$")
        g.savefig(OUT / f"part3_posterior_n{n}.png")
        plt.close()

def main() -> None:
    rng = np.random.default_rng(SEED)
    part1_truncated_poisson(rng)
    part2_methods(rng)
    part3_bayesian(rng)

if __name__ == "__main__":
    main()
