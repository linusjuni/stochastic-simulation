import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import factorial
from scipy.stats import norm, expon, pareto, erlang
from utils.plotting import histogram

np.random.seed(42)

def draw_prior(rho, n):
    X = np.random.normal(loc=0, scale=1, size=n)
    Z = np.random.normal(loc=0, scale=1, size=n)
    Y = rho * X + np.sqrt(1 - rho**2) * Z
    return np.exp(X), np.exp(Y)

def prior_dist(theta, psi, rho):
    fst_term = 1 / (2 * np.pi * theta * psi * np.sqrt(1 - rho**2))
    exp_term = np.exp(-(np.log(theta)**2 - 2*rho*np.log(theta)*np.log(psi) + np.log(psi)**2) / (2*(1-rho**2)))
    return fst_term * exp_term

def likelihood(X, theta , psi):
    n = len(X)
    fst_term = (2 * np.pi * psi) ** (-n / 2)
    exp_term = np.exp(-(1 / (2 * psi)) * np.sum((X - theta)**2))
    return fst_term * exp_term

def posterior(theta, psi, X, rho):
    return likelihood(X, theta, psi) * prior_dist(theta, psi, rho)

def metropolis_hastings_part2(n_samples, init_state_theta, init_state_psi, X_obs, Sigma, thinning=10):
    samples = [(init_state_theta, init_state_psi)]
    for _ in range(n_samples):
        curr_state_theta, curr_state_psi = samples[-1]
        delta = np.random.normal(loc=0, scale=Sigma)
        p = np.random.random()
        if p < 0.5:
            proposed_theta , proposed_psi = curr_state_theta + delta, curr_state_psi
        else:
            proposed_theta , proposed_psi = curr_state_theta, curr_state_psi + delta
        if proposed_psi <= 0 or proposed_theta <= 0:
            samples.append((curr_state_theta, curr_state_psi))
            continue
        acceptance_ratio = min(1, posterior(proposed_theta, proposed_psi, X_obs, rho) / posterior(curr_state_theta, curr_state_psi, X_obs, rho))
        if np.random.random() < acceptance_ratio:
            samples.append((proposed_theta, proposed_psi))
        else:
            samples.append((curr_state_theta, curr_state_psi))
    return samples[::thinning]
        
def log_prior(theta, psi, rho):
    return (-np.log(2 * np.pi) - np.log(theta) - np.log(psi) - 0.5 * np.log(1 - rho**2)
            - (np.log(theta)**2 - 2*rho*np.log(theta)*np.log(psi) + np.log(psi)**2)
            / (2 * (1 - rho**2)))


def log_likelihood(X, theta, psi):
    n = len(X)
    return -0.5 * n * np.log(2 * np.pi * psi) - np.sum((X - theta)**2) / (2 * psi)


def log_posterior(theta, psi, X, rho):
    return log_prior(theta, psi, rho) + log_likelihood(X, theta, psi)
        
def metropolis_hastings_log(n_samples, init_state_theta, init_state_psi, X_obs, Sigma, thinning=10):
    samples = [(init_state_theta, init_state_psi)]
    for _ in range(n_samples):
        curr_state_theta, curr_state_psi = samples[-1]
        delta = np.random.normal(loc=0, scale=Sigma)
        p = np.random.random()
        if p < 0.5:
            proposed_theta , proposed_psi = curr_state_theta + delta, curr_state_psi
        else:
            proposed_theta , proposed_psi = curr_state_theta, curr_state_psi + delta
        if proposed_psi <= 0 or proposed_theta <= 0:
            samples.append((curr_state_theta, curr_state_psi))
            continue
        acceptance_ratio = min(1, np.exp(log_posterior(proposed_theta, proposed_psi, X_obs, rho) - log_posterior(curr_state_theta, curr_state_psi, X_obs, rho)))
        if np.random.random() < acceptance_ratio:
            samples.append((proposed_theta, proposed_psi))
        else:
            samples.append((curr_state_theta, curr_state_psi))
    return samples[::thinning]
        
        
if __name__ == "__main__":
    # DONE 
    # part 1) Metropolis-Hastings for joint density of i, j with A = 8 and m = 10
    n_samples = int(1e5)
    rho = 0.5
    X, Y = draw_prior(rho, n_samples)
    # g = sns.jointplot(x=X, y=Y, kind="scatter", alpha=0.3)
    # g.plot_joint(sns.kdeplot, color="red", levels=8, zorder=2)

    # g.figure.suptitle(f"Prior Samples with rho={rho}", y=1.02)
    # g.set_axis_labels(r"$\theta$", r"$\psi$")
    # g.savefig("exercises/day5/mathias/prior_samples.png")
    # plt.show()
    theta, psi = X[0], Y[0]
    # sample gaussian with mean theta and variance psi
    samples = np.random.normal(loc=theta, scale=np.sqrt(psi), size=10)
    histogram(samples, discrete=False, kde=True, title=f"Samples from Gaussian with N(theta, psi) (rho={rho}), n=10", xlabel="Value", ylabel="Frequency")
    plt.savefig("exercises/day5/mathias/gaussian_samples_theta_psi.png")
    # plt.show()

    # c) draw posterior samples from prior and likelihood
    Sigma = 1.0
    X_obs = np.random.normal(loc=theta, scale=np.sqrt(psi), size=10)
    samples = metropolis_hastings_part2(n_samples, init_state_theta=theta, init_state_psi=psi, X_obs=X_obs, Sigma=Sigma)
    burn_in = int(0.1 * len(samples))
    theta_s = [s[0] for s in samples[burn_in:]]
    psi_s = [s[1] for s in samples[burn_in:]]
    g = sns.jointplot(x=theta_s, y=psi_s, kind="scatter", alpha=0.3)
    g.plot_joint(sns.kdeplot, color="red", levels=8, zorder=2)
    g.figure.suptitle(f"Posterior Samples with rho={rho}, n={n_samples}", y=1.02)
    g.set_axis_labels(r"$\theta$", r"$\psi$")
    g.savefig("exercises/day5/mathias/posterior_samples_xobs10.png")
    # plt.show()

    
    n_samples = int(1e5)
    Sigma = 1.0
    X_obs = np.random.normal(loc=theta, scale=np.sqrt(psi), size=1000)
    samples = metropolis_hastings_log(n_samples, init_state_theta=theta, init_state_psi=psi, X_obs=X_obs, Sigma=Sigma)
    burn_in = int(0.1 * len(samples))
    theta_s = [s[0] for s in samples[burn_in:]]
    psi_s = [s[1] for s in samples[burn_in:]]
    g = sns.jointplot(x=theta_s, y=psi_s, kind="scatter", alpha=0.3)
    g.plot_joint(sns.kdeplot, color="red", levels=8, zorder=2)
    g.figure.suptitle(f"Log Posterior Samples with rho={rho}, n={n_samples}", y=1.02)
    g.set_axis_labels(r"$\theta$", r"$\psi$")
    g.savefig("exercises/day5/mathias/log_posterior_samples.png")
    # plt.show()

    Sigma = 0.01
    X_obs = np.random.normal(loc=theta, scale=np.sqrt(psi), size=10)
    samples = metropolis_hastings_part2(n_samples, init_state_theta=theta, init_state_psi=psi, X_obs=X_obs, Sigma=Sigma)
    burn_in = int(0.1 * len(samples))
    theta_s = [s[0] for s in samples[burn_in:]]
    psi_s = [s[1] for s in samples[burn_in:]]
    g = sns.jointplot(x=theta_s, y=psi_s, kind="scatter", alpha=0.3)
    g.plot_joint(sns.kdeplot, color="red", levels=8, zorder=2)
    g.figure.suptitle(f"Posterior Samples with rho={rho}, n={n_samples}", y=1.02)
    g.set_axis_labels(r"$\theta$", r"$\psi$")
    g.savefig("exercises/day5/mathias/posterior_samples_xobs10sigma_001.png")
    # plt.show()

    
    n_samples = int(1e5)
    Sigma = 0.01
    X_obs = np.random.normal(loc=theta, scale=np.sqrt(psi), size=1000)
    samples = metropolis_hastings_log(n_samples, init_state_theta=theta, init_state_psi=psi, X_obs=X_obs, Sigma=Sigma)
    burn_in = int(0.1 * len(samples))
    theta_s = [s[0] for s in samples[burn_in:]]
    psi_s = [s[1] for s in samples[burn_in:]]
    g = sns.jointplot(x=theta_s, y=psi_s, kind="scatter", alpha=0.3)
    g.plot_joint(sns.kdeplot, color="red", levels=8, zorder=2)
    g.figure.suptitle(f"Log Posterior Samples with rho={rho}, n={n_samples}", y=1.02)
    g.set_axis_labels(r"$\theta$", r"$\psi$")
    g.savefig("exercises/day5/mathias/log_posterior_samples_sigma_001.png")
    # plt.show()

    Sigma = 50
    X_obs = np.random.normal(loc=theta, scale=np.sqrt(psi), size=10)
    samples = metropolis_hastings_part2(n_samples, init_state_theta=theta, init_state_psi=psi, X_obs=X_obs, Sigma=Sigma)
    burn_in = int(0.1 * len(samples))
    theta_s = [s[0] for s in samples[burn_in:]]
    psi_s = [s[1] for s in samples[burn_in:]]
    g = sns.jointplot(x=theta_s, y=psi_s, kind="scatter", alpha=0.3)
    g.plot_joint(sns.kdeplot, color="red", levels=8, zorder=2)
    g.figure.suptitle(f"Posterior Samples with rho={rho}, n={n_samples}", y=1.02)
    g.set_axis_labels(r"$\theta$", r"$\psi$")
    g.savefig("exercises/day5/mathias/posterior_samples_xobs10_sigma50.png")
    # plt.show()

    
    n_samples = int(1e5)
    Sigma = 50
    X_obs = np.random.normal(loc=theta, scale=np.sqrt(psi), size=1000)
    samples = metropolis_hastings_log(n_samples, init_state_theta=theta, init_state_psi=psi, X_obs=X_obs, Sigma=Sigma)
    burn_in = int(0.1 * len(samples))
    theta_s = [s[0] for s in samples[burn_in:]]
    psi_s = [s[1] for s in samples[burn_in:]]
    g = sns.jointplot(x=theta_s, y=psi_s, kind="scatter", alpha=0.3)
    g.plot_joint(sns.kdeplot, color="red", levels=8, zorder=2)
    g.figure.suptitle(f"Log Posterior Samples with rho={rho}, n={n_samples}", y=1.02)
    g.set_axis_labels(r"$\theta$", r"$\psi$")
    g.savefig("exercises/day5/mathias/log_posterior_samples_sigma50.png")
    # plt.show()

    




    




    



