from __future__ import annotations

import numpy as np

def draw_prior(rho: float, rng: np.random.Generator) -> tuple[float, float]:
    """Draw (theta, psi) from the prior via (xi, gamma)."""
    X = rng.normal(loc=0, scale=1)
    Z = rng.normal(loc=0, scale=1)
    Y = rho * X + np.sqrt(1 - rho**2) * Z
    return float(np.exp(X)), float(np.exp(Y))

def log_prior(theta: float, psi: float, rho: float) -> float:
    """Log-density of the prior on (theta, psi)."""
    return (-np.log(2 * np.pi) - np.log(theta) - np.log(psi) - 0.5 * np.log(1 - rho**2)
            - (np.log(theta)**2 - 2*rho*np.log(theta)*np.log(psi) + np.log(psi)**2)
            / (2 * (1 - rho**2)))

def log_likelihood(X: np.ndarray, theta: float, psi: float) -> float:
    """Log-likelihood of the data X given (theta, psi)."""
    n = len(X)
    return -0.5 * n * np.log(2 * np.pi * psi) - np.sum((X - theta)**2) / (2 * psi)

def log_posterior(theta: float, psi: float, X: np.ndarray, rho: float) -> float:
    """Unnormalised log-posterior."""
    return log_prior(theta, psi, rho) + log_likelihood(X, theta, psi)

def mh_posterior(X_obs: np.ndarray, rho: float, n_samples: int, rng: np.random.Generator, step: float = 0.5, burn_in: int = 1000, thin: int = 10) -> np.ndarray:
    """Coordinate-wise random-walk M-H for the posterior."""
    samples = np.empty((n_samples, 2))
    # Start at a reasonable point
    curr_theta = max(0.1, np.mean(X_obs))
    curr_psi = max(0.1, np.var(X_obs))
    
    total_steps = burn_in + n_samples * thin
    for i in range(total_steps):
        delta = rng.normal(loc=0, scale=step)
        if rng.random() < 0.5:
            prop_theta, prop_psi = curr_theta + delta, curr_psi
        else:
            prop_theta, prop_psi = curr_theta, curr_psi + delta
            
        if prop_psi <= 0 or prop_theta <= 0:
            pass # reject
        else:
            acc_ratio = min(1.0, np.exp(log_posterior(prop_theta, prop_psi, X_obs, rho) - log_posterior(curr_theta, curr_psi, X_obs, rho)))
            if rng.random() < acc_ratio:
                curr_theta, curr_psi = prop_theta, prop_psi
                
        if i >= burn_in and (i - burn_in) % thin == 0:
            samples[(i - burn_in) // thin] = (curr_theta, curr_psi)
            
    return samples
