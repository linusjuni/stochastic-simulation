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
    n = len(X_obs)
    sum_x = float(np.sum(X_obs))
    sum_x2 = float(np.sum(X_obs ** 2))

    def log_post(theta: float, psi: float) -> float:
        ll = -0.5 * n * np.log(2 * np.pi * psi) - (sum_x2 - 2 * theta * sum_x + n * theta**2) / (2 * psi)
        return log_prior(theta, psi, rho) + ll

    total_steps = burn_in + n_samples * thin
    deltas = rng.normal(0.0, step, total_steps)
    which = rng.random(total_steps) < 0.5   # True → update theta
    u = rng.random(total_steps)

    samples = np.empty((n_samples, 2))
    curr_theta = max(0.1, sum_x / n)
    curr_psi = max(0.1, sum_x2 / n - (sum_x / n) ** 2)
    curr_lp = log_post(curr_theta, curr_psi)

    for i in range(total_steps):
        if which[i]:
            prop_theta, prop_psi = curr_theta + deltas[i], curr_psi
        else:
            prop_theta, prop_psi = curr_theta, curr_psi + deltas[i]

        if prop_theta > 0 and prop_psi > 0:
            prop_lp = log_post(prop_theta, prop_psi)
            if u[i] < np.exp(min(0.0, prop_lp - curr_lp)):
                curr_theta, curr_psi = prop_theta, prop_psi
                curr_lp = prop_lp

        if i >= burn_in and (i - burn_in) % thin == 0:
            samples[(i - burn_in) // thin] = (curr_theta, curr_psi)

    return samples
