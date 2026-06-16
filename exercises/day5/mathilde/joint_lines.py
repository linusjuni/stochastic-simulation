from __future__ import annotations

import math
import numpy as np

def unnormalized_joint_pmf(A1: float, A2: float, i: int, j: int) -> float:
    return ((A1 ** i) / math.factorial(i)) * ((A2 ** j) / math.factorial(j))

def theoretical_grid(A1: float, A2: float, m: int) -> np.ndarray:
    grid = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            if i + j <= m:
                grid[i, j] = unnormalized_joint_pmf(A1, A2, i, j)
    return grid / np.sum(grid)

def mh_direct(A1: float, A2: float, m: int, n: int, rng: np.random.Generator, burn_in: int = 1000, thin: int = 10) -> np.ndarray:
    """Direct 2D M-H random walk."""
    samples = np.empty((n, 2), dtype=int)
    curr_i, curr_j = 0, 0
    
    total_steps = burn_in + n * thin
    for step in range(total_steps):
        r = rng.random()
        di, dj = 0, 0
        if r < 0.25:
            di = 1
        elif r < 0.5:
            di = -1
        elif r < 0.75:
            dj = 1
        else:
            dj = -1
            
        prop_i, prop_j = curr_i + di, curr_j + dj
        
        if prop_i < 0 or prop_i > m or prop_j < 0 or prop_j > m or prop_i + prop_j > m:
            pass
        else:
            acc_prob = min(1.0, unnormalized_joint_pmf(A1, A2, prop_i, prop_j) / unnormalized_joint_pmf(A1, A2, curr_i, curr_j))
            if rng.random() < acc_prob:
                curr_i, curr_j = prop_i, prop_j
                
        if step >= burn_in and (step - burn_in) % thin == 0:
            samples[(step - burn_in) // thin] = (curr_i, curr_j)
            
    return samples

def mh_coordinate(A1: float, A2: float, m: int, n: int, rng: np.random.Generator, burn_in: int = 1000, thin: int = 10) -> np.ndarray:
    """Coordinate-wise M-H choosing one coordinate to update at random."""
    samples = np.empty((n, 2), dtype=int)
    curr_i, curr_j = 0, 0
    
    total_steps = burn_in + n * thin
    for step in range(total_steps):
        if rng.random() < 0.5:
            prop_i, prop_j = curr_i + (1 if rng.random() < 0.5 else -1), curr_j
        else:
            prop_i, prop_j = curr_i, curr_j + (1 if rng.random() < 0.5 else -1)
            
        if prop_i < 0 or prop_i > m or prop_j < 0 or prop_j > m or prop_i + prop_j > m:
            pass
        else:
            acc_prob = min(1.0, unnormalized_joint_pmf(A1, A2, prop_i, prop_j) / unnormalized_joint_pmf(A1, A2, curr_i, curr_j))
            if rng.random() < acc_prob:
                curr_i, curr_j = prop_i, prop_j
                
        if step >= burn_in and (step - burn_in) % thin == 0:
            samples[(step - burn_in) // thin] = (curr_i, curr_j)
            
    return samples

def gibbs(A1: float, A2: float, m: int, n: int, rng: np.random.Generator, burn_in: int = 1000, thin: int = 10) -> np.ndarray:
    """Gibbs sampling with exact conditionals."""
    samples = np.empty((n, 2), dtype=int)
    i, j = 0, 0
    
    total_steps = burn_in + n * thin
    for step in range(total_steps):
        # sample i | j
        support_i = np.arange(m - j + 1)
        cond_density_i = np.array([(A1 ** k) / math.factorial(k) for k in support_i])
        i = rng.choice(support_i, p=cond_density_i / np.sum(cond_density_i))
        
        # sample j | i
        support_j = np.arange(m - i + 1)
        cond_density_j = np.array([(A2 ** k) / math.factorial(k) for k in support_j])
        j = rng.choice(support_j, p=cond_density_j / np.sum(cond_density_j))
        
        if step >= burn_in and (step - burn_in) % thin == 0:
            samples[(step - burn_in) // thin] = (i, j)
            
    return samples
