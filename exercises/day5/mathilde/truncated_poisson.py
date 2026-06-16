from __future__ import annotations

import math
import numpy as np

def unnormalized_pmf(A: float, i: int) -> float:
    return (A ** i) / math.factorial(i)

def mh_sample(A: float, m: int, n: int, rng: np.random.Generator, burn_in: int = 1000, thin: int = 10) -> np.ndarray:
    samples = np.empty(n, dtype=int)
    current_state = 0
    
    total_steps = burn_in + n * thin
    for step in range(total_steps):
        # Propose state with equal probability -1 or 1
        proposed_state = current_state + (1 if rng.random() < 0.5 else -1)
        
        # Reject if outside bounds
        if proposed_state < 0 or proposed_state > m:
            pass 
        else:
            acceptance_prob = min(1.0, unnormalized_pmf(A, proposed_state) / unnormalized_pmf(A, current_state))
            if rng.random() < acceptance_prob:
                current_state = proposed_state
                
        if step >= burn_in and (step - burn_in) % thin == 0:
            samples[(step - burn_in) // thin] = current_state
            
    return samples
