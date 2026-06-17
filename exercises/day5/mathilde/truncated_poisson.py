from __future__ import annotations

import math
import numpy as np

def unnormalized_pmf(A: float, i: int) -> float:
    return (A ** i) / math.factorial(i)

def mh_sample(A: float, m: int, n: int, rng: np.random.Generator, burn_in: int = 1000, thin: int = 10) -> np.ndarray:
    weights = np.array([unnormalized_pmf(A, i) for i in range(m + 1)])
    total_steps = burn_in + n * thin
    moves = np.where(rng.random(total_steps) < 0.5, 1, -1)
    u = rng.random(total_steps)

    samples = np.empty(n, dtype=int)
    current_state = 0
    for step in range(total_steps):
        proposed_state = current_state + moves[step]
        if 0 <= proposed_state <= m and u[step] < weights[proposed_state] / weights[current_state]:
            current_state = proposed_state

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples[(step - burn_in) // thin] = current_state

    return samples
