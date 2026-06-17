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
    weights = theoretical_grid(A1, A2, m)
    total_steps = burn_in + n * thin
    r = rng.random(total_steps)
    u = rng.random(total_steps)

    samples = np.empty((n, 2), dtype=int)
    curr_i, curr_j = 0, 0
    for step in range(total_steps):
        ri = r[step]
        if ri < 0.25:
            di, dj = 1, 0
        elif ri < 0.5:
            di, dj = -1, 0
        elif ri < 0.75:
            di, dj = 0, 1
        else:
            di, dj = 0, -1

        prop_i, prop_j = curr_i + di, curr_j + dj
        if 0 <= prop_i <= m and 0 <= prop_j <= m and prop_i + prop_j <= m:
            if u[step] < weights[prop_i, prop_j] / weights[curr_i, curr_j]:
                curr_i, curr_j = prop_i, prop_j

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples[(step - burn_in) // thin] = (curr_i, curr_j)

    return samples

def mh_coordinate(A1: float, A2: float, m: int, n: int, rng: np.random.Generator, burn_in: int = 1000, thin: int = 10) -> np.ndarray:
    """Coordinate-wise M-H choosing one coordinate to update at random."""
    weights = theoretical_grid(A1, A2, m)
    total_steps = burn_in + n * thin
    coord = rng.random(total_steps) < 0.5
    move = np.where(rng.random(total_steps) < 0.5, 1, -1)
    u = rng.random(total_steps)

    samples = np.empty((n, 2), dtype=int)
    curr_i, curr_j = 0, 0
    for step in range(total_steps):
        if coord[step]:
            prop_i, prop_j = curr_i + move[step], curr_j
        else:
            prop_i, prop_j = curr_i, curr_j + move[step]

        if 0 <= prop_i <= m and 0 <= prop_j <= m and prop_i + prop_j <= m:
            if u[step] < weights[prop_i, prop_j] / weights[curr_i, curr_j]:
                curr_i, curr_j = prop_i, prop_j

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples[(step - burn_in) // thin] = (curr_i, curr_j)

    return samples

def gibbs(A1: float, A2: float, m: int, n: int, rng: np.random.Generator, burn_in: int = 1000, thin: int = 10) -> np.ndarray:
    """Gibbs sampling with exact conditionals."""
    cum1 = np.cumsum([(A1 ** k) / math.factorial(k) for k in range(m + 1)])
    cum2 = np.cumsum([(A2 ** k) / math.factorial(k) for k in range(m + 1)])
    total_steps = burn_in + n * thin
    u_i = rng.random(total_steps)
    u_j = rng.random(total_steps)

    samples = np.empty((n, 2), dtype=int)
    i, j = 0, 0
    for step in range(total_steps):
        c_i = cum1[: m - j + 1]
        i = int(np.searchsorted(c_i, u_i[step] * c_i[-1]))
        c_j = cum2[: m - i + 1]
        j = int(np.searchsorted(c_j, u_j[step] * c_j[-1]))

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples[(step - burn_in) // thin] = (i, j)

    return samples
