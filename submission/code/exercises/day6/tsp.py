from __future__ import annotations

import numpy as np


def euclidean_matrix(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def route_cost(route: np.ndarray, D: np.ndarray) -> float:
    return D[route, np.roll(route, -1)].sum()


def swap_proposal(route: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    new_route = route.copy()
    i, j = rng.choice(len(route), size=2, replace=False)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def reverse_proposal(route: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    i, j = sorted(rng.choice(len(route), size=2, replace=False))
    new_route = route.copy()
    new_route[i:j + 1] = new_route[i:j + 1][::-1]
    return new_route


def sqrt_cooling(k: int) -> float:
    return 1 / np.sqrt(1 + k)


def log_cooling(k: int) -> float:
    return 1 / np.log(2 + k)


def simulated_annealing(
    D: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
    cooling=sqrt_cooling,
    proposal=swap_proposal,
    init: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    n = D.shape[0]
    x = rng.permutation(n) if init is None else init.copy()
    cost_x = route_cost(x, D)

    best_route, best_cost = x.copy(), cost_x
    cost_history = np.empty(n_iter)

    for k in range(n_iter):
        y = proposal(x, rng)
        cost_y = route_cost(y, D)
        delta = cost_y - cost_x

        if delta <= 0 or rng.random() < np.exp(-delta / cooling(k)):
            x, cost_x = y, cost_y
            if cost_x < best_cost:
                best_route, best_cost = x.copy(), cost_x

        cost_history[k] = cost_x

    return best_route, best_cost, cost_history
