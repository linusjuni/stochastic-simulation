"""
Run with:
uv run -m exercises.day6.mathilde.main
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from exercises.day6.mathilde.tsp import (
    euclidean_matrix,
    log_cooling,
    reverse_proposal,
    route_cost,
    simulated_annealing,
    sqrt_cooling,
    swap_proposal,
)
from utils.plotting import figure

SEED = 42
N = 20
N_ITER = 20_000
OUT = Path(__file__).resolve().parent


def plot_route(ax, points: np.ndarray, route: np.ndarray, title: str) -> None:
    tour = np.append(route, route[0])
    ax.plot(points[tour, 0], points[tour, 1], "o-")
    ax.set_title(title)
    ax.set_aspect("equal")


def part1a_circle(rng: np.random.Generator) -> None:
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    D = euclidean_matrix(points)

    init_route = rng.permutation(N)
    best_route, best_cost, _ = simulated_annealing(D, N_ITER, rng, init=init_route)

    polygon_perimeter = 2 * N * np.sin(np.pi / N)
    print("\nPart 1a: circle sanity check")
    print(f"Initial route cost: {route_cost(init_route, D):.4f}")
    print(f"Best route cost:    {best_cost:.4f}")
    print(f"Polygon perimeter:  {polygon_perimeter:.4f}")

    with figure(figsize=(10, 5), save=OUT / "circle_sanity.png") as fig:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        plot_route(ax1, points, init_route, "Initial route")
        plot_route(ax2, points, best_route, "Final route")


def part1b_random(rng: np.random.Generator) -> None:
    points = rng.uniform(size=(N, 2))
    D = euclidean_matrix(points)

    init_route = rng.permutation(N)
    best_route, best_cost, cost_history = simulated_annealing(D, N_ITER, rng, init=init_route)

    print("\nPart 1b: random points")
    print(f"Initial route cost: {route_cost(init_route, D):.4f}")
    print(f"Best route cost:    {best_cost:.4f}")

    with figure(figsize=(10, 5), save=OUT / "random_route.png") as fig:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        plot_route(ax1, points, best_route, "Final route")
        ax2.plot(cost_history)
        ax2.set_title("Cost vs. iteration")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Tour cost")


def part2_cost_matrix(rng: np.random.Generator) -> None:
    D = np.loadtxt(Path(__file__).resolve().parents[1] / "cost.csv", delimiter=",")
    n = D.shape[0]
    init_route = np.arange(n)

    best_route, best_cost, cost_history = simulated_annealing(D, N_ITER, rng, init=init_route)

    print("\nPart 2: Learn cost matrix")
    print(f"Initial route cost: {route_cost(init_route, D):.4f}")
    print(f"Best route cost:    {best_cost:.4f}")
    print(f"Best route (1-indexed towns): {best_route + 1}")

    coolings = {"sqrt": sqrt_cooling, "log": log_cooling}
    proposals = {"swap": swap_proposal, "reverse": reverse_proposal}
    seeds = range(5)

    print("\nCooling/proposal experiment (best cost averaged over 5 seeds)")
    print(f"{'cooling':<8} {'proposal':<8} {'mean best cost':>15}")

    best_config = None
    best_config_cost = np.inf
    best_config_history = None

    for cooling_name, cooling in coolings.items():
        for proposal_name, proposal in proposals.items():
            costs = np.empty(len(seeds))
            histories = []
            for s in seeds:
                exp_rng = np.random.default_rng(SEED + s)
                _, c, h = simulated_annealing(
                    D, N_ITER, exp_rng, cooling=cooling, proposal=proposal, init=init_route
                )
                costs[s] = c
                histories.append(h)
            mean_cost = costs.mean()
            print(f"{cooling_name:<8} {proposal_name:<8} {mean_cost:>15.4f}")
            if mean_cost < best_config_cost:
                best_config_cost = mean_cost
                best_config = (cooling_name, proposal_name)
                best_config_history = histories[int(np.argmin(costs))]

    with figure(figsize=(8, 5), save=OUT / "tsp_costmatrix.png") as fig:
        ax = fig.add_subplot(111)
        ax.plot(best_config_history)
        ax.set_title(f"Cost history (best config: {best_config[0]} cooling, {best_config[1]} proposal)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Tour cost")


def main() -> None:
    rng = np.random.default_rng(SEED)
    part1a_circle(rng)
    part1b_random(rng)
    part2_cost_matrix(rng)


if __name__ == "__main__":
    main()
