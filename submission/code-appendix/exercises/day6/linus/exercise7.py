import numpy as np

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

SEED = settings.SEED
N = 20
N_ITER = 20_000
PLOTS = "exercises/day6/linus/plots"


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
    return 1.0 / np.sqrt(1 + k)


def log_cooling(k: int) -> float:
    return 1.0 / np.log(2 + k)


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
    logger.info("Part 1a: circle sanity check")
    logger.info("Initial route cost", cost=round(float(route_cost(init_route, D)), 4))
    logger.info("Best route cost (SA)", cost=round(best_cost, 4))
    logger.info("Polygon perimeter 2N*sin(pi/N)", value=round(polygon_perimeter, 4))

    with figure(figsize=(10, 5), save=f"{PLOTS}/circle_sanity.png") as fig:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        plot_route(ax1, points, init_route, "Initial route")
        plot_route(ax2, points, best_route, "Final route (SA)")


def part1b_random(rng: np.random.Generator) -> None:
    points = rng.uniform(size=(N, 2))
    D = euclidean_matrix(points)

    init_route = rng.permutation(N)
    best_route, best_cost, cost_history = simulated_annealing(D, N_ITER, rng, init=init_route)

    logger.info("Part 1b: random points in unit square")
    logger.info("Initial route cost", cost=round(float(route_cost(init_route, D)), 4))
    logger.info("Best route cost (SA)", cost=round(best_cost, 4))

    with figure(figsize=(10, 5), save=f"{PLOTS}/random_route.png") as fig:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        plot_route(ax1, points, best_route, "Final route")
        ax2.plot(cost_history)
        ax2.set_title("Cost vs. iteration")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Tour cost")


def part2_cost_matrix(rng: np.random.Generator) -> None:
    D = np.loadtxt("exercises/day6/cost.csv", delimiter=",")
    n = D.shape[0]
    init_route = np.arange(n)
    init_cost = route_cost(init_route, D)

    best_route, best_cost, _ = simulated_annealing(D, N_ITER, rng, init=init_route)

    logger.info("Part 2: general cost matrix")
    logger.info("Initial route cost", cost=round(float(init_cost)))
    logger.info("Best route cost (SA)", cost=round(best_cost))
    logger.info("Best route (1-indexed towns)", route=(best_route + 1).tolist())

    coolings = {"1/sqrt(1+k)": sqrt_cooling, "1/log(2+k)": log_cooling}
    proposals = {"swap": swap_proposal, "reverse": reverse_proposal}

    logger.info("Cooling/proposal experiment — mean best cost over 5 seeds")
    for cooling_name, cooling in coolings.items():
        for proposal_name, proposal in proposals.items():
            costs = np.empty(5)
            for s in range(5):
                exp_rng = np.random.default_rng(SEED + s)
                _, c, _ = simulated_annealing(
                    D, N_ITER, exp_rng, cooling=cooling, proposal=proposal, init=init_route
                )
                costs[s] = c
            logger.info(
                "Result",
                cooling=cooling_name,
                proposal=proposal_name,
                mean_best_cost=round(float(costs.mean()), 1),
            )


def main() -> None:
    rng = np.random.default_rng(SEED)
    part1a_circle(rng)
    part1b_random(rng)
    part2_cost_matrix(rng)


if __name__ == "__main__":
    main()
