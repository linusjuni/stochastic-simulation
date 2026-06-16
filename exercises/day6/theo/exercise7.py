# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for synthesizing, code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

import os
import numpy as np
import matplotlib.pyplot as plt

# Part 1
def random_points(n):
    return np.random.uniform(0, 1, size=(n, 2))

def circle_points(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return np.random.permutation(pts)

def euclidean_cost_matrix(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))

def tour_length(route, cost):
    return cost[route, np.roll(route, -1)].sum()

# Simulated annealing core
def swap_proposal(route):
    new = route.copy()
    i, j = np.random.choice(len(route), size=2, replace=False)
    new[i], new[j] = new[j], new[i]
    return new

def two_opt_neighboors(route):
    new = route.copy()
    i, j = sorted(np.random.choice(len(route), size=2, replace=False))
    new[i:j + 1] = new[i:j + 1][::-1]
    return new

def cool_log(k):
    return 1.0 / np.log(2 + k)

def cool_sqrt(k):
    return 1.0 / np.sqrt(1 + k)

def simulated_annealing(cost, n_iter, cooling_fn=cool_log,
                        proposal_fn=swap_proposal, route0=None):
    """Minimise tour_length over permutations via simulated annealing.

    At iteration k (slide 11): propose Y from the current route, accept if it is
    no worse, otherwise accept with probability exp(-(f(Y)-f(X)) / T_k). We also
    remember the best tour ever visited, since the chain's final state need not
    be the best one seen. cooling_fn(k) gives T_k; proposal_fn(route) the
    candidate.
    """
    n = cost.shape[0]
    route = np.random.permutation(n) if route0 is None else route0.copy()
    cur_cost = tour_length(route, cost)
    best_route, best_cost = route.copy(), cur_cost

    history = np.empty(n_iter)
    for k in range(n_iter):
        T = cooling_fn(k)
        cand = proposal_fn(route)
        cand_cost = tour_length(cand, cost)
        delta = cand_cost - cur_cost
        if delta <= 0 or np.random.uniform() < np.exp(-delta / T):
            route, cur_cost = cand, cand_cost
        if cur_cost < best_cost:
            best_route, best_cost = route.copy(), cur_cost
        history[k] = cur_cost

    return best_route, best_cost, history

# ── Part 1: run + plots ─────────────────────────────────────────────────────────

def plot_route(points, route, title, path):
    """Draw the closed tour through the points in the plane."""
    closed = np.append(route, route[0])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(points[closed, 0], points[closed, 1], "-o", color="steelblue", zorder=1)
    ax.scatter(points[route[0], 0], points[route[0], 1],
               c="crimson", s=90, zorder=2, label="start")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

# ── Part 2: general cost matrix + experiments ───────────────────────────────────

def run_many(cost, n_iter, cooling_fn, proposal_fn, n_runs):
    """Run SA n_runs times from independent random starts (slides: multiple runs
    are beneficial). Returns the array of best costs, the overall best route, and
    the best-so-far convergence curve of the winning run."""
    best_costs = np.empty(n_runs)
    top_route, top_cost, top_curve = None, np.inf, None
    for r in range(n_runs):
        route, cst, hist = simulated_annealing(cost, n_iter, cooling_fn, proposal_fn)
        best_costs[r] = cst
        if cst < top_cost:
            top_route, top_cost = route, cst
            top_curve = np.minimum.accumulate(hist)  # best-so-far over the run
    return best_costs, top_route, top_curve

def mean_convergence(cost, n_iter, cooling_fn, n_runs, seed):
    """Average best-so-far curve over n_runs random starts (swap proposal).

    mean_curve[k] is the expected best cost found within a budget of k+1
    iterations, so the whole curve is a dense iteration-budget sweep. Seeding
    makes different schedules face identical random starts for a paired
    comparison."""
    np.random.seed(seed)
    curves = np.empty((n_runs, n_iter))
    for r in range(n_runs):
        _, _, hist = simulated_annealing(cost, n_iter, cooling_fn, swap_proposal)
        curves[r] = np.minimum.accumulate(hist)
    return curves.mean(axis=0), curves.std(axis=0)

if __name__ == "__main__":
    np.random.seed(42)
    N = 20
    N_ITER = 20000

    # Sanity check: points on a circle. The optimum is to walk the rim, i.e. the
    # perimeter of the regular N-gon inscribed in the circle, length 2N*sin(pi/N).
    circ = circle_points(N, radius=1.0)
    c_cost = euclidean_cost_matrix(circ)
    c_route, c_len, _ = simulated_annealing(c_cost, N_ITER)
    optimum = 2 * N * np.sin(np.pi / N)
    print(f"[circle] SA length = {c_len:.4f}  optimum = {optimum:.4f}  "
          f"gap = {100 * (c_len - optimum) / optimum:.2f}%")
    plot_route(circ, c_route, f"Circle sanity check (length {c_len:.3f})",
               "part1_circle_sanity.png")

    # Random points in the unit square.
    pts = random_points(N)
    p_cost = euclidean_cost_matrix(pts)
    p_route, p_len, hist = simulated_annealing(p_cost, N_ITER)
    print(f"[random] SA length = {p_len:.4f}")
    plot_route(pts, p_route, f"Random points (length {p_len:.3f})",
               "part1_random_route.png")

    # ── Part 2 ─────────────────────────────────────────────────────────────────
    cost_path = os.path.join(os.path.dirname(__file__), "..", "cost.csv")
    C = np.loadtxt(cost_path, delimiter=",")
    print(f"\n[part2] cost matrix {C.shape}, symmetric = {np.allclose(C, C.T)}")

    N_ITER2, N_RUNS = 20000, 10

    # Experiment 1: cooling schedules (swap proposal). Both schedules from the
    # slides keep T < 1.5, tiny next to cost differences of order 100, so on this
    # matrix SA accepts almost no uphill moves and behaves like greedy descent.
    schedules = {
        "1/ln(2+k)": cool_log,
        "1/sqrt(1+k)": cool_sqrt,
    }
    print(f"\nCooling schedules (swap, {N_RUNS} runs x {N_ITER2} iters):")
    cooling_curves = {}
    for name, fn in schedules.items():
        np.random.seed(1)  # paired: both schedules face identical random starts
        bc, _, curve = run_many(C, N_ITER2, fn, swap_proposal, N_RUNS)
        cooling_curves[name] = curve
        print(f"  {name:28s} best={bc.min():6.0f}  mean={bc.mean():7.1f}  std={bc.std():5.1f}")

    # Experiment 2: proposal mechanism, fixing the better-performing slide
    # schedule (1/sqrt) so we isolate the effect of the neighbourhood.
    proposals = {"swap": swap_proposal, "reverse (2-opt)": two_opt_neighboors}
    print(f"\nProposals (1/sqrt cooling, {N_RUNS} runs x {N_ITER2} iters):")
    proposal_results = {}
    for name, fn in proposals.items():
        np.random.seed(2)  # paired: both proposals face identical random starts
        bc, broute, curve = run_many(C, N_ITER2, cool_sqrt, fn, N_RUNS)
        proposal_results[name] = (bc, broute, curve)
        print(f"  {name:18s} best={bc.min():6.0f}  mean={bc.mean():7.1f}  std={bc.std():5.1f}")

    best_name = min(proposal_results, key=lambda k: proposal_results[k][0].min())
    best_bc, best_route, _ = proposal_results[best_name]
    print(f"\nBest tour ({best_name}, cost {best_bc.min():.0f}): {best_route.tolist()}")

    # Convergence plots for both experiments.
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name, curve in cooling_curves.items():
        ax[0].plot(curve, label=name)
    ax[0].set(title="Cooling schedules (swap proposal)",
              xlabel="iteration", ylabel="best cost so far")
    ax[0].legend()
    for name, (_, _, curve) in proposal_results.items():
        ax[1].plot(curve, label=name)
    ax[1].set(title="Proposals (1/sqrt cooling)",
              xlabel="iteration", ylabel="best cost so far")
    ax[1].legend()
    fig.savefig("part2_convergence.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Experiment 3: does the better schedule depend on the iteration budget?
    # The mean best-so-far curve at iteration k equals the expected result of a
    # budget-k run, so one long run per start is a dense budget sweep. Same seed
    # => both schedules face identical starts (paired comparison).
    BUDGET = 50000
    budget_curves = {
        name: mean_convergence(C, BUDGET, fn, N_RUNS, seed=1)
        for name, fn in {"1/ln(2+k)": cool_log, "1/sqrt(1+k)": cool_sqrt}.items()
    }
    print(f"\nIteration-budget sweep (swap, {N_RUNS} runs, mean best cost):")
    for cp in (1000, 5000, 20000, 50000):
        row = "   ".join(f"{name} = {curve[cp - 1]:7.1f}"
                         for name, (curve, _) in budget_curves.items())
        print(f"  budget {cp:6d}:   {row}")

    fig, ax = plt.subplots(figsize=(7, 5))
    iters = np.arange(1, BUDGET + 1)
    # dashed second curve so the exact overlap of the two schedules is visible
    for (name, (mean_c, std_c)), ls in zip(budget_curves.items(), ["-", "--"]):
        ax.plot(iters, mean_c, ls, linewidth=2, label=name)
        ax.fill_between(iters, mean_c - std_c, mean_c + std_c, alpha=0.2)
    ax.set(xscale="log", xlabel="iteration budget", ylabel="mean best cost",
           title="Cooling schedule vs iteration budget (swap)")
    ax.legend()
    fig.savefig("part2_budget.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

