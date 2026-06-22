# this is derived by claude OPUS 
import numpy as np

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import Settings

from project1.mathias.part2.task7 import simulate_continuous_time_markov_chain
from project1.mathias.part3.task12 import (
    Q,
    STATES,
    create_fake_time_series_data,
)

logger = get_logger(__name__)
settings = Settings()
RANDOM_NUMBER_GENERATOR = np.random.default_rng(settings.SEED)
NO_OF_WOMEN = 1000
LAG = 48
ABSORBING_STATE = 4
CONVERGENCE_TOLERANCE = 1e-3
MAX_ITERATIONS = 100
MAX_BRIDGE_ATTEMPTS = 100_000


def simulate_bridge(rng, start_state, end_state, duration, Q):
    """
    Simulate a CTMC over [0, duration] starting in `start_state`, conditioned on
    the state at time `duration` equaling `end_state` (rejection sampling).

    Returns
    -------
    jumps : list[tuple[int, int]]
        Sequence of (from_state, to_state) jumps that occurred in (0, duration).
    sojourns : np.ndarray of shape (n_states,)
        Total time spent in each non-absorbing state during [0, duration].
    """
    n_states = Q.shape[0]
    for _ in range(MAX_BRIDGE_ATTEMPTS):
        jumps = []
        sojourns = np.zeros(n_states)
        state = start_state
        current_time = 0.0

        while True:
            if state == ABSORBING_STATE:
                final_state = ABSORBING_STATE
                break

            exit_rate = -Q[state, state]
            holding = rng.exponential(1 / exit_rate)

            if current_time + holding >= duration:
                sojourns[state] += duration - current_time
                final_state = state
                break

            sojourns[state] += holding
            current_time += holding

            jump_probabilities = Q[state] / exit_rate
            jump_probabilities[state] = 0
            next_state = rng.choice(n_states, p=jump_probabilities)
            jumps.append((state, next_state))
            state = next_state

        if final_state == end_state:
            return jumps, sojourns

    raise RuntimeError(
        f"Bridge from state {start_state} to {end_state} over {duration} months "
        f"failed after {MAX_BRIDGE_ATTEMPTS} attempts."
    )


def accumulate_sufficient_statistics(rng, time_series_data, Q_current, lag=LAG):
    n_states = Q_current.shape[0]
    N = np.zeros((n_states, n_states))
    S = np.zeros(n_states)

    for series in time_series_data:
        for k in range(len(series) - 1):
            jumps, sojourns = simulate_bridge(
                rng,
                start_state=series[k],
                end_state=series[k + 1],
                duration=lag,
                Q=Q_current,
            )
            for (i, j) in jumps:
                N[i, j] += 1
            S += sojourns

    return N, S


def update_Q(N, S, structure_mask):
    """Equation (2): q_ij = N_ij / S_i for i != j, diagonals = -row sums.

    `structure_mask` is a boolean matrix with True where transitions are allowed;
    enforces the known zero-pattern (e.g. cancer cannot regress).
    """
    n_states = N.shape[0]
    Q_new = np.zeros_like(N, dtype=float)
    for i in range(n_states):
        for j in range(n_states):
            if i != j and structure_mask[i, j] and S[i] > 0:
                Q_new[i, j] = N[i, j] / S[i]
        Q_new[i, i] = -np.sum(Q_new[i])
    return Q_new


def mcem(rng, time_series_data, Q_initial, structure_mask, tol=CONVERGENCE_TOLERANCE, max_iterations=MAX_ITERATIONS):
    Q_current = Q_initial.copy()
    history = [Q_current.copy()]

    for iteration in range(max_iterations):
        N, S = accumulate_sufficient_statistics(rng, time_series_data, Q_current)
        Q_new = update_Q(N, S, structure_mask)

        diff = np.max(np.abs(Q_new - Q_current))
        logger.info(
            "MCEM iteration",
            iteration=iteration,
            infinity_norm_change=float(diff),
        )

        Q_current = Q_new
        history.append(Q_current.copy())

        if diff < tol:
            logger.info("MCEM converged", iteration=iteration, tolerance=tol)
            break
    else:
        logger.info("MCEM hit max iterations without converging", max_iterations=max_iterations)

    return Q_current, history


def build_initial_Q(structure_mask, lag=LAG):
    """Initial guess: uniform rate of 1/lag on every allowed off-diagonal entry."""
    n_states = structure_mask.shape[0]
    Q_initial = np.zeros((n_states, n_states))
    rate = 1.0 / lag
    for i in range(n_states):
        for j in range(n_states):
            if i != j and structure_mask[i, j]:
                Q_initial[i, j] = rate
        Q_initial[i, i] = -np.sum(Q_initial[i])
    return Q_initial

def from_history_to_diffs(history):
    diffs = []
    for i in range(1, len(history)):
        diff = np.max(np.abs(history[i] - history[i-1]))
        diffs.append(diff)
    return diffs

if __name__ == "__main__":
    lifetimes, trajectories = simulate_continuous_time_markov_chain(
        RANDOM_NUMBER_GENERATOR, initial_state=0, no_of_women=NO_OF_WOMEN, Q=Q
    )
    time_series_data, _ = create_fake_time_series_data(lifetimes, trajectories, lag=LAG)

    structure_mask = Q != 0
    np.fill_diagonal(structure_mask, False)

    Q_initial = build_initial_Q(structure_mask, lag=LAG)

    Q_estimated, history = mcem(
        RANDOM_NUMBER_GENERATOR,
        time_series_data,
        Q_initial,
        structure_mask,
    )

    np.set_printoptions(precision=5, suppress=True)
    print("True Q:")
    print(Q)
    print("\nInitial Q^(0):")
    print(Q_initial)
    print("\nEstimated Q:")
    print(Q_estimated)
    print("\nAbsolute error |Q_est - Q_true|:")
    print(np.abs(Q_estimated - Q))

    diffs = from_history_to_diffs(history)
    with figure(figsize=(10, 6), save="project1/mathias/part3/figs/mcem_convergence.png") as fig:
        ax = fig.add_subplot(111)
        ax.plot(diffs, marker="o")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Infinity norm of Q change")
        ax.set_title("MCEM Convergence (log scale)")
        ax.axhline(y=CONVERGENCE_TOLERANCE, color="red", linestyle="--", label="Convergence Tolerance")
        ax.legend()
        
        
        
