"""Task 13 — recover Q from the coarse observations with Monte-Carlo EM.

We only observe each woman's state every 48 months (Task 12). The transition
rates have the unbiased estimator q_ij = N_ij / S_i (jumps i->j over total
sojourn in i), but N_ij and S_i are hidden. MCEM fills them in:

    E-step: for every observed interval y_k -> y_{k+1}, simulate a complete
            CTMC trajectory over 48 months under the current Q and *reject*
            unless it lands in y_{k+1}; accumulate N_ij and S_i from the
            accepted bridges.
    M-step: q_ij = N_ij / S_i (i != j); diagonal from eq. (1).

Iterate until ||Q^(k) - Q^(k+1)||_inf < 1e-3. This approximately recovers the
true Q used to generate the data.

Run with:
    uv run python -m project1.part3.task13
"""
import numpy as np

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)

# States: 0=post-surgery, 1=local, 2=distant, 3=both, 4=death (absorbing).
N_STATES = 5
DEATH = 4
N_WOMEN = 1000
DT = 48  # months between screenings
TOL = 1e-3  # the project's |Q^(k) - Q^(k+1)|_inf criterion (reported, not used to stop)
N_ITERS = 30  # fixed iteration budget
BURN_IN = 5  # iterates discarded before averaging
MAX_REJECTIONS = 100_000  # safety cap on the rejection bridge


def true_rate_matrix() -> np.ndarray:
    """Part 2 transition-rate matrix Q (the parameters we try to recover)."""
    Q = np.array(
        [
            [0.0, 0.005, 0.0025, 0.0, 0.001],
            [0.0, 0.0, 0.005, 0.004, 0.005],
            [0.0, 0.0, 0.0, 0.003, 0.005],
            [0.0, 0.0, 0.0, 0.0, 0.009],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def fill_diagonal(Q: np.ndarray) -> np.ndarray:
    """Set the diagonal to the negative off-diagonal row-sum (eq. 1)."""
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


# Transitions the clinical model allows (a complication, once present, persists;
# death is reachable from any state). State 0 cannot jump straight to "both" (3).
ALLOWED = [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]


def initial_guess() -> np.ndarray:
    """Generic starting Q: a small positive rate on every allowed transition."""
    Q = np.zeros((N_STATES, N_STATES))
    for i, j in ALLOWED:
        Q[i, j] = 0.01
    return fill_diagonal(Q)


# --- data generation (Task 12, copied inline to keep this file self-contained) ---


def simulate_ctmc(
    Q: np.ndarray, start: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Event-driven CTMC until death; returns (states, entry_times)."""
    states, times = [start], [0.0]
    t, s = 0.0, start
    while s != DEATH:
        rate = -Q[s, s]
        t += rng.exponential(1.0 / rate)
        probs = Q[s].copy()
        probs[s] = 0.0
        s = int(rng.choice(len(Q), p=probs / rate))
        states.append(s)
        times.append(t)
    return np.array(states), np.array(times)


def observe(states: np.ndarray, entry_times: np.ndarray, dt: int = DT) -> np.ndarray:
    """State on the screening grid t = 0, dt, 2*dt, ...; ends in death."""
    observations = []
    t = 0
    while True:
        idx = np.searchsorted(entry_times, t, side="right") - 1
        s = int(states[idx])
        observations.append(s)
        if s == DEATH:
            break
        t += dt
    return np.array(observations)


# --- MCEM ---


def simulate_bridge(
    Q: np.ndarray, start: int, target: int, duration: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Rejection-sampled CTMC bridge over one screening interval.

    Simulate from `start` for `duration` months under Q, recording jump counts
    N_ij and sojourn times S_i. Accept only if the state at `duration` equals
    `target`, otherwise resimulate. Returns (N_ij, S_i) for the accepted bridge.
    """
    for _ in range(MAX_REJECTIONS):
        n_ij = np.zeros((N_STATES, N_STATES))
        s_i = np.zeros(N_STATES)
        t, s = 0.0, start
        while True:
            rate = -Q[s, s]
            if rate <= 0:  # absorbed in death: remains here until `duration`
                break
            dt = rng.exponential(1.0 / rate)
            if t + dt >= duration:
                s_i[s] += duration - t
                break
            s_i[s] += dt
            t += dt
            probs = Q[s].copy()
            probs[s] = 0.0
            s_new = int(rng.choice(N_STATES, p=probs / rate))
            n_ij[s, s_new] += 1
            s = s_new
        if s == target:
            return n_ij, s_i
    raise RuntimeError(
        f"bridge {start}->{target} not matched in {MAX_REJECTIONS} attempts"
    )


def estep(
    observed: list[np.ndarray], Q: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate N_ij and S_i over all women's bridged intervals."""
    N = np.zeros((N_STATES, N_STATES))
    S = np.zeros(N_STATES)
    for y in observed:
        for a, b in zip(y[:-1], y[1:]):
            n_ij, s_i = simulate_bridge(Q, int(a), int(b), DT, rng)
            N += n_ij
            S += s_i
    return N, S


def mstep(N: np.ndarray, S: np.ndarray) -> np.ndarray:
    """q_ij = N_ij / S_i for i != j; diagonal from eq. (1)."""
    Q = np.zeros((N_STATES, N_STATES))
    nonzero = S > 0
    Q[nonzero] = N[nonzero] / S[nonzero, None]
    return fill_diagonal(Q)


if __name__ == "__main__":
    rng = np.random.default_rng(settings.SEED)

    Q_true = true_rate_matrix()
    logger.info("Generating observed series (Task 12 data)", n=N_WOMEN, seed=settings.SEED)
    observed = [observe(*simulate_ctmc(Q_true, 0, rng)) for _ in range(N_WOMEN)]

    # Each E-step uses one trajectory per interval, so the iterates carry Monte-Carlo
    # noise and never settle deterministically: |Q^(k)-Q^(k+1)|_inf drops fast, then
    # bounces around a noise floor. We therefore run a fixed budget, report when the
    # 1e-3 criterion is first met, and take the estimate as the post-burn-in mean.
    logger.info("Running MCEM", iters=N_ITERS, burn_in=BURN_IN)
    Q = initial_guess()
    deltas, iterates = [], []
    for k in range(1, N_ITERS + 1):
        N, S = estep(observed, Q, rng)
        Q_new = mstep(N, S)
        deltas.append(float(np.max(np.abs(Q - Q_new))))
        Q = Q_new
        iterates.append(Q.copy())
        logger.info("MCEM iteration", k=k, delta_inf=round(deltas[-1], 5))

    crossings = [k for k, d in enumerate(deltas, 1) if d < TOL]
    first_cross = crossings[0] if crossings else None
    floor = float(np.mean(deltas[BURN_IN:]))

    Q_hat = np.mean(iterates[BURN_IN:], axis=0)  # post-burn-in average estimate
    Q_sd = np.std(iterates[BURN_IN:], axis=0)  # Monte-Carlo spread of the iterates

    np.set_printoptions(precision=5, suppress=True)
    logger.success(
        "Convergence summary",
        first_iter_below_1e3=first_cross,
        noise_floor_delta=round(floor, 6),
    )
    logger.success("Recovered Q (post-burn-in mean)\n" + str(Q_hat))
    logger.success("Monte-Carlo std of recovered Q\n" + str(Q_sd))
    logger.success("True Q\n" + str(Q_true))
    logger.success(
        "Max abs error on rates", max_abs_err=round(float(np.max(np.abs(Q_hat - Q_true))), 5)
    )

    with figure(figsize=(8, 5), save="project1/plots/task13/convergence.png") as fig:
        ax = fig.add_subplot(111)
        ax.semilogy(range(1, len(deltas) + 1), deltas, marker="o")
        ax.axhline(TOL, color="black", linestyle="--", label=f"tolerance = {TOL:g}")
        ax.axhline(
            floor, color="C1", linestyle=":", label=f"noise floor $\\approx$ {floor:.1e}"
        )
        ax.set(
            title="MCEM convergence",
            xlabel="Iteration $k$",
            ylabel=r"$\|Q^{(k)} - Q^{(k+1)}\|_\infty$",
        )
        ax.legend()
