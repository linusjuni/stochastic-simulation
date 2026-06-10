from __future__ import annotations

import heapq

import numpy as np


def simulate(interarrival: np.ndarray, service: np.ndarray, m: int) -> float:
    # loss system: a customer is blocked (lost) if all m servers are busy on arrival
    arrivals = np.cumsum(interarrival)
    departures: list[float] = []  # min-heap of busy-server departure times
    blocked = 0

    for t, s in zip(arrivals, service):
        # free up any servers that finished before this arrival
        while departures and departures[0] <= t:
            heapq.heappop(departures)
        if len(departures) < m:
            heapq.heappush(departures, t + s)
        else:
            blocked += 1

    return blocked / len(arrivals)


def erlang_b(A: float, m: int) -> float:
    # numerically stable recursion for Erlang's B-formula, avoids large factorials
    b = 1.0
    for i in range(1, m + 1):
        b = A * b / (i + A * b)
    return b


# --- inter-arrival generators (mean 1) ---
def exp_interarrival(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.exponential(1.0, n)


def erlang_interarrival(shape: int):
    def gen(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.gamma(shape, scale=1.0 / shape, size=n)

    return gen


def hyperexp_interarrival(n: int, rng: np.random.Generator) -> np.ndarray:
    # mixture of two exponentials; mean = p1/lam1 + p2/lam2 = 0.8/0.8333 + 0.2/5 = 1
    p = [0.8, 0.2]
    lam = [0.8333, 5.0]
    branch = rng.choice(2, size=n, p=p)
    rates = np.where(branch == 0, lam[0], lam[1])
    return rng.exponential(1.0 / rates)


# --- service-time generators (mean 8) ---
def exp_service(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.exponential(8.0, n)


def constant_service(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.full(n, 8.0)


def pareto_service(k: float):
    # inversion sampling: X = beta / U^(1/k); beta chosen so E[X] = beta*k/(k-1) = 8
    beta = 8.0 * (k - 1) / k

    def gen(n: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.uniform(size=n)
        return beta / u ** (1.0 / k)

    return gen


def lognormal_service(n: int, rng: np.random.Generator) -> np.ndarray:
    # mean 8, with sigma=1 => mu = ln(8) - sigma^2/2
    sigma = 1.0
    mu = np.log(8.0) - sigma**2 / 2
    return rng.lognormal(mean=mu, sigma=sigma, size=n)
