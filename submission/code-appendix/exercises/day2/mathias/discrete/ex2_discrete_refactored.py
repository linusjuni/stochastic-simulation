import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from utils.plotting import histogram


np.random.seed(42)

OUT = "exercises/day2/mathias"
SIZE = 10_000


# ---------------------------------------------------------------- #
# (a) Geometric distribution via inverse CDF
# ---------------------------------------------------------------- #

def geometric_sample(p, size):
    U = np.random.random(size)
    return np.floor(np.log(U) / np.log(1 - p)) + 1


# ---------------------------------------------------------------- #
# 6-point distribution
# ---------------------------------------------------------------- #

P_SIX = np.array([7/48, 5/48, 1/8, 1/16, 1/4, 5/16])
VALUES = np.arange(1, 7)


def six_point_crude(size):
    cdf = np.cumsum(P_SIX)
    U = np.random.random(size)
    return VALUES[np.searchsorted(cdf, U)]


def six_point_rejection(size):
    k = len(P_SIX)
    c = P_SIX.max()
    samples = np.empty(size, dtype=int)
    n = 0
    while n < size:
        candidate = np.random.randint(0, k)
        if np.random.random() < P_SIX[candidate] / c:
            samples[n] = candidate + 1
            n += 1
    return samples


def build_alias_tables(p):
    k = len(p)
    F = np.array(p, dtype=float) * k
    L = np.arange(k)
    G = [i for i in range(k) if F[i] >= 1]
    S = [i for i in range(k) if F[i] <  1]
    while S:
        i, j = G[0], S[0]
        L[j] = i
        F[i] -= 1 - F[j]
        if F[i] < 1 - 1e-12:
            G.pop(0)
            S.append(i)
        S.pop(0)
    return F, L


def six_point_alias(size):
    F, L = build_alias_tables(P_SIX)
    k = len(F)
    samples = np.empty(size, dtype=int)
    for n in range(size):
        I = int(np.floor(k * np.random.random()))
        samples[n] = (I if np.random.random() <= F[I] else L[I]) + 1
    return samples


# ---------------------------------------------------------------- #
# Diagnostics
# ---------------------------------------------------------------- #

def chi_squared(samples, expected_probs):
    counts = np.array([np.sum(samples == v) for v in VALUES])
    expected = expected_probs * len(samples)
    return stats.chisquare(counts, f_exp=expected)


def plot_six_point(samples, title, path):
    fig, ax = plt.subplots()
    counts = np.array([np.sum(samples == v) for v in VALUES])
    width = 0.4
    ax.bar(VALUES - width/2, counts / len(samples), width=width,
           label="observed", edgecolor="black")
    ax.bar(VALUES + width/2, P_SIX, width=width,
           label="expected", edgecolor="black")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def benchmark(fn, *args, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #

if __name__ == "__main__":
    # (a) Geometric
    p_geom = 0.1
    geom = geometric_sample(p_geom, SIZE)
    fig, ax = plt.subplots()
    histogram(geom, discrete=True,
              title=f"Geometric, p={p_geom}, n={SIZE}",
              xlabel="Value", ylabel="Counts", ax=ax)
    fig.savefig(f"{OUT}/geometric_dist.png")
    plt.close(fig)
    print(f"[geometric]   mean obs={geom.mean():.3f}  exp={1/p_geom:.3f}")
    print(f"[geometric]   var  obs={geom.var():.3f}  exp={(1-p_geom)/p_geom**2:.3f}")
    print()

    # (b)(a-c) Three methods for the 6-point distribution
    methods = {
        "crude":     six_point_crude,
        "rejection": six_point_rejection,
        "alias":     six_point_alias,
    }

    for name, fn in methods.items():
        samples = fn(SIZE)
        plot_six_point(samples,
                       f"6-point, {name}, n={SIZE}",
                       f"{OUT}/six_point_{name}.png")
        chi = chi_squared(samples, P_SIX)
        t = benchmark(fn, SIZE)
        print(f"[{name:9s}]  chi2={chi.statistic:7.3f}  "
              f"p={chi.pvalue:.3f}  time={t*1000:6.2f} ms")
