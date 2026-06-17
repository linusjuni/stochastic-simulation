import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from utils.plotting import histogram

np.random.seed(42)

SIZE = 10000
p = 0.1 # probability

def geometric_dist(p, U):
    # sample geometric dist via the Uniform distribution
    x = np.floor((np.log(U)/np.log(1-p))) + 1
    return x

def six_point_dist_rejection_sampling(SIZE):
    p_x = {
        1: 7/48,
        2: 5/48,
        3: 1/8,
        4: 1/16,
        5: 1/4,
        6: 5/16
    }
    # find the maximum probability
    C = max(p_x.values())
    dist = []
    for i in range(SIZE):
        U1 = np.random.random()
        U2 = np.random.random()
        candidate = int(np.floor(U1 * 6)) + 1
        if U2 < p_x[candidate] / C:
            dist.append(candidate)
    
    return dist

def six_point_dist_direct_crude_method(p_x):
    U = np.random.random()
    cdf_x = np.cumsum(list(p_x.values()))
    for i in range(0, len(cdf_x)):
        if U < cdf_x[i]:
            return i + 1

def build_alias_tables(p):
    """
    Slide 16. p: length-k probabilities (sum to 1).
    Returns F, L. Both 0-indexed.
    """
    k = len(p)
    F = np.array(p, dtype=float) * k     # step 2: scaled probs, mean = 1
    L = np.arange(k)                      # step 1: each class is its own alias initially

    # step 3: partition into surplus (G) and deficit (S)
    G = [i for i in range(k) if F[i] >= 1]
    S = [i for i in range(k) if F[i] <  1]

    # step 4: pair deficits with surpluses until S is empty
    while S:
        i = G[0]   # a surplus class — the donor
        j = S[0]   # a deficit class — the receiver

        L[j] = i                          # bucket j's top is aliased to i
        F[i] = F[i] - (1 - F[j])          # i donated (1 - F[j]) of its mass

        if F[i] < 1 - 1e-12:              # i dropped below 1 → now a deficit
            G.pop(0)
            S.append(i)

        S.pop(0)                          # j is fully assigned

    return F, L


def sample_alias(F, L):
    """
    Slide 14-15. One draw. Returns a 0-indexed class.
    """
    k = len(F)
    U1, U2 = np.random.random(), np.random.random()
    I = int(np.floor(k * U1))             # pick a bucket uniformly
    return I if U2 <= F[I] else L[I]      # bottom of bucket vs top


def six_point_dist_alias(SIZE):
    p = np.array([7/48, 5/48, 1/8, 1/16, 1/4, 5/16])
    F, L = build_alias_tables(p)
    samples = np.zeros(SIZE, dtype=int)
    for n in range(SIZE):
        samples[n] = sample_alias(F, L) + 1   # +1 for 1-indexed display
    return samples



if __name__ == "__main__":
    dist = np.zeros(shape=SIZE)
    for i in range(SIZE):
        U_i = np.random.random()
        dist[i] = geometric_dist(p, U_i)
    # Theoretical mean and variance of geometric distribution
    theoretical_mean = 1/p
    theoretical_variance = (1-p) / (p**2)
    print(f"Theoretical mean: {theoretical_mean}, Theoretical variance: {theoretical_variance}")
    # Sample mean and variance
    print(f"Sample mean: {np.mean(dist)}, Sample variance: {np.var(dist)}")
    
 
    histogram(dist, bins="auto", discrete=True, title="Geometric dist 10000 samples, p = 0.1", xlabel="Value", ylabel="Counts")
    # plt.hist(x=dist, bins=30)
    plt.savefig("exercises/day2/mathias/geometric_dist.png")
    # plt.show()

    six_point_dist = six_point_dist_rejection_sampling(SIZE)
    p_x = {
    1: 7/48,
    2: 5/48,
    3: 1/8,
    4: 1/16,
    5: 1/4,
    6: 5/16
    }
    theoretical_mean = sum([k * v for k, v in p_x.items()])
    theoretical_variance = sum([k**2 * v for k, v in p_x.items()]) - theoretical_mean**2

    print(f"Theoretical mean: {theoretical_mean}, Theoretical variance: {theoretical_variance}")
    print(f"Sample mean: {np.mean(six_point_dist)}, Sample variance: {np.var(six_point_dist)}")
    # plot the distribution of the six point distribution in the same histogram but different colors
    
    histogram(six_point_dist, bins="auto", discrete=True, title="Six point distribution via rejection sampling, 10000 samples", xlabel="Value", ylabel="Counts")
    plt.savefig("exercises/day2/mathias/six_point_dist.png")
    plt.show()
    
    six_point_dist_direct = []
    for i in range(SIZE):
        six_point_dist_direct.append(six_point_dist_direct_crude_method(p_x))
    
    print(f"Theoretical mean: {theoretical_mean}, Theoretical variance: {theoretical_variance}")
    print(f"Sample mean: {np.mean(six_point_dist_direct)}, Sample variance: {np.var(six_point_dist_direct)}")


    
    histogram(six_point_dist_direct, bins="auto", discrete=True, title="Six point distribution via direct crude method, 10000 samples", xlabel="Value", ylabel="Counts")
    plt.savefig("exercises/day2/mathias/six_point_dist_direct.png")
    plt.show()

    six_point_dist_alias_samples = six_point_dist_alias(SIZE)
    print(f"Theoretical mean: {theoretical_mean}, Theoretical variance: {theoretical_variance}")
    print(f"Sample mean: {np.mean(six_point_dist_alias_samples)}, Sample variance: {np.var(six_point_dist_alias_samples)}")

    histogram(six_point_dist_alias_samples, bins="auto", discrete=True, title="Six point distribution via alias method, 10000 samples", xlabel="Value", ylabel="Counts")
    plt.savefig("exercises/day2/mathias/six_point_dist_alias.png")
    plt.show()