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
    for i in range(1, SIZE):
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

def alias_method()
    

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