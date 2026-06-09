import numpy as np
import matplotlib.pyplot as plt


# from utils.plotting import histogram

np.random.seed(42)

SIZE = 10000
p = 0.1 # probability

def geometric_dist(p, U):
    # sample geometric dist via the Uniform distribution
    x = np.floor((np.log(U)/np.log(1-p))) + 1
    return x

if __name__ == "__main__":
    dist = np.zeros(shape=SIZE)
    for i in range(SIZE):
        U_i = np.random.random()
        dist[i] = geometric_dist(p, U_i)

    # histogram(dist, bins="auto", title="Geometric dist 10000 samples", xlabel="Value", y="Counts")
    plt.hist(x=dist, bins=30)
    plt.savefig("./geometric_dist.png")
    plt.show()

    