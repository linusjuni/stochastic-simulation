import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest
from utils.plotting import histogram

def linear_congruential_generator(a, c, M, x0, size):
    output = np.zeros(shape=size)
    output[0] = x0
    for i in range(1, size):
        y = (a * output[i-1] + c)
        val = y - (y // M) * M
        output[i] = val

    return output

def run_and_display_kolmogorov_smirnov_test(output):
    # Normalize the output to the range [0, 1]
    normalized_output = output / np.max(output)
    
    # Perform the Kolmogorov-Smirnov test against a uniform distribution
    ks_statistic, p_value = kstest(normalized_output, 'uniform')
    
    print(f"KS Statistic: {ks_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

if __name__ == "__main__":
    a = 5
    c = 1
    M = 16
    x0 = 1
    size = 10000
    
    output = linear_congruential_generator(a, c, M, x0, size)
    # plot_histogram(output, bins=16)
    histogram(output, bins="auto", discrete=True, title="Histogram of Generated Random Numbers", xlabel="Value", ylabel="Frequency")
    plt.show()

    ##### Kolmogorov-Smirnov test #####
    run_and_display_kolmogorov_smirnov_test(output)
    
    ##### System available random generator #####
    system_output = np.random.rand(size)
    run_and_display_kolmogorov_smirnov_test(system_output)
    
    