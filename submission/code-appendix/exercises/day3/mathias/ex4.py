import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import expon, norm, pareto, kstest, chi2
from scipy.stats import t as student_t
from math import factorial

from utils.plotting import histogram

np.random.seed(42)  # For reproducibility


def erlang_dist(m, mu):
    X = np.random.exponential(scale=mu, size=(m))
    return np.sum(X)

def poisson_dist(mu, size):
    dist = np.random.poisson(lam=mu, size=size)
    return dist

def simulate(arrival_process, service_process, num_services, customers):
    # initiate time for each service)
    service_times = np.zeros(num_services)
    t = 0.0
    blocked = 0
    for _ in range(customers):
        t = t + arrival_process()
        
        if np.min(service_times > t):
            blocked += 1
            
        else:
            service_ID = np.argmin(service_times) # queue service with earliest finishing time
            service_times[service_ID] = t + service_process()

    return blocked / customers

    return np.random.exponential(scale=mu_service)

def arrival_process_1():
    return np.random.exponential(scale=mu_arrival)

def service_process_1():
    return np.random.exponential(scale=mu_service)
            
def CI_fractions(fractions, confidence=0.95):
    mean = np.mean(fractions)
    var  = np.var(fractions, ddof=1)
    n = len(fractions)
    t_quantile = student_t.ppf(0.975, df=n - 1)
    return (mean - t_quantile * np.sqrt(var / n), mean + t_quantile * np.sqrt(var / n))

def arrival_process_2(k):
    return erlang_dist(m=k, mu=1/k)

def service_process_2_a():
    return service_process_1()

def service_process_2_b():
    return service_process_1()

def arrival_process_2_b(p1, lam1, p2, lam2):
    U = np.random.random()
    if U < p1:
        return np.random.exponential(scale=1/lam1)
    else:
        return np.random.exponential(scale=1/lam2)

def arrival_process_3():
    return arrival_process_1()

def service_process_3_a():
    return 8.0

def service_process_3_b(k, beta=1.0):
    return np.random.pareto(a=k) * beta

def service_process_3_c():
    return np.random.uniform(0, 16)

def service_process_3_d():
    return np.random.chisquare(df=8)   # mean = df = 8

def erlang_analytic_solution(lam_, m):
    num = (lam_ ** m) / factorial(m)
    terms = [(lam_ ** i) / factorial(i) for i in range(m + 1)]
    denom = sum(terms)
    return num / denom


def banner(text):
    print("\n" + "=" * 72)
    print(f" {text}")
    print("=" * 72)


def report(label, fractions, ref=None, store=None):
    mean = np.mean(fractions)
    lo, hi = CI_fractions(fractions)
    suffix = f"   (Erlang B ref = {ref:.4f})" if ref is not None else ""
    print(f"  {label:<22} blocking_fraction = {mean:.4f}   95% CI = [{lo:.4f}, {hi:.4f}]{suffix}")
    if store is not None:
        store.append((label, mean, lo, hi))



if __name__ == "__main__":

    RUNS = 10 # repeat simulation 10 times to get confidence intervals on blocking probability
    NO_OF_SERVICES = 10
    mu_arrival = 1.0
    mu_service = 8.0
    CUSTOMERS = 10000

    # Analytic reference (Erlang B). Note: argument is the buggy one for now;
    # the audit flagged this — left unchanged on purpose.
    lam_ = mu_service / mu_arrival  # offered traffic A = lambda / mu
    erlang_blocking = erlang_analytic_solution(lam_, m=NO_OF_SERVICES)

    results = []   # collects (label, mean, ci_lo, ci_hi) for the summary plot

    # 1) Poisson arrivals, exponential service
    banner("Task 1  —  Poisson arrivals, Exp(mean=8) service")
    bp = [simulate(arrival_process_1, service_process_1, NO_OF_SERVICES, CUSTOMERS) for _ in range(RUNS)]
    report("Poisson / Exp", bp, ref=erlang_blocking, store=results)

    # 2a) Erlang inter-arrivals (renewal process)
    banner("Task 2(a)  —  Erlang(k) inter-arrivals (mean 1), Exp service")
    for k_val in [1, 2, 5, 10]:
        bp = [simulate(lambda: arrival_process_2(k_val), service_process_2_a, NO_OF_SERVICES, CUSTOMERS) for _ in range(RUNS)]
        report(f"Erlang k={k_val}", bp, store=results)

    # 2b) Hyper-exponential inter-arrivals
    banner("Task 2(b)  —  Hyper-exp inter-arrivals, Exp service")
    p1, lam1 = 0.8, 0.83333
    p2, lam2 = 0.2, 5.0
    bp = [simulate(lambda: arrival_process_2_b(p1, lam1, p2, lam2), service_process_2_b, NO_OF_SERVICES, CUSTOMERS) for _ in range(RUNS)]
    report("Hyper-exp", bp, store=results)

    # 3a) Constant service
    banner("Task 3(a)  —  Poisson arrivals, constant service = 8")
    bp = [simulate(arrival_process_3, service_process_3_a, NO_OF_SERVICES, CUSTOMERS) for _ in range(RUNS)]
    report("Constant 8.0", bp, ref=erlang_blocking, store=results)

    # 3b) Pareto service
    banner("Task 3(b)  —  Poisson arrivals, Pareto service")
    for k_val in [1.05, 1.5, 2.05, 3.0]:
        bp = [simulate(arrival_process_3, lambda: service_process_3_b(k_val), NO_OF_SERVICES, CUSTOMERS) for _ in range(RUNS)]
        report(f"Pareto k={k_val}", bp, ref=erlang_blocking, store=results)

    # 3c) Uniform(0,16) service  mean is 8
    banner("Task 3(c)  —  Poisson arrivals, Uniform(0,16) service")
    bp = [simulate(arrival_process_3, service_process_3_c, NO_OF_SERVICES, CUSTOMERS) for _ in range(RUNS)]
    report("Uniform(0,16)", bp, ref=erlang_blocking, store=results)

    # 3d) Chi-squared(df=8) service  (mean 8 → same offered traffic as Part 1)
    banner("Task 3(d)  —  Poisson arrivals, Chi-squared(df=8) service")
    bp = [simulate(arrival_process_3, service_process_3_d, NO_OF_SERVICES, CUSTOMERS) for _ in range(RUNS)]
    report("Chi2(df=8)", bp, ref=erlang_blocking, store=results)

    # 4) Analytic Erlang B (verification reference for Poisson-arrival cases)
    banner("Task 4  —  Erlang B analytic reference")
    print(f"  A used:    lam_ = {lam_}   (m = {NO_OF_SERVICES})")
    print(f"  Erlang B:  {erlang_blocking:.4f}")
    print("  Valid for any service distribution PROVIDED arrivals are Poisson.")

    # ----- Summary plot: all blocking fractions with 95% CI -----
    labels = [r[0] for r in results]
    means  = np.array([r[1] for r in results])
    los    = np.array([r[2] for r in results])
    his    = np.array([r[3] for r in results])
    yerr   = np.vstack([means - los, his - means])

    fig, ax = plt.subplots(figsize=(13, 5))
    xs = np.arange(len(labels))
    ax.errorbar(xs, means, yerr=yerr, fmt="o", capsize=4, color="C0")
    ax.axhline(erlang_blocking, color="red", linestyle="--", linewidth=1,
               label=f"Erlang B = {erlang_blocking:.4f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(r"blocking fraction $\hat{p}$")
    ax.set_title("Blocking fraction with 95% CI across all experiments")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("exercises/day3/mathias/blocking_summary.png", dpi=120)
    
    