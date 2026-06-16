import numpy as np
import scipy.optimize
import scipy.stats as st
from math import factorial

from utils.logger import get_logger
from utils.plotting import figure
from utils.settings import settings

logger = get_logger(__name__)
rng = np.random.default_rng(settings.SEED)
TRUE_VALUE = np.e - 1
PLOTS = "exercises/day4/linus/plots"


def ci(samples: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, lo, hi, sample_var) for a 95% t-CI."""
    n = len(samples)
    mean = float(samples.mean())
    var = float(samples.var(ddof=1))
    hw = st.t.ppf(0.975, df=n - 1) * np.sqrt(var / n)
    return mean, mean - hw, mean + hw, var


# Parts 1–4: variance reduction for ∫₀¹ eˣ dx
def part1_crude(n: int = 100) -> tuple[float, float, float, float]:
    u = rng.uniform(size=n)
    return ci(np.exp(u))


def part2_antithetic(n_pairs: int = 50) -> tuple[float, float, float, float]:
    u = rng.uniform(size=n_pairs)
    return ci((np.exp(u) + np.exp(1 - u)) / 2)


def part3_control_variate(n: int = 100) -> tuple[float, float, float, float]:
    u = rng.uniform(size=n)
    x = np.exp(u)
    c = -np.cov(x, u, ddof=1)[0, 1] / np.var(u, ddof=1)
    logger.info("Control variate coefficient", c=round(c, 4))
    return ci(x + c * (u - 0.5))


def part4_stratified(n: int = 100, k: int = 10) -> tuple[float, float, float, float]:
    n_reps = n // k
    ys = np.empty(n_reps)
    for i in range(n_reps):
        v = rng.uniform(size=k)
        u = (np.arange(k) + v) / k
        ys[i] = np.exp(u).mean()
    return ci(ys)


# Part 5: control variate on the blocking system
def erlang_b(offered_load: float, m: int) -> float:
    a = offered_load
    num = a**m / factorial(m)
    return num / sum(a**i / factorial(i) for i in range(m + 1))


def simulate_blocking(m: int, n: int) -> tuple[float, float]:
    """M/M/m/m queue. Returns (blocking_fraction, sum_of_interarrivals)."""
    server_free_at = np.zeros(m)
    t = 0.0
    blocked = 0
    total_interarrivals = 0.0
    for _ in range(n):
        dt = rng.exponential(1.0)
        total_interarrivals += dt
        t += dt
        if server_free_at.min() <= t:
            idx = server_free_at.argmin()
            server_free_at[idx] = t + rng.exponential(8.0)
        else:
            blocked += 1
    return blocked / n, total_interarrivals


def part5_blocking_cv(n_customers: int = 10_000, n_reps: int = 100) -> None:
    m = 10

    # pilot run to estimate c = Cov(B, S) / Var(S)
    pilot_b, pilot_s = [], []
    for _ in range(n_reps):
        b, s = simulate_blocking(m, n_customers)
        pilot_b.append(b)
        pilot_s.append(s)
    c = np.cov(pilot_b, pilot_s)[0, 1] / np.var(pilot_s, ddof=1)
    logger.info("Blocking CV coefficient", c=round(c, 6))

    # main run: Y_i = B_i - c * (S_i - n)
    bs, ys = [], []
    for _ in range(n_reps):
        b, s = simulate_blocking(m, n_customers)
        bs.append(b)
        ys.append(b - c * (s - n_customers))

    crude_mean, crude_lo, crude_hi, _ = ci(np.array(bs))
    cv_mean, cv_lo, cv_hi, _ = ci(np.array(ys))

    logger.info(
        "Blocking crude",
        mean=round(crude_mean, 6),
        ci_width=round(crude_hi - crude_lo, 6),
    )
    logger.info("Blocking CV", mean=round(cv_mean, 6), ci_width=round(cv_hi - cv_lo, 6))
    logger.info("Erlang B reference", value=round(erlang_b(8.0, 10), 6))


# Part 7: importance sampling for P(Z > a)
def part7_tail_probability() -> None:
    for n in [100, 1_000, 10_000]:
        for a in [2, 4]:
            true_p = float(st.norm.sf(a))

            # crude MC: draw Z ~ N(0,1), check Z > a
            z = rng.standard_normal(size=n)
            crude_mean, crude_lo, crude_hi, _ = ci((z > a).astype(float))
            logger.info(
                "P(Z>a) crude",
                n=n,
                a=a,
                true=f"{true_p:.2e}",
                mean=f"{crude_mean:.2e}",
                ci_width=f"{crude_hi - crude_lo:.2e}",
            )

            # IS: draw X ~ N(a, sigma2), weight by f(X)/g(X)
            for sigma2 in [1, 2, 3]:
                x = rng.normal(loc=a, scale=np.sqrt(sigma2), size=n)
                w = st.norm.pdf(x, 0, 1) / st.norm.pdf(x, a, np.sqrt(sigma2))
                is_mean, is_lo, is_hi, _ = ci((x > a).astype(float) * w)
                logger.info(
                    "P(Z>a) IS",
                    n=n,
                    a=a,
                    sigma2=sigma2,
                    mean=f"{is_mean:.2e}",
                    ci_width=f"{is_hi - is_lo:.2e}",
                )


# Part 8: IS with Exp(lambda) for ∫₀¹ eˣ dx
def _var_w(lam: float) -> float:
    """Analytical variance of IS weight W when proposal is Exp(lambda)."""
    return (np.e ** (2 + lam) - 1) / (lam * (2 + lam)) - (np.e - 1) ** 2


def part8_is_exponential(n: int = 10_000) -> None:
    res = scipy.optimize.minimize_scalar(_var_w, bounds=(0.01, 20), method="bounded")
    lam_opt = float(res.x)
    logger.info(
        "Optimal lambda", lam_opt=round(lam_opt, 4), var_w=round(float(res.fun), 4)
    )

    # crude benchmark at same n
    crude_mean, crude_lo, crude_hi, _ = part1_crude(n)
    logger.info(
        "IS Exp crude MC",
        mean=round(crude_mean, 6),
        ci_width=round(crude_hi - crude_lo, 6),
    )

    # IS estimator: W = e^{(1+lam)*x} / lam for x ~ Exp(lam), x <= 1
    for lam in [0.5, 1.0, lam_opt, 2.0, 5.0]:
        x = rng.exponential(1 / lam, size=n)
        w = (x <= 1).astype(float) * np.exp(x * (1 + lam)) / lam
        is_mean, is_lo, is_hi, _ = ci(w)
        logger.info(
            "IS Exp",
            lam=round(lam, 3),
            mean=round(is_mean, 6),
            ci_width=round(is_hi - is_lo, 6),
        )

    lams = np.linspace(0.1, 8, 500)
    with figure(figsize=(7, 4), save=f"{PLOTS}/var_w_vs_lambda.png") as fig:
        ax = fig.add_subplot(111)
        ax.plot(lams, [_var_w(lam) for lam in lams], lw=2, label="Var(W)")
        ax.axvline(lam_opt, color="tomato", linestyle="--", label=f"λ* = {lam_opt:.3f}")
        ax.axhline(
            0.2420, color="steelblue", linestyle="--", label="crude MC Var ≈ 0.242"
        )
        ax.set_xlabel("λ")
        ax.set_ylabel("Var(W)")
        ax.set_title("IS variance vs λ for Exp(λ) proposal")
        ax.set_ylim(0, 3)
        ax.legend()


def main() -> None:
    logger.info("Exercise 5 — Variance Reduction", seed=settings.SEED)

    # Parts 1–4: compare CI widths at ~100 function evaluations
    logger.info("Parts 1-4: variance reduction for integral of e^x on [0,1]")
    names = ["Crude MC", "Antithetic", "Control variate", "Stratified"]
    fns = [part1_crude, part2_antithetic, part3_control_variate, part4_stratified]
    results = []
    for name, fn in zip(names, fns):
        mean, lo, hi, var = fn()
        results.append((name, mean, lo, hi))
        logger.info(
            name,
            mean=round(mean, 6),
            lo=round(lo, 6),
            hi=round(hi, 6),
            ci_width=round(hi - lo, 6),
            true=round(TRUE_VALUE, 6),
        )

    with figure(figsize=(8, 4), save=f"{PLOTS}/ci_comparison.png") as fig:
        ax = fig.add_subplot(111)
        for i, (name, mean, lo, hi) in enumerate(results):
            ax.plot([lo, hi], [i, i], lw=2.5, color="steelblue")
            ax.plot(mean, i, "o", color="steelblue", markersize=6)
        ax.axvline(TRUE_VALUE, color="tomato", linestyle="--", lw=1.5,
                   label=f"true $e-1 = {TRUE_VALUE:.4f}$")
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Estimate")
        ax.set_title("95% confidence intervals (100 function evaluations)")
        ax.legend()

    # Part 5
    logger.info("Part 5: blocking system control variate")
    part5_blocking_cv()

    # Part 7
    logger.info("Part 7: IS for P(Z > a)")
    part7_tail_probability()

    # Part 8
    logger.info("Part 8: IS with Exp(lambda) for integral of e^x")
    part8_is_exponential()


if __name__ == "__main__":
    main()
