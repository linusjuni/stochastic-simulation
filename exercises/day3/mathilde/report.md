# Exercise 4 – Discrete Event Simulation of a Blocking System

System: m = 10 servers, no waiting room, mean inter-arrival = 1, mean service
time = 8 ⇒ offered traffic A = 8 Erlang. The simulation follows the
**event-by-event principle**: an event list (priority queue) of arrival and
departure events, a state variable (number of busy servers) and a statistical
accumulator (blocked count). Each configuration is run as **10 × 10,000
customers**; the 10 runs are treated as independent **sub-samples** and give a
95% CI from a t-distribution over the 10 per-run blocked fractions.

**Erlang B (A=8, m=10) = 0.1217**

## Part 1: Verification (Poisson arrivals, exponential service)

| Configuration         | Fraction | 95% CI            | Erlang B in CI? |
|------------------------|----------|-------------------|-----------------|
| Poisson / Exponential  | 0.1221   | [0.1184, 0.1258]  | Yes             |

The simulation matches the analytical Erlang B formula closely, confirming the
event-by-event implementation is correct.

## Part 2: Renewal arrival processes (exponential service)

| Configuration              | Fraction | 95% CI            | Erlang B in CI? |
|-----------------------------|----------|-------------------|-----------------|
| Erlang(2) arrivals          | 0.0950   | [0.0911, 0.0990]  | No (lower)      |
| Hyperexponential arrivals   | 0.1398   | [0.1347, 0.1449]  | No (higher)     |

Erlang B requires **Poisson arrivals**; here it no longer applies.
- **Erlang(2)** inter-arrival times have *lower variance* than exponential
  (more regular, less bursty), so customers are more evenly spaced ⇒ fewer
  collisions ⇒ **less blocking** than Erlang B predicts.
- **Hyperexponential** inter-arrival times have *higher variance* (a mix of
  very short and very long gaps — bursty traffic), causing customers to
  cluster ⇒ **more blocking** than Erlang B predicts.

This illustrates that blocking probability depends on the *variability* of the
arrival process, not just its mean rate.

## Part 3: Poisson arrivals, varied service distributions

| Configuration         | Fraction | 95% CI            | Erlang B in CI? |
|-------------------------|----------|-------------------|-----------------|
| Constant service        | 0.1202   | [0.1157, 0.1247]  | Yes             |
| Pareto k=1.05            | 0.0012   | [0.0002, 0.0022]  | No (lower)      |
| Pareto k=2.05            | 0.1178   | [0.1126, 0.1229]  | Yes             |
| Lognormal service        | 0.1208   | [0.1160, 0.1257]  | Yes             |

Erlang B is **insensitive to the service-time distribution** (only the mean
matters) — confirmed for constant, Pareto k=2.05, and lognormal service, all
of which match Erlang B.

**Pareto k=1.05 is the exception**, and it is *not* a violation of
insensitivity but a **finite-sample bias problem**. With k=1.05 (k-1 = 0.05),
the theoretical mean E[X] = β·k/(k-1) = 8 is finite, but the variance is
infinite (k < 2). The distribution is so heavy-tailed that almost all draws
are tiny (β ≈ 0.38), and the rare huge values needed to pull the sample mean
up to 8 essentially never appear in a sample of 10,000. The *empirical* mean
service time is therefore far below 8, so the system is effectively
under-loaded and blocking collapses to ≈ 0. This is the "problem" the
exercise asks to identify: with k close to 1, simulation results for a finite
n severely underestimate the true mean and hence the true blocking probability.

## Part 4: Comparison and Interpretation

See `comparison.png` for all configurations plotted against the Erlang B
reference line. Summary:

- **CI widths** are similar (≈ 0.004–0.005) for all "well-behaved"
  distributions (Part 1, constant, Pareto k=2.05, lognormal), reflecting
  similar run-to-run variability at A=8.
- **Pareto k=1.05** has a much *narrower* absolute CI but is centered far from
  Erlang B — its low value is a systematic bias, not high variance, *for this
  blocking metric*. (The service-time sample mean itself would have huge
  run-to-run variance.)
- **Part 2 CIs do not contain Erlang B** at all, confirming that the formula's
  Poisson-arrival assumption is essential — the direction of the deviation
  (down for regular arrivals, up for bursty arrivals) is consistent with
  arrival-process variance driving blocking probability.
