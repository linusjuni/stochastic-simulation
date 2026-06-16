# Exercise 8 – Bootstrapping

The bootstrap is a non-parametric technique for estimating the variance (and
standard error) of an estimator when no analytical expression is available. We
replace the unknown distribution by the **empirical distribution** $F_e$, which
puts mass $1/n$ on each observation, and resample from it: a bootstrap sample is
$n$ draws **with replacement** from the data. For $k$ bootstrap samples we
compute replicates $\hat\theta^*_i$ and estimate the variance with the sample
variance of the replicates,
$\hat V[\hat\theta] = \frac{1}{k-1}\sum_{i=1}^k (\hat\theta^*_i - \bar\theta^*)^2$.

## Part 1 — Ross Ch. 8, Exercise 13

Estimate $p = P\{a < \frac1n\sum X_i - \mu < b\}$ with $n=10$,
$a=-5$, $b=5$, data $56,101,78,67,93,87,64,72,80,69$.

Under $F_e$ the mean equals the sample mean $\bar x$, so we replace the unknown
$\mu$ by $\bar x$ and estimate $p$ as the proportion of bootstrap resample means
with $a < \bar X^* - \bar x < b$ ($k=10{,}000$).

| Quantity | Value |
|----------|-------|
| $\bar x$ | 76.70 |
| $\hat p$ | 0.7658 |

## Part 2 — Ross Ch. 8, Exercise 15

Bootstrap estimate of $\mathrm{Var}(S^2)$ for $n=15$, data
$5,4,9,6,21,17,11,20,7,10,21,15,13,16,8$, with $S^2$ the sample variance
(divisor $n-1$). We resample, recompute $S^{2*}$ for each bootstrap sample, and
take the sample variance of the replicates ($k=10{,}000$).

| Quantity | Value |
|----------|-------|
| observed $S^2$ | 34.31 |
| $\widehat{\mathrm{Var}}(S^2)$ | 57.92 |

As a check, the closed-form case of Exercise 14 ($n=2$, $X=\{1,3\}$) gives
$\mathrm{Var}(S^2)=1$; the bootstrap program reproduces $\approx 1.00$.

## Part 3 — Sample mean vs. sample median (Pareto)

A program returns the sample median together with a bootstrap estimate of its
variance ($k=100$ replicates), run on $n=200$ Pareto($\beta=1,k=1.05$)
observations (sampled by inversion $X=\beta U^{-1/k}$).

| Estimator | Estimate | Bootstrap variance | Bootstrap std. error |
|-----------|----------|--------------------|----------------------|
| Sample mean   | 7.39 | 8.642 | 2.940 |
| Sample median | 2.06 | 0.018 | 0.136 |

The bootstrap variance of the mean is roughly 470 times that of the median. For
Pareto with $k=1.05$ the mean is finite ($\beta k/(k-1)=21$) but the variance is
infinite ($k<2$): the distribution is extremely heavy-tailed (the sample's
largest value here is $\approx 678$). A few rare, huge observations dominate the
sample mean, making it imprecise and biased low (7.39 vs. the true 21) for
$n=200$. The sample median depends only on the central order statistics, so it
is robust to the heavy tail and far more precise — exactly the situation the
bootstrap is designed to quantify.
