# Exercise 7 – Simulated Annealing for the TSP

Simulated annealing (Metropolis–Hastings with a decreasing temperature
schedule) is applied to the travelling salesman problem. The state is a
permutation of cities, the cost is the cyclic tour length
$f(S)=\sum_i D[S_i, S_{i+1}]$. At iteration $k$ a new tour $Y$ is proposed by
swapping two random cities; if $\Delta = f(Y)-f(X) \le 0$ it is accepted,
otherwise it is accepted with probability $\exp(-\Delta / T_k)$. Two cooling
schedules are used: $T_k = 1/\sqrt{1+k}$ and $T_k = 1/\ln(2+k)$. The
best-so-far route is tracked separately, since the schedules never reach 0.
All runs use $N_{\text{iter}} = 20{,}000$.

## Part 1a: Circle sanity check

20 points placed evenly on a unit circle have a known optimal tour: visiting
them in angular order, giving a perimeter of $2N\sin(\pi/N) = 6.2574$.

| | Cost |
|---|---|
| Initial (random) route | 28.2882 |
| Best route (SA) | 6.2574 |
| Polygon perimeter | 6.2574 |

SA recovers the exact optimum. `circle_sanity.png` shows the initial route
(crossing itself many times) and the final route, which is the clean,
non-self-intersecting polygon.

## Part 1b: Random points

20 points drawn uniformly in the unit square:

| | Cost |
|---|---|
| Initial (random) route | 9.5276 |
| Best route (SA) | 4.3009 |

`random_route.png` shows the final route (no obvious crossings) and the
cost-vs-iteration trajectory, which drops sharply within the first ~2000
iterations and then flattens out.

## Part 2: Learn cost matrix

The 20x20 asymmetric cost matrix from Learn was used, starting from the
identity route $0,1,\dots,19$:

| | Cost |
|---|---|
| Initial (identity) route | 3404 |
| Best route (SA) | 1339 |

Best route (1-indexed towns):
`13 18 7 17 3 20 11 16 10 9 19 15 4 1 8 5 2 14 12 6`

### Cooling schedule x proposal mechanism

Mean best cost over 5 seeds:

| Cooling | Proposal | Mean best cost |
|---------|----------|-----------------|
| sqrt | swap | 1201.0 |
| sqrt | reverse | 811.8 |
| log | swap | 1201.0 |
| log | reverse | 811.8 |

The **proposal mechanism dominates**: segment-reversal (2-opt style) reaches
substantially lower costs than plain swaps for both cooling schedules. The
two cooling schedules behave almost identically for this problem size and
iteration budget. `tsp_costmatrix.png` shows the cost history for the best
configuration (sqrt cooling, reverse proposal), converging to ~812 within
about 2000 iterations.
