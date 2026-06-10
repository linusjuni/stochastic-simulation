# Exercise 1 — LCG Results

LCG recurrence: `x_i = (a * x_{i-1} + c) mod m`, `U_i = x_i / m`, n=10,000, seed=67.

Pass threshold: p > 0.05. Fail marked **bold**.

## Test results

| Name | a | c | m | Chi² p | KS p | Run p | Corr lag 1 p | Corr lag 2 p | Corr lag 5 p | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Numerical Recipes | 1664525 | 1013904223 | 2³² | 0.181 | 0.113 | 0.317 | **0.001** | **0.019** | **0.004** | Full period (Hull-Dobell). Uniform but subtle serial correlation at short lags. |
| 42, 67, 69 | 42 | 67 | 69 | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | Period=22 — cycles 454× in n=10,000. Only 22 unique values. Everything fails. |
| Degenerate (a=1) | 1 | 1 | 2³² | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | Pure counter: U_i ≈ (seed+i)/m, nearly zero variance. Every test fails catastrophically. |
| Poor multiplicative | 3 | 0 | 2¹⁶ | 0.585 | 0.158 | **0.000** | **0.000** | 0.051 | **0.014** | Passes uniformity but strong serial correlation and run clustering. Short effective period. |
