# Exercise 2 – Discrete Random Variables

## Part 1: Geometric Distribution

Three values of p were simulated (n = 10,000 each). Sample means match the
theoretical mean 1/p in all cases, and chi-square goodness-of-fit tests show
no evidence against the geometric distribution.

| p   | Sample mean | Theory (1/p) | χ²     | p-value |
|-----|-------------|--------------|--------|---------|
| 0.1 | 10.058      | 10.000       | 30.68  | 0.949   |
| 0.3 | 3.323       | 3.333        | 16.65  | 0.275   |
| 0.7 | 1.429       | 1.429        | 9.87   | 0.452   |

Smaller p produces a heavier right tail (larger spread), while larger p
concentrates mass near k = 1.

## Part 2: Six-Point Distribution

Target probabilities: X ∈ {1,...,6} with p = [7/48, 5/48, 6/48, 3/48, 12/48, 15/48].

All three methods were simulated with n = 10,000:

| Method    | χ² (n=10k) | p-value |
|-----------|------------|---------|
| Crude     | 5.64       | 0.342   |
| Rejection | 13.20      | 0.022   |
| Alias     | 9.28       | 0.098   |

The rejection p-value of 0.022 is a random-seed artefact (type I error at 5%
level); with n = 1,000,000 all three methods pass convincingly (see Part 3).

## Part 3: Comparison (n = 1,000,000)

| Method    | χ²    | p-value | Time (s) |
|-----------|-------|---------|----------|
| Crude     | 1.96  | 0.854   | 0.034    |
| Rejection | 5.06  | 0.408   | 0.047    |
| Alias     | 3.67  | 0.598   | 0.018    |

All three methods are statistically correct. The alias method is fastest
(≈ 2× over rejection), with the crude method in between.

## Part 4: Recommendations

**Crude (direct CDF inversion)**
- Setup: O(k), draw: O(log k) with binary search (or O(k) with linear scan).
- Best for: small k, one-time or infrequent sampling, or when the distribution
  changes between calls. No extra memory overhead.
- Drawback: scales poorly with large k.

**Rejection method**
- Setup: O(k), draw: O(1) expected but with expected 1/(k·max(p)) ≈ 1.88
  uniforms per accepted sample for this distribution. Wasteful when probabilities
  are very uneven (high max(p) means many rejections).
- Best for: moderate k, uniform-ish distributions, situations where simplicity
  of implementation matters.
- Drawback: sampling cost is unpredictable and grows with pmf unevenness.

**Alias method**
- Setup: O(k), draw: exactly 2 uniforms per sample (O(1) worst case).
- Best for: large k or repeated high-volume sampling from a fixed distribution.
  Constant draw cost regardless of k or pmf shape.
- Drawback: O(k) memory and a slightly more complex setup; not suited for
  distributions that change frequently.

**Summary rule**: for a fixed pmf sampled many times, use alias. For a small
or changing pmf, crude is simpler and fast enough. Rejection sits in between
and is rarely the best choice unless the distribution is nearly uniform.
