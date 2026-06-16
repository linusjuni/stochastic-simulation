# Potential changes to report/exercises/day2.tex

## 1. Discrete slide references are wrong (systematic off-by-3)

All slide references in the Exercise 2 / discrete section are too low by ~3. The continuous references (slides 5, 6, 7, 9, 11) are all correct — the error is exclusive to the discrete PDF.

| Location in report | Report says | Actual slide in PDF | Content on that slide |
|---|---|---|---|
| Line 21, geometric CDF | "slide 9" | 12/22 | Geometric distribution derivation |
| Line 97, crude method | "slide 8" | 11/22 | Discrete distributions: Direct (crude) method |
| Line 106, rejection algorithm | "slide 11" | 15/22 | Simple rejection algorithm |
| Line 106, rejection correctness | "slide 18" | 16/22 | General rejection / correctness proof |
| Line 113, alias method | "slides 14--16" | ~17--18/22 | Alias method description |

**Fix:** Update slide numbers in the five locations above.

---

## 2. `\section` missing `*` (line 393)

```latex
% current (wrong)
\section{Part 4 --- Pareto via Composition}

% should be
\section*{Part 4 --- Pareto via Composition}
```

Every other heading in the file uses `\section*{}` (unnumbered). This one is missing the `*`, so it will render as a numbered section in the compiled PDF, breaking the visual consistency.

---

## Things verified as correct (no change needed)

- **Geometric formula**: `np.ceil(log(u)/log(1-p))` is equivalent to ⌊log(U)/log(1-p)⌋ + 1 from the slides — for continuous U, ceil(x) = floor(x)+1 almost surely.
- **Six-point probabilities**: Report fractions (7/48, 5/48, 1/8, 1/16, 1/4, 5/16) match code exactly.
- **Crude sample**: `searchsorted(cdf, u, side="left")` correctly implements F(x_{i-1}) < U ≤ F(x_i).
- **Rejection**: Using c = max p_i (equality rather than strict >) is fine — the mode is always accepted, distribution is still correct.
- **Rejection acceptance rate**: 1/(6 × 5/16) ≈ 0.53 → ~1.88 proposals → ~3.75 uniforms per draw is correct.
- **Alias build/sample**: Code matches the report snippet and slide algorithm.
- **Exponential sampler**: -log(U)/λ is correct (uses 1-U ~ U symmetry from slide derivation).
- **Pareto inversion**: β U^{-1/k} matches the slide and code.
- **Box-Muller**: Formula and code match slides 9 and 11 of the continuous PDF.
- **Pareto composition**: Mathematically verified — Λ ~ Gamma(k, 1/β), X = β + Exp(Λ) gives P(X > x) = (β/x)^k (Pareto Type I). KS p-value 0.860 confirms it.
- **Pareto moments**: E[X] = βk/(k-1), V[X] = β²k/((k-1)²(k-2)) match in slides, code, and report.
- **Confidence intervals**: t-pivot for mean, χ²-pivot for variance — formulas, code, and report all consistent.
