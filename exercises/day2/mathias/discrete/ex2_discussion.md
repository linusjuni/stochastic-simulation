# Exercise 2 — Discussion of parts (c) and (d)

Target distribution on `{1..6}`:

| X    | 1     | 2     | 3    | 4    | 5    | 6     |
|------|-------|-------|------|------|------|-------|
| pᵢ   | 7/48  | 5/48  | 1/8  | 1/16 | 1/4  | 5/16  |

All experiments use `n = 10,000` samples, seed `42`.

---

## (c) Comparison of the three methods

### Criteria

1. **Correctness** — does the empirical distribution match the target? (chi-squared goodness-of-fit, 5 d.o.f.)
2. **Runtime** — wall-clock for `n = 10,000` draws (best of 5).
3. **Random numbers consumed** — uniforms per accepted sample.
4. **Setup cost** — work done once before sampling.
5. **Scaling with `k`** — how the method behaves as the support size grows.

### Results

| Method     | χ²    | p-value | Time (ms) | Uniforms / sample      | Setup |
|------------|-------|---------|-----------|------------------------|-------|
| Crude      | 2.851 | 0.723   |  0.18     | 1                      | O(k) cum. sum |
| Rejection  | 1.073 | 0.956   | 29.25     | 2 / acceptance ≈ 3.75  | none  |
| Alias      | 1.136 | 0.951   |  6.80     | 2                      | O(k) table build |

### Discussion

**Correctness.** All three p-values are large (≥ 0.72), so none of the methods can be rejected at any reasonable level. The empirical bars sit on top of the target bars across all six classes (see `six_point_{crude,rejection,alias}.png`). Statistically the methods are indistinguishable.

**Runtime.** Crude is ~40× faster than rejection and ~38× faster than alias here — but that gap is an artefact of implementation, not theory. Crude is fully vectorised via `np.searchsorted`, while rejection and alias use per-sample Python loops. In a compiled language (or with vectorised rejection/alias) the gap would shrink dramatically. What is *robust* across implementations is:

- Crude with linear search is `O(k)` per sample; with a sorted CDF + binary search it is `O(log k)`; with `searchsorted` over a batch of uniforms, it is effectively amortised `O(log k)` per sample.
- Rejection is `O(1)` per *accepted* sample, but the acceptance probability is `1 / (k · max pᵢ) = 1 / (6 · 5/16) ≈ 0.533`. So on average ~1.88 proposals are needed per output, each costing 2 uniforms — hence ≈ 3.75 uniforms per sample.
- Alias is `O(1)` per sample with **exactly** 2 uniforms and a single comparison. This is why, despite the same per-sample loop overhead as rejection, alias is ~4× faster: no retries.

**Random-number budget.** Crude is the most frugal (1 uniform per sample). Alias uses 2 uniforms with no waste. Rejection wastes uniforms — anything below `5/16` of `max p` is sub-optimal here, and the waste grows quickly if the target is more skewed.

**Setup cost.** Crude builds a length-`k` cumulative sum (trivial). Alias builds the `F` and `L` tables in `O(k)` but with non-trivial bookkeeping (the deficit/surplus partition). Rejection has no setup at all.

---

## (d) Recommendations and tradeoffs

| Situation | Recommended method | Reason |
|-----------|-------------------|--------|
| Small `k`, one-off sampling, or `pᵢ` recomputed each call | **Crude** | Easy to write, fast enough, no setup, only 1 uniform per sample. |
| Distribution is fixed and many samples are needed; `k` is moderate-to-large | **Alias** | Constant-time per sample regardless of `k`, with a tight uniform budget. The setup is amortised over many draws. |
| Probabilities are expensive or only known up to a constant; or you only have an upper-bound envelope | **Rejection** | Doesn't need a normalised CDF; just `pᵢ ≤ C qᵢ`. The natural building block for continuous methods and MCMC later. |
| `pᵢ` highly peaked (one class dominates) | **Avoid rejection**, prefer alias or crude | Acceptance rate collapses to `1 / (k · max pᵢ)`. With our 5/16 max this is already only 53%; for sharper distributions it can become catastrophic. |
| You'll later move to continuous distributions or MCMC | Learn **rejection** carefully | It generalises directly; crude and alias do not. |

### Per-method tradeoffs

**Crude (inverse-CDF).**
- **Pros:** Conceptually trivial; one uniform per sample; vectorises beautifully via `searchsorted`; no setup beyond a cumulative sum.
- **Cons:** Linear search is `O(k)` — bad for large `k`. The standard remedies (binary, indexed search) reintroduce setup complexity. Doesn't generalise to continuous distributions whose CDF can't be inverted in closed form.

**Rejection.**
- **Pros:** No setup; works when `pᵢ` is unnormalised or expensive; generalises directly to continuous distributions and is the foundation for MCMC / Hamiltonian samplers.
- **Cons:** Wastes uniforms; performance is dictated by the envelope tightness. For highly peaked discrete distributions it is the worst of the three. Requires a constant `C` that bounds `pᵢ / qᵢ` — finding a tight `C` is the whole art.

**Alias.**
- **Pros:** Truly constant-time sampling regardless of `k`. Exactly 2 uniforms per sample. Optimal for the "fixed distribution, many samples" regime.
- **Cons:** Setup is `O(k)` with bookkeeping that is easy to get wrong. The tables must be rebuilt if `pᵢ` changes. Specific to *discrete* distributions — does not generalise.

### Empirical takeaway from this exercise

For our 6-point distribution with `n = 10,000`, all three methods give statistically valid samples (p > 0.7). The choice is therefore not about correctness but about cost: **crude** wins on simplicity and per-sample uniform budget, **alias** wins on per-sample constant-time guarantees once the tables exist, and **rejection** is the slowest and most wasteful here — but the only one of the three that will still apply when we move beyond inverse-CDF-tractable distributions.
