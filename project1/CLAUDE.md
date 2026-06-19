# Project 1 — Breast-cancer Markov model

Guide for solving Project 1 (`docs/project_description.md`) at A-student quality.
Read this before writing any code.

## What "A-student" means here

The grade is in the *analysis*, not the simulator. For every simulated quantity:

1. **Simulate** it with a small, readable simulator.
2. **Validate** it against an analytical result the project gives you (`p_0 P^t`,
   phase-type pmf/mean, matrix-exponential CDF). A simulation is only trusted once
   it matches theory.
3. **Quantify uncertainty** — report confidence intervals, not bare point estimates.
4. **Test** the claim with the *appropriate* statistical test (χ², KS, log-rank),
   stating H₀, the statistic, the p-value, and the conclusion in plain language.

Never report a number without a CI or a comparison-to-theory next to it.

## Code philosophy: one self-contained file per task

This is coursework that gets **handed in and read top-to-bottom**, not a library. Optimise
for *each task being understandable and runnable on its own*, not for DRY.

- **One file per task**, named after the task: `task7.py`, `task8.py`, …
- **Grouped by part in a subfolder**: `project1/part1/`, `project1/part2/`, `project1/part3/`.
- **Self-contained.** Each task file defines the things it needs *inline* — the transition
  matrix, the state constants, the simulator, the statistical test. **Duplication across
  task files is fine and expected.** Do not build a shared `models.py` / `markov.py` /
  `stats.py` that every task imports; that coupling is exactly what we're avoiding. If
  task 8 needs the same simulator as task 7, copy it in. A reader (and a grader) should be
  able to open one file and see the whole story.
- **The only shared dependency is `utils/`** — generic infrastructure, not project logic.
- Each file is runnable on its own and ends in an `if __name__ == "__main__":` block.
- Run each task **as a module from the repo root**, e.g. `uv run python -m project1.part2.task7`.
  Plain `python project1/part2/task7.py` fails to import `utils` — only `-m` (run from the
  repo root) puts the repo root on `sys.path`. Figure save paths are relative to that root.

`project1/task1.py` is the reference for this style — match it.

```
project1/
  part1/  task1.py … task5.py     (task6 is report-only)
  part2/  task7.py … task10.py    (task11 is report-only)
  part3/  task12.py  task13.py
  plots/  task1/ … task13/        (figures, one folder per task)
  report.md                       (the report-only / discussion tasks)
```

## Use the shared `utils/` (the one allowed dependency)

These give consistency for free — use them in every task file:

- **Reproducible, identical seed across all tasks.** Always:
  ```python
  from utils.settings import settings
  rng = np.random.default_rng(settings.SEED)   # built once, in the __main__ block
  ```
  Thread that `rng` through every function — never call global `np.random.*`. Because every
  task seeds from the same `settings.SEED`, runs are reproducible and comparable.
- **Plotting:** `from utils.plotting import figure, histogram`. Save with the `figure`
  context manager to `project1/plots/taskN/<name>.png`:
  ```python
  with figure(figsize=(8, 5), save="project1/plots/task7/lifetimes.png") as fig:
      ax = fig.add_subplot(111)
      ...
  ```
- **Logging:** `from utils.logger import get_logger`; `logger = get_logger(__name__)`.
  Report results with structured fields: `logger.success("...", mean=..., ci=...)`.

## Lecture map — where the method for each task lives

Use the slides under `../lectures/dayX/` as the source of truth for *method*. Mapping:

| Need | Slides |
| --- | --- |
| Sampling a discrete transition from a row of `P` (inverse/cdf method) | `day2/Sampling from discrete distributions.pdf` |
| Exponential sojourn times for the CTMC | `day2/Sampling from continuous distributions.pdf` |
| χ² and KS goodness-of-fit tests (Tasks 2, 3, 8) | `day1/Testing Random Number Generators.pdf` |
| Event-driven continuous-time simulation (Part 2 & 3) | `day3/Discrete event simulation.pdf`, `day3/Ferry example.pdf` |
| Control variates + CIs (Task 5) | `day4/Variance reduction methods.pdf` |
| Markov-chain theory, `p_0 P^t`, stationary behaviour | `day5/Markov chains.pdf` |
| MCEM / accept-reject bridge sampling (Task 13) | `day5/MCMC.pdf` |
| Bootstrap CIs for mean & std (Task 7) | `day6/Bootstrapping.pdf` |

If a method is unclear, open the PDF for that day rather than guessing.

## Per-task hints (the parts that are easy to get wrong)

States (0-based internally, 1–5 in the report): 0=post-surgery, 1=local recurrence,
2=distant metastasis, 3=both, 4=death (absorbing — stop the simulation here).

- **Task 1** — lifetime = number of steps to absorption. Report the *proportion that ever
  visit a local-recurrence state* (state 1, or state 3 reached via "both").
- **Task 2** — compare the empirical state distribution at `t=120` to `p_0 P^120`
  with a χ² goodness-of-fit test; pool bins with expected count < 5 before testing.
- **Task 3** — `P_s` is `P` with last row/col removed; `p_s` is the death column for
  the 4 transient states. Validate empirical lifetimes against the phase-type pmf
  (χ² or KS). Check the empirical mean against `π(I−P_s)⁻¹·1`.
- **Task 4** — rejection sampling: keep simulating until 1000 women satisfy
  *(alive at month 12) AND (visited state 1 or 2 within 12 months)*; report the
  conditional mean lifetime with a CI. Track and report the acceptance rate.
- **Task 5** — control variate: `X` = indicator(death within 350 months) per batch of
  200; `Z` = mean lifetime of that batch (known/estimable expectation). Estimate
  `c* = −Cov(X,Z)/Var(Z)`, form `X + c*(Z − E[Z])`, and report the **variance ratio**
  crude-vs-controlled across the 100 batches.
- **Task 7** — event-driven CTMC: in state `i`, draw sojourn `~Exp(rate=−q_ii)`, then
  jump to `j≠i` with prob `q_ij/(−q_ii)`. Report mean, std, **each with a CI**
  (bootstrap or analytic for the mean). Build `Q` so the diagonal = −(row off-diagonal sum).
- **Task 8** — KS test of empirical lifetimes against `F_T(t)=1−p_0 exp(Q_s t)·1`
  (`scipy.linalg.expm`).
- **Task 9–10** — Kaplan-Meier `Ŝ(t)=(N−d(t))/N` for both Q matrices on one axis;
  the `*` diagonal entries of `Q_treatment` are the negative row-sum (eq. 1).
  Log-rank test for the significance question.
- **Task 12** — observe each woman every 48 months: `Y = (X(0), X(48), …, 5)`; the
  series must end in death (5).
- **Task 13** — MCEM. The hard step is the bridge: between consecutive observations
  `y_k → y_{k+1}`, re-simulate the CTMC over 48 months and **reject** unless it lands
  in `y_{k+1}`. Accumulate `N_ij` (jump counts) and `S_i` (total sojourn) across all
  women, set `q_ij = N_ij/S_i`, fill the diagonal via eq. (1), iterate until
  `‖Q^(k)−Q^(k+1)‖_∞ < 1e-3`. Sanity-check the recovered `Q` against the true `Q`.

## Writing the report (`.tex` parts)

Each task gets a standalone LaTeX snippet at the project root (`project1/task7.tex`, …),
modelled on `project1/task1.tex`. Structure: `\subsection*{Task N}`, plain-language prose,
and one `figure` float (`\includegraphics[...]{sections//partN//media/<png>}` + `\caption`
+ `\label`/`\autoref`). State numbers in rounded, human terms ("about 256 months", "9\%").

**Answer only the questions the project description poses for that task — nothing more.**
This is the single most important rule for the write-up:

- Re-read the task in `docs/project_description.md` and list the *exact* questions it asks.
  Write up those, in that order, and stop. For Task 7 that is precisely: the lifetime
  histogram, mean + CI, std + CI, and the proportion with distant recurrence by 30.5 months.
- **Answer each question by restating it as the opening of the sentence**, then giving the
  result — don't bury the answer behind "we also looked at…". Reframe the question into a
  declarative answer. E.g. "The proportion of women whose cancer had reappeared distantly
  within the first 30.5 months was found to be about 9\%." (not "We also looked at how
  quickly the cancer spreads: in about 9\% of women…"). This makes it obvious which question
  each sentence answers.
- **Do not import analysis the task didn't ask for** — no validation against theory, no
  derivations, no comparisons borrowed from another task or from the lecture slides, no
  "this confirms the simulation is correct". Validation belongs to the task that explicitly
  asks for it (e.g. the phase-type/CDF comparison is *Task 8's* job, not Task 7's).
- Sanity-checks we run *in code* (e.g. comparing a simulated mean to an analytic one) are
  for our own confidence; they stay out of the report unless the task requests them.
- A paragraph or two, matching `task1.tex`'s length and reading level.

**Put numeric results in a `booktabs` table; keep prose for interpretation.** Estimates,
confidence intervals, test statistics, and p-values read far better in a table than buried
in a sentence. Use `\toprule`/`\midrule`/`\bottomrule` (requires `\usepackage{booktabs}`),
a `\caption`, a `\label` you `\autoref`, `\renewcommand{\arraystretch}{1.1}`, and right-align
numeric columns (`l...rr`). Template:

```latex
\begin{table}[h]
    \centering
    \caption{Lifetime after surgery for 1000 simulated women, with 95\% CIs.}
    \label{tab:task7}
    \renewcommand{\arraystretch}{1.1}
    \begin{tabular}{lrr}
        \toprule
        Quantity                 & Estimate & 95\% CI \\
        \midrule
        Mean lifetime (months)   & $256$ & $[245,\ 267]$ \\
        Std.\ deviation (months) & $176$ & $[167,\ 185]$ \\
        \bottomrule
    \end{tabular}
\end{table}
```

**Prose first, then the float.** Introduce and interpret a result in the text *before* the
table or figure it refers to, and reference the float with `\autoref{tab:...}` /
`\autoref{fig:...}` (e.g. "The mean lifetime and its spread are summarised in
\autoref{tab:task7}."). The reader meets the explanation first and the float second — never
drop a table or figure in cold. Don't restate every cell in the prose.

## Report-only tasks

Tasks 6 and 11 are discussion (modelling assumptions, Erlang sojourns). No code —
write them up in `project1/report.md`; don't implement anything.
