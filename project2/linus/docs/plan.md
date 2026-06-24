# Project 2 — Simulation of Epidemics: Plan & Architecture

Course: 02443 Stochastic Simulation
Topic: Stochastic modelling of infectious-disease spread (SIR-type compartmental models)

---

## 1. What the project asks

The project is **open-ended**. We model infectious diseases and their spread using
**SIR-type compartmental models**, but built as a *stochastic* simulation rather than
the classical deterministic ODEs — exactly what SSI's expert group did during COVID
(they simulated scenarios rather than solving differential equations).

We must do **Part I** plus **"at least a handful"** of the Part II tracks.

### Part I — Basic modelling (required)

Build the stochastic SIR (S → I → R), sweep parameters (literature values for real
diseases), and answer:

- **(a)** Probability the disease dies out (stochastic extinction — impossible in the ODE).
- **(b)** Can the disease show cyclical / oscillatory behaviour.
- **(c)** For a highly deadly disease, can the whole population be wiped out.
- **(d)** For which population sizes / parameters is the deterministic ODE precise enough
  (i.e. when does stochasticity stop mattering).

### Part II — Extensions (pick several)

- **(a)** More compartments with different contagion rates → **SEIR** (Exposed/latent),
  asymptomatic states.
- **(b)** **Subgroups** (families, workplaces, public transport) with different exposure.
- **(c)** **Spatial** distribution + movement patterns that depend on state.
- **(d)** **Vaccination** at varying effectiveness.

---

## 2. Engine decision: Gillespie (continuous-time, event-by-event)

**Decision: use the Gillespie algorithm — exact continuous-time stochastic simulation.**
Not a fixed-time-step discrete chain, and not both.

This is strongly supported by our **own course material**, not just the literature.

### Why — the course points directly at it

The relevant lecture is **Day 3 — "Discrete event simulation (The event-by-event
principle)"**. Watch the terminology trap: it is called *discrete-event* simulation, but
slide 4 defines its characteristics as:

> - **Continuous** but asynchronous time
> - Systems described by **discrete state variables**

That is *exactly* a stochastic SIR model: continuous time, integer compartment counts
(S, I, R). "Discrete-event" means **events occur at discrete instants in continuous
time** — the opposite of a fixed-time-step "discrete-time" scheme. So the approach the
course teaches is the continuous-time / Gillespie one.

The Gillespie loop maps **one-to-one** onto the lecture's event-by-event algorithm
(Day 3, slide 6):

| Lecture (Day 3, slide 6) | Our SIR engine |
| --- | --- |
| Real-time clock | `t` |
| State variables | `S, I, R` counts |
| Event list + advance clock to next event | draw next event time as `Exp(total rate)` |
| Invoke event-handling routine | apply infection or recovery → move one individual |
| Statistical accumulators | record trajectory / peak / extinction |
| Generate & schedule future events | recompute rates from new state, loop |

### Supporting pieces are already in our syllabus

- **Exponential inter-event times** = the Poisson process from **Day 3, slides 13–14**
  (`Sᵢ ~ Exp(λ)`). Gillespie's waiting time is just `Exp(total event rate)`.
- **Sampling exponentials / discrete events** = **Day 2** (Sampling from continuous /
  discrete distributions) — inverse-CDF for the exponential waiting time and for picking
  which event fires.
- **Confidence intervals** for "probability the disease dies out" (Part I-a), peak size,
  etc. = **Day 3 subsampling / replications, slides 16–19**. Run `n` independent
  simulations and use the slide-19 formula:
  `θ̄ ± (S_θ / √n) · t_{α/2}(n−1)`.
- **Burn-in** (**Day 3, slide 7**) applies if we study endemic / steady-state behaviour
  (the SIRS cyclical track, Part I-b).
- **Variance reduction** (**Day 4**) is optional polish for the Monte-Carlo estimates
  (common random numbers when comparing parameter settings, etc.).

### Why not the alternatives

- **Fixed-time-step (chain-binomial / tau-leap):** introduces time-discretization error,
  and is described in the literature as the inefficient "brute-force" approach — most
  small steps nothing happens. Not what Day 3 teaches.
- **Support both engines:** over-engineering. The only place a non-stochastic baseline is
  needed is Part I-d, and there the baseline is the **deterministic ODE** via
  `scipy.integrate.solve_ivp`, *not* a second stochastic engine. So: Gillespie for
  stochastic, `solve_ivp` for the deterministic overlay — two clean things, not three.

### External confirmation

Stochastic SIR is standardly a **continuous-time Markov chain simulated with Gillespie**;
the fixed-small-step alternative is explicitly the inefficient brute-force method.

- Stochastic Epidemic Modelling — https://arxiv.org/pdf/2211.00138
- ETH — Stochastic simulation of epidemics (SIR) — https://ethz.ch/content/dam/ethz/special-interest/usys/ibz/theoreticalbiology/education/learningmaterials/701-1424-00L/stochSIR.pdf
- EoN: Epidemics on Networks — https://arxiv.org/pdf/2001.02436

---

## 3. Architecture: a shared base class

The base class **is** the Day-3 event-by-event loop, generalized over a list of
transitions. Subclasses declare *only* their compartments and transition rates; the
engine, RNG seeding, trajectory recording, and statistics are written once and shared.

```
CompartmentalModel (base)          ← owns the Gillespie / event-by-event loop
  • state: dict[str, int]                  (S, I, R, ...)
  • transitions(): list[(rate, src, dst)]  ← the ONE thing subclasses override
  • step():  total = Σ rates;  dt ~ Exp(total);  pick event ∝ rate;  move 1 person
  • run(t_max): loop step() until t_max OR total rate == 0  (→ extinction, Part I-a/c)
  • returns a Trajectory (times[], states[]) for plotting + accumulators

SIR(CompartmentalModel)    → S→I: β·S·I/N ,  I→R: γ·I
SEIR / SIRS                → add E or R→S   (Part I-b cycles, Part II-a)
VaccinationModel           → Part II-d
SubgroupModel / Spatial    → Part II-b/c
```

Supporting components (shared, written once):

- `Trajectory` — container holding `times[]` and per-compartment `states[]`; knows how to
  give peak infected, time-to-peak, final size, extinction flag.
- **Deterministic overlay** — `scipy.integrate.solve_ivp` solver for the SIR ODEs, used
  only for the Part I-d comparison against the stochastic mean.
- **Subsampling / CI helper** — run `n` independent replications, return `θ̄` and the
  Day-3 confidence interval. Used for extinction probability, peak size, etc.

Everyone reuses the existing `utils/`:

- `utils.logger.get_logger` — structured logging (same pattern as project1).
- `utils.plotting.figure` / `histogram` — consistent plots.
- `utils.settings.settings.SEED` — reproducible RNG seeding
  (`np.random.default_rng(settings.SEED)`).

`pyproject.toml` `[tool.hatch.build.targets.wheel].packages` should add `project2` so the
package is importable like `project1`.

---

## 4. How the work divides across the team

Teammates each subclass `CompartmentalModel` and override **only `transitions()`** — the
shared engine means tracks can be built in parallel without touching the core.

| Person | Piece | Depends on |
| --- | --- | --- |
| Owner of base | `CompartmentalModel` + `Trajectory` + `SIR` + CI helper | nothing |
| 2 | Part I analysis (a–d): extinction prob, ODE comparison (Part I-d), param sweeps | base |
| 3 | `SEIR` / `SIRS` (cyclical behaviour, Part I-b + Part II-a) | base |
| 4 | `VaccinationModel` + `SubgroupModel` (Part II-b/d); optional Spatial (II-c) | base |

---

## 5. Lecture → task cheat sheet

| Need | Lecture |
| --- | --- |
| Event-by-event simulation loop (the engine itself) | Day 3 — Discrete event simulation, slides 4–6 |
| Exponential waiting times / Poisson process | Day 3, slides 13–14 |
| Sampling exponentials & discrete events (inverse-CDF) | Day 2 — Sampling from continuous / discrete distributions |
| Confidence intervals via replications / subsampling | Day 3, slides 16–19 |
| Burn-in for steady-state (endemic / SIRS) | Day 3, slide 7 |
| Variance reduction (optional, e.g. common random numbers) | Day 4 — Variance reduction methods |
| Deterministic ODE baseline (Part I-d) | `scipy.integrate.solve_ivp` (not a lecture; standard SIR ODE) |

---

## 6. Open next step

Scaffold `project2/linus/`: the `CompartmentalModel` base, a `Trajectory` container, a working
`SIR` subclass, the `solve_ivp` deterministic overlay, and the subsampling/CI helper —
reusing `utils/logger`, `utils/plotting`, `utils/settings`, and registering `project2` in
`pyproject.toml`.
