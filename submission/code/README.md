# 02443 Stochastic Simulation — Group 67 Code

Source code for the exercise hand-ins (days 1–6).

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python package/environment manager)
- Python >= 3.13 (uv will provision this automatically)

## Setup

Run everything from this directory (the `code/` root). First create the
virtual environment from the locked dependencies:

```bash
uv sync
```

## Running the exercises

Each command below runs one exercise. Always run them from this `code/` root
so that the `exercises` and `utils` packages resolve correctly.

```bash
# Day 1 — Random number generation and statistical tests
uv run -m exercises.day1.exercise1

# Day 2 — Sampling from discrete distributions
uv run -m exercises.day2.exercise2.main
# Day 2 — Sampling from continuous distributions
uv run -m exercises.day2.exercise3.main

# Day 3 — Discrete-event simulation / blocking system
uv run -m exercises.day3.main

# Day 4 — Variance reduction methods
uv run python exercises/day4/exercise5.py

# Day 5 — MCMC: Metropolis-Hastings, Gibbs, Bayesian modelling
uv run python exercises/day5/metropolis_hastings.py
uv run python exercises/day5/bayesian_modelling.py
uv run python exercises/day5/goodness_of_fit.py

# Day 6 — Simulated annealing / TSP
uv run -m exercises.day6.main
# Day 6 — Bootstrap
uv run -m exercises.day6.exercise8.main
```

## Notes

- Days 1, 2, 3, and 6 are run as modules (`-m`); days 4 and 5 are standalone
  scripts (`python <path>`).
- Scripts that produce plots save them under `report/plots/dayN/`, which is
  created automatically on first run.
- The random seeds are fixed in each script, so results are reproducible.

## Layout

```
code/
├── pyproject.toml          # project metadata and dependencies
├── uv.lock                 # locked dependency versions
├── utils/                  # shared logging, plotting, and settings helpers
└── exercises/
    ├── day1/               # exercise1.py
    ├── day2/               # exercise2/, exercise3/
    ├── day3/               # blocking_system.py, main.py
    ├── day4/               # exercise5.py
    ├── day5/               # metropolis_hastings.py, bayesian_modelling.py, goodness_of_fit.py
    └── day6/               # cost.csv, tsp.py, main.py, exercise8/
```
