# How to run replications and get a confidence interval

> **How-to guide.** Task-oriented. Assumes you already know what
> an SIR model and the base engine are. For the *why*, see `plan.md`.

You want to run many independent simulations and report a number with error bars
(e.g. peak infected, final size, probability of extinction). Use the two shared
helpers in `project2/linus/analysis`.

## Steps

### 1. Write a factory that builds one fresh model

A model is single-use, so you give `replicate` a function that builds a new one
from an RNG stream:

```python
import numpy as np
from project2.linus.models import SIR

N, I0 = 1000, 1

def make_model(stream: np.random.Generator) -> SIR:
    return SIR(S=N - I0, I=I0, beta=0.3, gamma=0.1, rng=stream)
```

### 2. Run the replications

```python
from project2.linus.analysis import replicate
from utils.settings import settings

rng = np.random.default_rng(settings.SEED)
trajectories = replicate(make_model, n=2000, t_max=200.0, rng=rng)
```

You get back a list of `Trajectory` objects — one per run. Each run already has
its own independent RNG stream (spawned for you), so the runs are reproducible
*and* independent.

### 3. Pull one number out of each run

Use the `Trajectory` metrics. Any per-run quantity works:

```python
peaks       = [t.peak("I")          for t in trajectories]   # peak infected
final_sizes = [t.final("R")         for t in trajectories]   # total ever infected
minor       = [t.final("R") < 10    for t in trajectories]   # 0/1 indicator
```

### 4. Turn them into an estimate with a CI

```python
from project2.linus.analysis import confidence_interval

est = confidence_interval(peaks)     # also works on the 0/1 list for a probability
print(est)        # -> 351.2 +/- 4.1 (CI [347.1, 355.3], n=2000)

est.mean          # point estimate
est.lower, est.upper
```

`confidence_interval` uses the Day-3 replication formula
`mean +/- t_{alpha/2}(n-1) * s / sqrt(n)`. Pass `alpha=0.01` for a 99% CI.

## Full example

A complete, runnable use of all of the above is
`project2/linus/part1/extinction.py`:

```bash
uv run python -m project2.linus.part1.extinction
```

## See also

- `project2/linus/analysis/replication.py` — the source (reference).
- `project2/linus/docs/plan.md` — why we simulate this way (explanation).
