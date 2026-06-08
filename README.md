# 02443 Stochastic Simulation – Summer 2026

3-week course (8 June – 26 June).

## Structure

```plaintext
lectures/
  day1/ – day6/      # Slide PDFs for Part 1 lectures (Jun 8–15)

exercises/
  day1/ – day6/      # Exercise solutions for Part 1 (due report Jun 16)

project1/            # Markov breast cancer model (Jun 16–19, due Jun 19)

project2/            # Free-choice project (Jun 22–26)

utils/               # Shared helper functions
```

## Parts

| Part                                        | Dates     | Deliverable        |
| ------------------------------------------- | --------- | ------------------ |
| 1 – Lectures & exercises                    | Jun 8–15  | Report due Jun 16  |
| 2 – Project 1 (breast cancer Markov model)  | Jun 16–19 | Report due Jun 19  |
| 3 – Project 2 (free choice)                 | Jun 22–26 | Report due Jun 26  |

## Setup

```bash
uv sync
```

## Utils

### `utils/settings.py`

Single setting: `SEED` (default `69`). Override via environment variable.

```python
from utils.settings import settings

rng = np.random.default_rng(settings.SEED)
```

### `utils/logger.py`

Coloured stdout logger. Pass keyword arguments for structured context fields.

```python
from utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Starting", n=1000, seed=42)
logger.warning("Something off")
logger.error("Something failed")
logger.success("Done")
```

### `utils/plotting.py`

Applies seaborn whitegrid + muted palette globally on import. Provides a `figure` context manager that shows or saves the figure on exit.

```python
from utils.plotting import figure

# Show interactively
with figure(figsize=(8, 4)) as fig:
    ax = fig.add_subplot(111)
    ax.hist(samples, bins=10)

# Save to file (logs the absolute path on save, creates directories if needed)
with figure(figsize=(8, 4), save="output.png") as fig:
    ax = fig.add_subplot(111)
    ax.hist(samples, bins=10)
```
