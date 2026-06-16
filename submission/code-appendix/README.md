# 02443 Stochastic Simulation — Group 67 Code Appendix

This appendix contains the **complete** exercise code from every group member
(Linus, Mathias, Mathilde, Theo) for all six exercise days. It is provided as a
full record of each author's individual work, in addition to the curated
solutions in the main `code/` submission.

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python package/environment manager)
- Python >= 3.13 (uv will provision this automatically)

## Setup

Run everything from this directory (the `code-appendix/` root). Create the
virtual environment from the locked dependencies:

```bash
uv sync
```

## Running the code

Always run from this `code-appendix/` root so the `exercises` and `utils`
packages resolve correctly. Each author's files live under
`exercises/dayN/<author>/`.

Files written as importable modules are run with `-m`, for example:

```bash
uv run -m exercises.day1.mathilde.main
```

Standalone scripts are run directly, for example:

```bash
uv run python exercises/day5/mathias/metropolis_hastings.py
```

## Notes

- Some modules use absolute imports (`from exercises.dayN.<author>...`) and a
  few have cross-day dependencies, so the directory layout must be kept intact.
- Scripts that produce plots create their output directories automatically.
- Random seeds are fixed in the scripts, so results are reproducible.

## Layout

```
code-appendix/
├── pyproject.toml          # project metadata and dependencies
├── uv.lock                 # locked dependency versions
├── utils/                  # shared logging, plotting, and settings helpers
└── exercises/
    ├── day1/  linus/ · mathias/ · mathilde/ · theo/
    ├── day2/  linus/ · mathias/ · mathilde/ · theo/
    ├── day3/  linus/ · mathias/ · mathilde/ · theo/
    ├── day4/  linus/ · mathias/ · mathilde/ · theo/
    ├── day5/  linus/ · mathias/ · mathilde/ · theo/
    └── day6/  cost.csv · linus/ · mathilde/ · theo/
```
