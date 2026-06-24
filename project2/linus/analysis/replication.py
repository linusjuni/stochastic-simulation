from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats

from project2.linus.models.base import CompartmentalModel, Trajectory

#: a factory that builds a fresh model from its own RNG stream
ModelFactory = Callable[[np.random.Generator], CompartmentalModel]


def replicate(
    make_model: ModelFactory,
    n: int,
    t_max: float,
    rng: np.random.Generator,
) -> list[Trajectory]:
    """Run `n` independent simulations and return their trajectories.

    Parameters
    ----------
    make_model : callable
        ``make_model(stream)`` returns a fresh model seeded with `stream`.
    n : int
        Number of independent replications.
    t_max : float
        Time horizon passed to each ``model.run``.
    rng : np.random.Generator
        Parent generator; `n` independent child streams are spawned from it.
    """
    streams = rng.spawn(n)
    return [make_model(stream).run(t_max) for stream in streams]


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a confidence interval."""

    mean: float
    half_width: float
    lower: float
    upper: float
    n: int

    def __str__(self) -> str:
        return (
            f"{self.mean:.4g} +/- {self.half_width:.2g} "
            f"(CI [{self.lower:.4g}, {self.upper:.4g}], n={self.n})"
        )


def confidence_interval(values: Sequence[float], alpha: float = 0.05) -> Estimate:
    """Mean and (1 - alpha) confidence interval from independent replications.

    Uses the Day-3 replication formula:  mean +/- t_{alpha/2}(n-1) * s / sqrt(n),
    where ``s`` is the sample standard deviation. Works for any per-run quantity:
    a real number (peak size, final size) or a 0/1 indicator (did it die out).
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    mean = float(x.mean())
    if n < 2:
        return Estimate(mean, float("nan"), float("nan"), float("nan"), n)
    s = float(x.std(ddof=1))
    t = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    half_width = t * s / np.sqrt(n)
    return Estimate(mean, half_width, mean - half_width, mean + half_width, n)
