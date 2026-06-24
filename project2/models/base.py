from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class Transition:
    """One possible event: move a single individual from `source` to `target`.

    `source` / `target` may be None to model flows in/out of the population
    (e.g. a death removes from `source` to None; a birth adds from None to
    `target`). `rate` is a function of the current model state returning the
    instantaneous hazard of this transition.
    """

    name: str
    source: str | None
    target: str | None
    rate: Callable[["CompartmentalModel"], float]


class Trajectory:
    """Time-stamped record of a single simulation run.

    Stores the event times and the compartment counts immediately *after* each
    event. The state is piecewise-constant between recorded times, so plot it
    with ``drawstyle="steps-post"``.
    """

    def __init__(self, compartments: tuple[str, ...]):
        self.compartments = tuple(compartments)
        self.times: list[float] = []
        self._counts: dict[str, list[int]] = {c: [] for c in self.compartments}

    def record(self, t: float, state: Mapping[str, int]) -> None:
        self.times.append(t)
        for c in self.compartments:
            self._counts[c].append(state[c])

    @property
    def t(self) -> np.ndarray:
        return np.asarray(self.times)

    def __getitem__(self, compartment: str) -> np.ndarray:
        return np.asarray(self._counts[compartment])

    # Convenience metrics for the Part I questions
    def peak(self, compartment: str) -> int:
        """Largest count ever reached in a compartment (e.g. peak infected)."""
        return int(self[compartment].max())

    def time_of_peak(self, compartment: str) -> float:
        return float(self.t[int(self[compartment].argmax())])

    def final(self, compartment: str) -> int:
        return int(self._counts[compartment][-1])

    def extinct(self, infected: str = "I") -> bool:
        """True if no infectious individuals remain (the outbreak died out)."""
        return self._counts[infected][-1] == 0


class CompartmentalModel(ABC):
    """Base class owning the Gillespie loop. Subclasses declare the model."""

    #: ordered compartment names; subclasses must set this
    compartments: tuple[str, ...] = ()

    def __init__(self, initial_state: Mapping[str, int], rng: np.random.Generator):
        missing = set(self.compartments) - set(initial_state)
        if missing:
            raise ValueError(f"initial_state missing compartments: {sorted(missing)}")
        self.state: dict[str, int] = {
            c: int(initial_state[c]) for c in self.compartments
        }
        self.rng = rng
        self.t: float = 0.0

    @abstractmethod
    def transitions(self) -> list[Transition]:
        """Return the transitions available from the current state."""

    @property
    def population(self) -> int:
        return sum(self.state.values())

    def _choose(self, rates: np.ndarray, total: float) -> int:
        """Pick a transition index with probability proportional to its rate."""
        u = self.rng.random() * total
        return int(np.searchsorted(np.cumsum(rates), u))

    def step(self) -> bool:
        """Advance the simulation by one event. Returns False if none can fire."""
        trans = self.transitions()
        rates = np.array([tr.rate(self) for tr in trans], dtype=float)
        total = rates.sum()
        if total <= 0.0:
            return False  # no events possible -> absorbing state (e.g. extinction)

        dt = self.rng.exponential(1.0 / total)
        tr = trans[self._choose(rates, total)]
        if tr.source is not None:
            self.state[tr.source] -= 1
        if tr.target is not None:
            self.state[tr.target] += 1
        self.t += dt
        return True

    def run(self, t_max: float, max_events: int = 10_000_000) -> Trajectory:
        """Simulate until `t_max`, the system reaches an absorbing state, or
        `max_events` events have fired. Returns the full Trajectory."""
        traj = Trajectory(self.compartments)
        traj.record(self.t, self.state)
        events = 0
        while self.t < t_max and events < max_events:
            if not self.step():
                break
            traj.record(self.t, self.state)
            events += 1
        return traj
