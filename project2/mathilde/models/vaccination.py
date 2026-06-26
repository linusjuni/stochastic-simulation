from __future__ import annotations

import numpy as np

from project2.linus.models.base import CompartmentalModel, Transition


class VaccinatedSIR(CompartmentalModel):
    """Leaky-vaccine SIR: a fraction of the population is vaccinated before
    t=0 and faces a reduced (not zero) infection rate thereafter.

    Coverage is fixed once, at construction, as the S/V split -- it is *not*
    an ongoing transition. This keeps the event-count profile identical to
    plain SIR (each individual transitions at most twice: S or V -> I -> R),
    so a run terminates the moment I=0, exactly like SIR.
    """

    compartments = ("S", "V", "I", "R")

    def __init__(
        self,
        *,
        S: int,
        V: int,
        I: int,
        beta: float,
        gamma: float,
        ve: float,
        rng: np.random.Generator,
        R: int = 0,
    ):
        super().__init__({"S": S, "V": V, "I": I, "R": R}, rng)
        self.beta = beta
        self.gamma = gamma
        self.ve = ve
        # vaccinated share of the susceptible-or-vaccinated pool at t=0,
        # frozen here since S/V drift once the epidemic starts
        self._p0 = V / (S + V) if (S + V) > 0 else 0.0

    @property
    def R0(self) -> float:
        """Basic reproduction number without vaccination."""
        return self.beta / self.gamma

    @property
    def Rv(self) -> float:
        """Effective reproduction number at t=0 given coverage and efficacy."""
        return self.R0 * (1.0 - self._p0 * self.ve)

    def transitions(self) -> list[Transition]:
        N = self.population
        return [
            Transition(
                "infection_S",
                "S",
                "I",
                lambda m: m.beta * m.state["S"] * m.state["I"] / N,
            ),
            Transition(
                "infection_V",
                "V",
                "I",
                lambda m: (1.0 - m.ve) * m.beta * m.state["V"] * m.state["I"] / N,
            ),
            Transition(
                "recovery",
                "I",
                "R",
                lambda m: m.gamma * m.state["I"],
            ),
        ]
