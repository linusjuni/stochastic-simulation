from __future__ import annotations

import numpy as np

from project2.linus.models.base import CompartmentalModel, Transition


class SIR(CompartmentalModel):
    compartments = ("S", "I", "R")

    def __init__(
        self,
        *,
        S: int,
        I: int,
        beta: float,
        gamma: float,
        rng: np.random.Generator,
        R: int = 0,
    ):
        super().__init__({"S": S, "I": I, "R": R}, rng)
        self.beta = beta
        self.gamma = gamma

    @property
    def R0(self) -> float:
        """Basic reproduction number."""
        return self.beta / self.gamma

    def transitions(self) -> list[Transition]:
        N = self.population
        return [
            Transition(
                "infection",
                "S",
                "I",
                lambda m: m.beta * m.state["S"] * m.state["I"] / N,
            ),
            Transition(
                "recovery",
                "I",
                "R",
                lambda m: m.gamma * m.state["I"],
            ),
        ]
