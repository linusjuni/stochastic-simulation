from __future__ import annotations

import numpy as np

from project2.linus.models.base import CompartmentalModel, Transition


class SIRS(CompartmentalModel):
    """SIR with waning immunity: recovered individuals return to susceptible.

    The extra R -> S flow (rate omega) replenishes the susceptible pool, which a
    closed SIR cannot do. This gives an endemic equilibrium and -- in the
    stochastic simulation -- sustained cyclical behaviour.
    """

    compartments = ("S", "I", "R")

    def __init__(
        self,
        *,
        S: int,
        I: int,
        beta: float,
        gamma: float,
        omega: float,
        rng: np.random.Generator,
        R: int = 0,
    ):
        super().__init__({"S": S, "I": I, "R": R}, rng)
        self.beta = beta
        self.gamma = gamma
        self.omega = omega

    @property
    def R0(self) -> float:
        """Basic reproduction number."""
        return self.beta / self.gamma

    @property
    def endemic_I(self) -> float:
        """Analytic endemic equilibrium of the deterministic SIRS model.

        I* = N (1 - 1/R0) * omega / (omega + gamma)  for R0 > 1.
        """
        N = self.population
        if self.R0 <= 1.0:
            return 0.0
        return N * (1.0 - 1.0 / self.R0) * self.omega / (self.omega + self.gamma)

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
            Transition(
                "waning",
                "R",
                "S",
                lambda m: m.omega * m.state["R"],
            ),
        ]
