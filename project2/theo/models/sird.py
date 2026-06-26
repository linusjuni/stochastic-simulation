# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for synthesizing, code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

from __future__ import annotations

import numpy as np

from project2.theo.models.base import CompartmentalModel, Transition


class SIRD(CompartmentalModel):
    compartments = ("S", "I", "R", "D")

    def __init__(
        self,
        *,
        S: int,
        I: int,
        beta: float,
        gamma: float,
        mu: float,
        rng: np.random.Generator,
        R: int = 0,
        D: int = 0,
        foi: str = "living",
    ):
        super().__init__({"S": S, "I": I, "R": R, "D": D}, rng)
        if foi not in ("living", "fixed"):
            raise ValueError(f"foi must be 'living' or 'fixed', got {foi!r}")
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.foi = foi
        self.N0 = S + I + R

    @property
    def living(self) -> int:
        """Alive individuals (S + I + R)."""
        return self.state["S"] + self.state["I"] + self.state["R"]

    @property
    def infection_denominator(self) -> int:
        """Force-of-infection denominator: the shrinking living population
        (``foi="living"``) or the fixed initial population (``foi="fixed"``)."""
        return self.living if self.foi == "living" else self.N0

    @property
    def R0(self) -> float:
        """Basic reproduction number; removal is recovery *and* death."""
        return self.beta / (self.gamma + self.mu)

    @property
    def cfr(self) -> float:
        """Case fatality ratio: fraction of infected individuals who die."""
        return self.mu / (self.gamma + self.mu)

    def transitions(self) -> list[Transition]:
        return [
            Transition(
                "infection",
                "S",
                "I",
                # guard against a zero denominator (everyone dead): then S == 0
                # too, so the true rate is 0 and we only avoid a 0/0 division.
                lambda m: (
                    m.beta * m.state["S"] * m.state["I"] / m.infection_denominator
                    if m.infection_denominator > 0
                    else 0.0
                ),
            ),
            Transition(
                "recovery",
                "I",
                "R",
                lambda m: m.gamma * m.state["I"],
            ),
            Transition(
                "death",
                "I",
                "D",
                lambda m: m.mu * m.state["I"],
            ),
        ]
