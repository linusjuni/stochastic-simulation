# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for synthesizing, code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

from __future__ import annotations

import numpy as np

from project2.theo.models.base import Transition
from project2.theo.models.sird import SIRD


class SIRSD(SIRD):
    def __init__(
        self,
        *,
        S: int,
        I: int,
        beta: float,
        gamma: float,
        mu: float,
        omega: float,
        rng: np.random.Generator,
        R: int = 0,
        D: int = 0,
        foi: str = "living",
    ):
        super().__init__(
            S=S, I=I, beta=beta, gamma=gamma, mu=mu, rng=rng, R=R, D=D, foi=foi
        )
        self.omega = omega

    def transitions(self) -> list[Transition]:
        return super().transitions() + [
            Transition(
                "waning",
                "R",
                "S",
                lambda m: m.omega * m.state["R"],
            ),
        ]
