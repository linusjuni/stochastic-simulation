# Author: Theodor la Cour, s225093
# Generative AI tools from Anthropic were used in the creation of this file.
# They have been used for synthesizing, code structering, coding, and verification.
# The author takes full responsibility for all content and decisions in this file.

from __future__ import annotations

import numpy as np

from project2.theo.models.base import Transition
from project2.theo.models.sird import SIRD


class SIRSD(SIRD):
    """SIRD with *waning immunity* (Part I(c), experiment 3).

    Adds a single transition to :class:`SIRD`: recovered individuals lose their
    immunity and return to the susceptible pool, ``R -> S`` at rate ``omega * R``.

    With ``omega > 0`` there is no permanent immune refuge, so the disease can
    keep recirculating (recurrent waves) instead of burning out. Because death
    (``D``) is the only permanent sink, a sustained outbreak grinds the living
    population down -- opening a route to total population extinction *even when
    recovery is possible* (CFR < 1), unlike plain SIRD. ``omega = 0`` recovers
    the ordinary SIRD.

    Mean immunity duration is ``1 / omega``.
    """

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
