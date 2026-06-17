from __future__ import annotations

"""LCG utilities.

The module provides three simple pieces:
- `LCG`: the integer-only linear congruential generator.
- `generate_ints`: returns raw integer outputs from the generator.
- `generate_uniforms`: returns scaled values in [0, 1).
"""

from dataclasses import dataclass


@dataclass
class LCG:
    # seed is the initial state; a is the multiplier, b is the increment, and m is the modulus.
    seed: int
    a: int
    b: int
    m: int

    def __post_init__(self) -> None:
        if self.m <= 0:
            raise ValueError("m must be positive")
        self.state = self.seed % self.m

    def next_int(self) -> int:
        self.state = (self.a * self.state + self.b) % self.m
        return self.state

    def next_float(self) -> float:
        return self.next_int() / self.m


def generate_ints(seed: int, a: int, b: int, m: int, count: int) -> list[int]:
    generator = LCG(seed, a, b, m)
    return [generator.next_int() for _ in range(count)]


def generate_uniforms(seed: int, a: int, b: int, m: int, count: int) -> list[float]:
    generator = LCG(seed, a, b, m)
    return [generator.next_float() for _ in range(count)]
