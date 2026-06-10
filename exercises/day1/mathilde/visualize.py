from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from exercises.day1.mathilde.stats_tests import bin_histogram


def save_histogram(values: list[float], path: Path, bins: int = 10, low: float = 0.0, high: float = 1.0, title: str = "Histogram") -> None:
    counts, edges = bin_histogram(values, bins, low, high)
    plt.figure(figsize=(8, 4))
    plt.bar(edges[:-1], counts, width=(high - low) / bins, align="edge", edgecolor="black")
    plt.xlabel("Value range")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_scatter(values: list[float], path: Path, lag: int = 1, title: str = "Scatter plot") -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(values[:-lag], values[lag:], s=6)
    plt.xlabel("u_i")
    plt.ylabel(f"u_i+{lag}")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
