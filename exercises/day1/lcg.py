from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plotting import figure


class LCG:
    def __init__(self, seed: int, multiplier: int, increment: int, modulus: int) -> None:
        if modulus <= 0:
            raise ValueError("modulus must be positive")
        self.state = seed % modulus
        self.multiplier = multiplier
        self.increment = increment
        self.modulus = modulus

    def next_int(self) -> int:
        self.state = (self.multiplier * self.state + self.increment) % self.modulus
        return self.state


def sequence(seed: int, multiplier: int, increment: int, modulus: int, count: int) -> list[int]:
    generator = LCG(seed, multiplier, increment, modulus)
    return [generator.next_int() for _ in range(count)]


def histogram_counts(values: list[int], classes: int, minimum: int, maximum: int) -> list[int]:
    if classes <= 0:
        raise ValueError("classes must be positive")
    if maximum < minimum:
        raise ValueError("maximum must be at least minimum")

    counts = [0] * classes
    span = maximum - minimum + 1

    for value in values:
        index = ((value - minimum) * classes) // span
        if index >= classes:
            index = classes - 1
        counts[index] += 1

    return counts


def histogram_text(values: list[int], classes: int, minimum: int, maximum: int, width: int = 40) -> str:
    counts = histogram_counts(values, classes, minimum, maximum)
    peak = max(counts) if counts else 0
    if peak == 0:
        peak = 1

    lines = []
    for index, count in enumerate(counts, start=1):
        bar_length = (count * width) // peak
        lines.append(f"{index:2d} | {'#' * bar_length} {count}")

    return "\n".join(lines)


def plot_histogram(values: list[int], classes: int, minimum: int, maximum: int) -> None:
    counts = histogram_counts(values, classes, minimum, maximum)
    bin_width = (maximum - minimum + 1) / classes
    left_edges = [minimum + index * bin_width for index in range(classes)]

    with figure(figsize=(8, 4)) as fig:
        ax = fig.add_subplot(111)
        ax.bar(left_edges, counts, width=bin_width, align="edge", edgecolor="black")
        ax.set_xlabel("Value range")
        ax.set_ylabel("Count")
        ax.set_title("LCG histogram")


def main() -> None:
    values = sequence(7, 5, 1, 16, 10000)
    plot_histogram(values, 10, 0, 15)


if __name__ == "__main__":
    main()
