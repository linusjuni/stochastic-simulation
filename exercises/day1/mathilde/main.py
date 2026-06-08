
from __future__ import annotations

from pathlib import Path
import random

from exercises.day1.mathilde.lcg import generate_uniforms
from exercises.day1.mathilde.stats_tests import analyze, run_smoke_tests
from exercises.day1.mathilde.visualize import save_histogram, save_scatter


def print_stats(label: str, stats: dict[str, dict[str, float]]) -> None:
    print(label)
    print(f"  chi-square: {stats['chi_square']['statistic']:.4f}")
    print(f"  ks: {stats['ks']['statistic']:.4f}")
    print(f"  runs z: {stats['runs']['z']:.4f}")
    print(f"  corr h=1: {stats['correlation_h1']['correlation']:.4f}")
    print(f"  corr h=2: {stats['correlation_h2']['correlation']:.4f}")
    print(f"  corr h=5: {stats['correlation_h5']['correlation']:.4f}")


def exercise_a_generate_lcg_values(sample_size: int) -> list[float]:
    good_params = {"seed": 7, "a": 16807, "b": 0, "m": 2_147_483_647}
    values = generate_uniforms(**good_params, count=sample_size)
    print("Exercise a: LCG histogram")
    return values


def exercise_b_evaluate_lcg(values: list[float]) -> dict[str, dict[str, float]]:
    stats = analyze(values)
    print("Exercise b: LCG quality")
    print_stats("  LCG", stats)
    return stats


def exercise_c_compare_bad_and_good(sample_size: int, output_dir: Path) -> None:
    bad_params = {"seed": 7, "a": 1, "b": 1, "m": 16}
    good_params = {"seed": 7, "a": 16807, "b": 0, "m": 2_147_483_647}

    bad_values = generate_uniforms(**bad_params, count=sample_size)
    good_values = generate_uniforms(**good_params, count=sample_size)

    print("Exercise c: bad vs good parameters")
    print_stats("  Bad LCG", analyze(bad_values))
    print_stats("  Good LCG", analyze(good_values))
    # Save bad LCG histogram for part (a) comparison and debugging
    save_histogram(bad_values, output_dir / "bad_lcg_histogram.png", title="Bad LCG histogram")


def exercise_d_compare_system_generator(sample_size: int, seeds: list[int]) -> None:
    system_values = [random.random() for _ in range(sample_size)]

    print("Exercise d: system generator comparison")
    print_stats("  System random", analyze(system_values))
    print(f"  Average over multiple seeds: {repeated_analysis(system_uniforms, seeds, sample_size)}")


def repeated_analysis(generator_factory, seeds: list[int], count: int) -> dict[str, float]:
    results = [analyze(generator_factory(seed, count)) for seed in seeds]
    keys = ["chi_square", "ks", "runs", "correlation_h1", "correlation_h2", "correlation_h5"]
    summary: dict[str, list[float]] = {key: [] for key in keys}

    for result in results:
        summary["chi_square"].append(result["chi_square"]["statistic"])
        summary["ks"].append(result["ks"]["statistic"])
        summary["runs"].append(result["runs"]["z"])
        summary["correlation_h1"].append(abs(result["correlation_h1"]["correlation"]))
        summary["correlation_h2"].append(abs(result["correlation_h2"]["correlation"]))
        summary["correlation_h5"].append(abs(result["correlation_h5"]["correlation"]))

    return {name: sum(values) / len(values) for name, values in summary.items()}


def system_uniforms(seed: int, count: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(count)]


def main() -> None:
    run_smoke_tests()

    output_dir = Path(__file__).resolve().parent
    sample_size = 10_000
    seeds = [1, 7, 11, 19, 37]

    lcg_values = exercise_a_generate_lcg_values(sample_size)
    exercise_b_evaluate_lcg(lcg_values)
    exercise_c_compare_bad_and_good(sample_size, output_dir)
    exercise_d_compare_system_generator(sample_size, seeds)

    save_histogram(lcg_values, output_dir / "good_lcg_histogram.png", title="Good LCG histogram")
    save_scatter(lcg_values, output_dir / "good_lcg_scatter.png", title="Good LCG scatter plot")

    system_values = [random.random() for _ in range(sample_size)]
    save_histogram(system_values, output_dir / "system_histogram.png", title="System random histogram")
    save_scatter(system_values, output_dir / "system_scatter.png", title="System random scatter plot")


if __name__ == "__main__":
    main()