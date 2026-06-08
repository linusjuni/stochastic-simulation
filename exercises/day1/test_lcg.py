import unittest

from exercises.day1.lcg import LCG, histogram_counts, sequence


class TestLCG(unittest.TestCase):
    def test_sequence(self) -> None:
        self.assertEqual(sequence(7, 5, 1, 16, 5), [4, 5, 10, 3, 0])

    def test_integer_states(self) -> None:
        generator = LCG(7, 5, 1, 16)
        values = [generator.next_int() for _ in range(4)]
        self.assertTrue(all(isinstance(value, int) for value in values))
        self.assertEqual(values, [4, 5, 10, 3])

    def test_histogram_counts(self) -> None:
        self.assertEqual(histogram_counts(list(range(10)), 5, 0, 9), [2, 2, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()