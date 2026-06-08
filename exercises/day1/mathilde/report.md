# Day 1 Report

## Parameter choice
- Bad parameters: `a = 1`, `b = 1`, `M = 16`.
- Good parameters: `a = 16807`, `b = 0`, `M = 2147483647`.

## Bad parameters
The bad choice produces a short modular cycle with obvious structure. The histogram is uneven, the scatter plot looks patterned, and the scratch-built chi-square, KS, runs, and correlation checks all move away from what we expect from a uniform generator.  

## Good parameters
The final choice uses a large prime modulus and a classic multiplier. It stays fully integer-based and gives a much better spread in the histogram and scatter plot. The statistical tests are also much closer to a reasonable uniform generator.

## System generator comparison
The script also evaluates Python's built-in `random` generator on 10,000 values and runs the same scratch-built chi-square, KS, runs, and correlation tests on it.

## Sample sufficiency
One sample is not enough to judge a generator. A single run can look acceptable just by chance. The code therefore evaluates multiple seeds and summarizes the results across several starting states. That is better than relying on one sample, even though it still does not prove perfect randomness.

## Output files
Running `day1/main.py` saves these PNGs in the `day1` folder:
- `good_lcg_histogram.png`
- `good_lcg_scatter.png`
- `system_histogram.png`
- `system_scatter.png`
