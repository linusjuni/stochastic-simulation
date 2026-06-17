from __future__ import annotations

import numpy as np
import scipy.stats as st


def t_ci(samples: np.ndarray, conf: float = 0.95) -> tuple[float, float, float, float]:
    n = len(samples)
    mean = float(samples.mean())
    se = float(samples.std(ddof=1)) / np.sqrt(n)
    hw = st.t.ppf((1.0 + conf) / 2.0, df=n - 1) * se
    return mean, mean - hw, mean + hw, 2.0 * hw
