from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


@contextmanager
def figure(
    *,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
    **save_kwargs: Any,
):
    """Create a styled figure, then show or save it on exit.

    Usage:
        with figure(figsize=(8, 4), save="output.png") as fig:
            ax = fig.add_subplot(111)
            ax.plot(...)
    """
    fig = plt.figure(figsize=figsize)
    try:
        yield fig
    finally:
        if save is not None:
            defaults: dict[str, Any] = {"dpi": 300, "bbox_inches": "tight"}
            defaults.update(save_kwargs)
            fig.savefig(save, **defaults)
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
