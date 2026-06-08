from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from utils.logger import get_logger

sns.set_theme(style="whitegrid", palette="muted")

logger = get_logger(__name__)


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
            path = Path(save)
            path.parent.mkdir(parents=True, exist_ok=True)
            defaults: dict[str, Any] = {"dpi": 300, "bbox_inches": "tight"}
            defaults.update(save_kwargs)
            fig.savefig(path, **defaults)
            plt.close(fig)
            logger.success("Saved figure", path=str(path.resolve()))
        else:
            plt.show()
            plt.close(fig)
