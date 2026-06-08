"""
plotting.py — reusable, pre-themed plotting helpers.

Usage from any notebook/file in the same folder:

    from plotting import histogram, set_theme

    set_theme()                      # optional: call once at top of a notebook
    histogram(arr, bins=16, title="LCG output")

If plotting.py lives elsewhere, add its folder to the path once:

    import sys; sys.path.append("/path/to/folder")
    from plotting import histogram
"""

from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------------------------------------------------
# Global theme. Call set_theme() once per session, or rely on histogram()
# applying it automatically the first time it runs.
# ----------------------------------------------------------------------
_THEME_APPLIED = False


def set_theme(
    context: Literal["paper", "notebook", "talk", "poster"] = "notebook",
    palette: str = "deep",
    font_scale: float = 1.1,
):
    """Apply a consistent seaborn theme across all plots.

    Parameters
    ----------
    context : str
        seaborn context controlling element sizes
        ("paper", "notebook", "talk", "poster").
    palette : str
        seaborn color palette name.
    font_scale : float
        Multiplier for all font sizes.
    """
    global _THEME_APPLIED
    sns.set_theme(
        context=context,
        style="whitegrid",
        palette=palette,
        font_scale=font_scale,
    )
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 110,
        "axes.titlesize": "large",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    _THEME_APPLIED = True


def histogram(
    data,
    bins="auto",
    title=None,
    xlabel="Value",
    ylabel="Count",
    color=None,
    stat="count",
    kde=False,
    discrete=False,
    ax=None,
    **kwargs,
):
    """Plot a pre-themed histogram with seaborn and return the Axes.

    Parameters
    ----------
    data : array-like
        The values to histogram (e.g. LCG output).
    bins : int | sequence | str
        Number of bins, explicit bin edges, or a binning rule
        ("auto", "fd", "sturges", ...). Ignored if discrete=True.
    title : str, optional
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    color : str, optional
        Bar color; defaults to the first palette color.
    stat : str
        What the bars show: "count", "density", "probability",
        "percent", or "frequency".
    kde : bool
        Overlay a kernel density estimate (continuous data only).
    discrete : bool
        If True, center one bar per integer value — ideal for the
        integer output of a linear congruential generator.
    ax : matplotlib.axes.Axes, optional
        Draw onto an existing Axes instead of creating a new figure.
    **kwargs
        Passed through to seaborn.histplot.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes, so you can keep customizing if needed.
    """
    if not _THEME_APPLIED:
        set_theme()

    data = np.asarray(data)

    if ax is None:
        _, ax = plt.subplots()

    sns.histplot(
        data,
        bins=bins if not discrete else "auto",
        color=color,
        stat=stat,
        kde=kde,
        discrete=discrete,
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
        **kwargs,
    )

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel.capitalize() if stat == "count" else stat.capitalize())

    return ax