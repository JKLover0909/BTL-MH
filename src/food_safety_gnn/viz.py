"""Shared accessible visualization defaults for project notebooks."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt

# Dataviz skill reference palette (light mode).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"


def apply_project_style() -> None:
    """Apply a consistent, readable style for notebook charts."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "text.color": TEXT_PRIMARY,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "figure.dpi": 120,
        }
    )


def series_color(index: int) -> str:
    """Return the fixed categorical color for a series index."""
    return SERIES[index % len(SERIES)]


def bar_chart(
    labels: Iterable[str],
    values: Iterable[float],
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "Count",
    color: str | None = None,
    horizontal: bool = False,
    figsize: tuple[float, float] = (8, 4.5),
):
    """Render a simple single-series bar/column chart."""
    apply_project_style()
    figure, axis = plt.subplots(figsize=figsize)
    labels = list(labels)
    values = list(values)
    paint = color or SERIES[0]
    if horizontal:
        axis.barh(labels, values, color=paint, height=0.7, edgecolor=SURFACE, linewidth=2)
        axis.set_xlabel(ylabel)
        axis.set_ylabel(xlabel)
    else:
        axis.bar(labels, values, color=paint, width=0.7, edgecolor=SURFACE, linewidth=2)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=30)
    axis.set_title(title)
    figure.tight_layout()
    return figure, axis


def line_chart(
    x_values: Iterable[float],
    y_values: Iterable[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str | None = None,
    figsize: tuple[float, float] = (8, 4.5),
):
    """Render a single-series line chart."""
    apply_project_style()
    figure, axis = plt.subplots(figsize=figsize)
    axis.plot(
        list(x_values),
        list(y_values),
        color=color or SERIES[0],
        linewidth=2,
        marker="o",
        markersize=4,
        markerfacecolor=color or SERIES[0],
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    figure.tight_layout()
    return figure, axis
