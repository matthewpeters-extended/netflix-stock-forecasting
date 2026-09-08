"""Shared plotting helpers.

One place for figure style and saving, so every chart in the repository looks
like it belongs to the same project.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from . import config

FIGSIZE_WIDE = (13, 5)
FIGSIZE_GRID = (14, 9)
DPI = 150


def use_project_style() -> None:
    """Apply a consistent, readable style. Call once per notebook."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def save(fig, name: str) -> None:
    """Save a figure into reports/figures/ under a stable name."""
    config.ensure_dirs()
    path = config.FIGURES / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"saved -> {path.relative_to(config.PROJECT_ROOT)}")
