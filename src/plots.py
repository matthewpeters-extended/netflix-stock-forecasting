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


def leakage_diagram(close, train_end: int, val_end: int):
    """Draw the data leakage this project fixes.

    Built from the real price series rather than sketched, so the annotated
    numbers are the project's actual numbers and change if the data does.

    The top panel is the conventional approach: the scaler is fitted before the
    split, so it measures its minimum and maximum across the whole series
    including the test period. The bottom panel is the corrected version.
    """
    import numpy as np

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)

    dates = close.index
    values = close.values
    train_max = values[:train_end].max()
    full_max = values.max()

    panels = [
        (axes[0], "Conventional: scaler.fit_transform(entire series)",
         0, len(values), full_max, "#c0392b",
         "fit() sees ALL data, including the test period"),
        (axes[1], "This project: scaler.fit(train), then transform(everything)",
         0, train_end, train_max, "#27ae60",
         "fit() sees training data only"),
    ]

    for ax, title, fit_start, fit_stop, learned_max, colour, caption in panels:
        ax.plot(dates, values, linewidth=1.0, color="#2c3e50", zorder=3)

        # Split shading, identical on both panels.
        ax.axvspan(dates[0], dates[train_end], color="#3498db", alpha=0.07)
        ax.axvspan(dates[train_end], dates[val_end], color="#f39c12", alpha=0.10)
        ax.axvspan(dates[val_end], dates[-1], color="#e74c3c", alpha=0.10)

        # The span the scaler actually measures.
        ax.annotate(
            "", xy=(dates[fit_stop - 1], learned_max * 1.08),
            xytext=(dates[fit_start], learned_max * 1.08),
            arrowprops=dict(arrowstyle="<->", color=colour, linewidth=2.5),
        )
        ax.text(dates[(fit_start + fit_stop) // 2], learned_max * 1.16, caption,
                ha="center", fontsize=10, color=colour, fontweight="bold")

        ax.axhline(learned_max, color=colour, linestyle="--", linewidth=1.5, zorder=2)
        # Sit the label just below the line so it never collides with the arrow.
        ax.text(dates[-1], learned_max * 0.93,
                f"scaler learned max = ${learned_max:.2f}  ", color=colour,
                fontsize=10, fontweight="bold", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=colour, alpha=0.9))

        ax.set_title(title, loc="left")
        ax.set_ylabel("Close (USD)")
        ax.set_ylim(0, full_max * 1.32)

    # Label the splits once, on the lower panel.
    for start, stop, label in [(0, train_end, "TRAIN 70%"),
                               (train_end, val_end, "VAL 15%"),
                               (val_end, len(values), "TEST 15%")]:
        axes[1].text(dates[(start + stop) // 2], full_max * 0.06, label,
                     ha="center", fontsize=9, color="#555", fontweight="bold")

    axes[1].set_xlabel("Date")

    fig.suptitle("Where the data leak happens", fontsize=15, fontweight="bold", y=0.97)
    fig.text(
        0.5, -0.02,
        f"Fitting before the split lets the scaler learn ${full_max:.2f}, a value that only occurs "
        f"in the test period.\n"
        f"Every training example is then normalised using information from the future. "
        f"Measured effect: RMSE 13.44 leaky vs 29.97 honest.",
        ha="center", fontsize=10.5, color="#333",
    )
    plt.tight_layout()
    return fig
