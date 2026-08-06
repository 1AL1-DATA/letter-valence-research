"""
Shared matplotlib style and colour palette for the letter-valence project.

Palette (used everywhere for visual consistency):

  black         #000000   primary text, axes, borders
  prussian_blue #14213d   background elements, line emphasis
  orange        #fca311   highlight bars / lines
  alabaster     #e5e5e5   neutral fills, light bars
  white         #ffffff   background, text on dark

Usage:

    from src.style import apply_style, PALETTE
    apply_style()
    fig, ax = plt.subplots()
    ax.bar(..., color=PALETTE["orange"])
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# Project palette — the only place these hex codes are defined.
PALETTE = {
    "black":        "#000000",
    "prussian":     "#14213d",
    "orange":       "#fca311",
    "alabaster":    "#e5e5e5",
    "white":        "#ffffff",
}

# Convenience semantic aliases. These map chart roles to palette entries
# so a figure can ask for "primary", "muted", etc. and we can change the
# mapping in one place.
SEMANTIC = {
    "primary":    PALETTE["prussian"],   # strongest emphasis
    "secondary":  PALETTE["orange"],     # secondary emphasis
    "muted":      PALETTE["alabaster"],  # neutral / background
    "text":       PALETTE["black"],      # labels, axes
    "highlight":  PALETTE["orange"],     # call-out values
    "negative":   PALETTE["prussian"],   # for "below baseline"
    "positive":   PALETTE["orange"],     # for "above baseline"
}


def apply_style() -> None:
    """Apply the project-wide matplotlib rcParams.

    Idempotent — call at the top of any figure script.
    """
    plt.rcParams.update({
        # Resolution
        "figure.dpi": 100,
        "savefig.dpi": 300,
        # Fonts
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        # Frame
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": PALETTE["black"],
        "axes.linewidth": 0.8,
        # Text colour
        "text.color": PALETTE["black"],
        "axes.labelcolor": PALETTE["black"],
        "xtick.color": PALETTE["black"],
        "ytick.color": PALETTE["black"],
        # Grid
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.color": PALETTE["black"],
        # Layout
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        # Legend
        "legend.frameon": False,
        "legend.edgecolor": PALETTE["black"],
    })


# Auto-apply on import so `import src.style` is enough.
apply_style()
