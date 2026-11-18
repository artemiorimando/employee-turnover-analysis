"""Shared matplotlib style for artemiorimando.com figures.

Site palette: navy #0d1b2a, blue #1a3a5c, accent #63b3ed, text #2d3748, muted #718096,
border #e2e8f0. Light background, no top/right spines, Inter-like sans font.
Figures are saved at 2x for retina screens with a white background.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

NAVY = "#0d1b2a"
BLUE = "#1a3a5c"
ACCENT = "#63b3ed"
TEXT = "#2d3748"
MUTED = "#718096"
BORDER = "#e2e8f0"
RED = "#e53e3e"
SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": BORDER,
    "axes.linewidth": 1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": BORDER,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.color": TEXT,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "axes.prop_cycle": mpl.cycler(color=[BLUE, ACCENT, NAVY, "#a0aec0", RED]),
})


def figure(width=9, height=5.2):
    return plt.subplots(figsize=(width, height))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)
