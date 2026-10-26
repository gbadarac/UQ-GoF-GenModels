"""Shared paths, style and statistics for the paper plots."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
INK, GRID, MUTED = "#33322e", "#c9c8c0", "#8a897f"
BLUE, RED, GREY = "#2a78d6", "#c0392b", "#9a9a9a"
BLUE_BAND, RED_BAND = "#b3cdf0", "#f0b9b1"


def med_ci(v, fit, rng, n_boot=2000):
    """68% interval of the median p-value, resampling FITS (the test sets of one fit are not independent)."""
    groups = [v[fit == b] for b in np.unique(fit)]
    nb = len(groups)
    meds = np.empty(n_boot)
    for i in range(n_boot):
        meds[i] = np.median(np.concatenate([groups[j] for j in rng.integers(0, nb, nb)]))
    return np.percentile(meds, [16, 84])


def style(ax, which="both"):
    ax.tick_params(direction="in", which=which, colors=INK, top=True, right=True)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def save(fig, stem, dpi=170):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"), dpi=dpi, bbox_inches="tight")
    print("wrote", os.path.join(FIGS, stem + ".pdf"))
