"""Shared figure style for the paper (p-value ladders + marginals).

    import paper_style as ps
    ps.apply()                      # serif fonts; call AFTER any other style (e.g. mplhep's CMS style)
    fig, ax = plt.subplots()
    ax.plot(x, y, **ps.LINE["bootstrap"], label="Bootstrap null")
    ps.style_axes(ax)
    ps.save(fig, "name")            # writes name.pdf + name.png

Colours are hex strings, usable anywhere in matplotlib.
"""
import matplotlib.pyplot as plt

# ---- ink (text, axes) --------------------------------------------------------
INK = "#33322e"      # all text, tick labels, tick marks
GRID = "#c9c8c0"     # spines, reference line at p = 0.5
MUTED = "#8a897f"    # secondary lines/labels (e.g. the N_test = N_train marker)

# ---- series colours ------------------------------------------------------------
BLUE = "#2a78d6"     # Gaussian null
RED = "#c0392b"      # Bootstrap null
GREY = "#9a9a9a"     # plug-in

# opaque band tints of the same hues (no alpha: overlapping alpha bands wash out to grey).
# Draw the wider band first and the narrower one on top.
BLUE_BAND = "#b3cdf0"
RED_BAND = "#f0b9b1"

# ---- line/marker specs for the p-value ladders (pass as **kwargs to ax.plot) ---
LINE = {
    "bootstrap": dict(color=RED, lw=1.9, ls="-.", marker="D", ms=5.5, mfc="white", mew=1.5),
    "gaussian": dict(color=BLUE, lw=1.9, ls="-", marker="o", ms=5.5, mfc=BLUE, mew=1.5),
    "plugin": dict(color=GREY, lw=1.4, ls=":", marker="^", ms=4.5),
}
ERRORBAR = dict(fmt="none", elinewidth=1.0, capsize=2.5)   # + ecolor=<series colour>

# ---- marginals -----------------------------------------------------------------
MODEL_LINE = dict(color=MUTED, lw=1.0, ls="-")                                # fitted model
DATA_POINTS = dict(fmt="x", ms=4, mew=1.0, color=INK, elinewidth=0.9)        # held-out data (errorbar)

TITLE_SIZE = 10.5
LEGEND = dict(frameon=False, labelcolor=INK)


def apply():
    """Serif text + serif math. Call after any other style so it wins."""
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})


def style_axes(ax):
    """Inward ticks on all four sides, ink-coloured ticks/labels, light grey spines."""
    ax.tick_params(direction="in", which="both", colors=INK, top=True, right=True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def ntrain_marker(ax, ntrain, label=r"$N_{\rm test}=N_{\rm train}$"):
    """Vertical dashed marker at N_test = N_train, with its rotated label (log-x axes)."""
    ax.axvline(ntrain, color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=1)
    ax.text(ntrain * 0.88, 0.97, label, rotation=90, ha="right", va="top", color=MUTED, fontsize=7.5)


def half_line(ax):
    """Dotted reference at p = 0.5."""
    ax.axhline(0.5, color=GRID, lw=1.0, ls=":", zorder=0)


def ntest_ticks(ax, ticks):
    """Log-x ticks labelled 1k, 10k, ... with minor ticks off."""
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000}k" if t >= 1000 else str(t) for t in ticks])
    ax.minorticks_off()


def save(fig, stem, dpi=170):
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
