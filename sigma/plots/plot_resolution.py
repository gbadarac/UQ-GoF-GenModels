"""Kernel-width appendix: 2D toy, K = 16 and 32, with the fine kernel width sigma = 0.1 (the nominal sigma = 0.4 is
Fig. 5). Median null p-value vs N_test under the bootstrap, Gaussian and point nulls; bars = 68%
interval of the median (bootstrap over fits). Open markers: cells with fewer fits than the arm has.

usage: python plot_resolution.py  -> figures/resolution_toy2d
Input: data/pvalues/resolution_toy2d.json
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from common import DATA, INK, GRID, MUTED, BLUE, RED, GREY, med_ci, style, save

NTES = [1000, 10000, 20000, 50000, 100000, 200000]
COLS = [("K16", r"$K=16$"), ("K32", r"$K=32$")]
ROWS = [("0.1", r"$\sigma=0.1$")]            # sigma = 0.4 is the main-text ladder (Fig. 5)
NULLS = [("p_bootstrap", "fit_bootstrap", RED, "-.", "D", "Bootstrap null"),
         ("p_gaussian", "fit_gaussian", BLUE, "-", "o", "Gaussian null"),
         ("p_point", "fit_gaussian", GREY, ":", "^", "Point null")]


def main():
    d = json.load(open(os.path.join(DATA, "pvalues", "resolution_toy2d.json")))
    rng = np.random.default_rng(20261021)
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(3.4 * len(COLS), 3.0 * len(ROWS)), sharey=True,
                             squeeze=False)
    for r, (w, wlab) in enumerate(ROWS):
        for c, (K, klab) in enumerate(COLS):
            ax = axes[r, c]
            for key, fkey, col, ls, mk, lab in NULLS:
                xs, ms, lo, hi, full, nfs = [], [], [], [], [], []
                for n in NTES:
                    cell = d.get(f"toy2d_{K}_sigma{w}_{n}")
                    if not cell or not cell[key]:
                        continue
                    v, fit = np.array(cell[key]), np.array(cell[fkey])
                    a, b = med_ci(v, fit, rng)
                    nf = len(np.unique(fit))
                    xs.append(n); ms.append(np.median(v)); lo.append(a); hi.append(b)
                    full.append(nf >= cell["n_fits"]); nfs.append(nf)
                xs, ms = np.array(xs), np.array(ms)
                ax.plot(xs, ms, color=col, ls=ls, lw=1.6, zorder=2, label=lab)
                ax.errorbar(xs, ms, yerr=[ms - np.array(lo), np.array(hi) - ms], fmt="none", ecolor=col, elinewidth=1,
                            capsize=2, zorder=3)
                for x, m, f, nf in zip(xs, ms, full, nfs):
                    ax.plot(x, m, mk, color=col, mfc=col if f else "white", ms=6 if mk != "^" else 5, mew=1.4, zorder=4)
                    if not f and key == "p_bootstrap":
                        ax.annotate(f"{nf}f", (x, m), xytext=(4, 5), textcoords="offset points", fontsize=6.5, color=col)
            ax.axhline(0.5, color=GRID, lw=1, ls=":", zorder=1)
            ax.axvline(100_000, color=MUTED, lw=1, ls="--", zorder=1)
            ax.set_xscale("log"); ax.set_xlim(700, 3e5); ax.set_ylim(-0.02, 1.0)
            ax.set_xticks([1000, 10000, 100000]); ax.set_xticklabels(["1k", "10k", "100k"])
            ax.set_title(f"{klab},  {wlab}", color=INK, fontsize=9.5)
            ax.tick_params(labelsize=8)
            style(ax, which="major")
            if r == len(ROWS) - 1:
                ax.set_xlabel(r"$N_{\rm test}$", color=INK)
        axes[r, 0].set_ylabel("null $p$-value (median over sets)", color=INK, fontsize=9)
    axes[0, 1].legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    save(fig, "resolution_toy2d", dpi=160)


if __name__ == "__main__":
    main()
