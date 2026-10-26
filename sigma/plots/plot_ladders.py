"""Closure tests (Figs. 5 and 9): median null p-value vs N_test under the bootstrap, Gaussian and point nulls, one
panel per number of components K. Bars: 68% interval of the median (bootstrap over fits).

usage: python plot_ladders.py toy2d     -> figures/ladder_toy2d   (2D toy,      K = 16, 32, 64, 128, 20 fits)
       python plot_ladders.py jets4d    -> figures/ladder_jets4d  (4D jet embedding, K = 128, 256, 512, 10 fits)
Input: data/pvalues/ladder_<bench>.json  {"<bench>_K<K>_<N_test>": {"p_bootstrap", "fit_bootstrap",
       "p_gaussian", "p_point", "fit_gaussian", "n_fits"}}  (per-set p-values; fit = index of the fit each set belongs to)
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from common import DATA, INK, GRID, MUTED, BLUE, RED, GREY, med_ci, style, save

CFG = {"toy2d": dict(ks=[16, 32, 64, 128], xlim=(700, 3e5), ticks=[1000, 10000, 100000]),
       "jets4d": dict(ks=[128, 256, 512], xlim=(70, 1.5e5), ticks=[100, 1000, 10000, 100000])}
NTRAIN = 100_000


def main(bench):
    cfg = CFG[bench]
    d = json.load(open(os.path.join(DATA, "pvalues", f"ladder_{bench}.json")))
    rng, rng_point = np.random.default_rng(20260924), np.random.default_rng(20260925)
    fig, axes = plt.subplots(1, len(cfg["ks"]), figsize=(2.9 * len(cfg["ks"]), 3.9), sharey=True)
    for ax, K in zip(axes, cfg["ks"]):
        pre = f"{bench}_K{K}_"
        ns = sorted(int(k[len(pre):]) for k in d if k.startswith(pre))
        cells = [d[f"{pre}{n}"] for n in ns]
        for null, col, ls, mk, mfc, lab in (("gaussian", BLUE, "-", "o", None, "Gaussian null"),
                                            ("bootstrap", RED, "-.", "D", "white", "Bootstrap null")):
            fits = [np.asarray(c[f"fit_{null}"], int) for c in cells]
            p = [np.asarray(c[f"p_{null}"], float) for c in cells]
            med = np.array([np.median(x) for x in p])
            ci = np.array([med_ci(x, f, rng) for x, f in zip(p, fits)])
            ax.plot(ns, med, color=col, lw=1.9, ls=ls, marker=mk, ms=5.5, mfc=mfc or col, mew=1.5, label=lab, zorder=3)
            ax.errorbar(ns, med, yerr=[med - ci[:, 0], ci[:, 1] - med], fmt="none", ecolor=col, elinewidth=1.0,
                        capsize=2.5, zorder=3)
            if null == "gaussian":         # the point null's p-values live with the Gaussian arm (same toy data)
                pp = [np.asarray(c["p_point"], float) for c in cells]
                mp = np.array([np.median(x) for x in pp])
                cp = np.array([med_ci(x, f, rng_point) for x, f in zip(pp, fits)])
                ax.plot(ns, mp, color=GREY, lw=1.4, ls=":", marker="^", ms=4.5, label="Point null", zorder=2)
                ax.errorbar(ns, mp, yerr=[mp - cp[:, 0], cp[:, 1] - mp], fmt="none", ecolor=GREY, elinewidth=1.0,
                            capsize=2.5, zorder=2)
        ax.axvline(NTRAIN, color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=1)
        ax.text(NTRAIN * 0.88, 0.97, r"$N_{\rm test}=N_{\rm train}$", rotation=90, ha="right", va="top", color=MUTED,
                fontsize=7.5)
        ax.axhline(0.5, color=GRID, lw=1.0, ls=":", zorder=0)
        ax.set_xscale("log"); ax.set_ylim(-0.02, 1.0); ax.set_xlim(*cfg["xlim"])
        ax.set_xticks(cfg["ticks"])
        ax.set_xticklabels([f"{t // 1000}k" if t >= 1000 else str(t) for t in cfg["ticks"]]); ax.minorticks_off()
        ax.set_title(f"$K={K}$,  $N_{{\\rm train}}=100k$", color=INK, fontsize=10.5)
        ax.set_xlabel(r"$N_{\rm test}$", color=INK)
        style(ax)
    axes[0].set_ylabel("null $p$-value  (median over sets)", color=INK)
    h, l = axes[1].get_legend_handles_labels()
    order = [l.index(x) for x in ("Bootstrap null", "Gaussian null", "Point null")]
    axes[0].legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=9, labelcolor=INK,
                   loc="upper left")
    fig.tight_layout()
    save(fig, f"ladder_{bench}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "toy2d")
