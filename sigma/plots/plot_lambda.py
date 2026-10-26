"""Lambda_max ablation: two panels (bootstrap null | Gaussian null), median null p-value vs N_test with one line per
tangent-block cut Lambda_max, shaded light (small Lambda_max, few kept directions) to dark (large). The point null
does not depend on Lambda_max and is drawn once per panel as a grey reference. 2D toy, K = 64, J = 500, sigma = 0.4,
N_train = 100k, 20 fits; bars = 68% interval of the median (bootstrap over fits), offset slightly in N_test.

usage: python plot_lambda.py  -> figures/lambda_toy2d_K64
Input: data/pvalues/lambda_toy2d_K64.json
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from common import DATA, INK, GRID, MUTED, GREY, med_ci, style, save

NTES = [1000, 10000, 20000, 50000, 100000, 200000]
ARMS = [("1e-4", r"10^{-4}", 89), ("1e-3", r"10^{-3}", 129), ("0.008", r"0.008", 185),
        ("0.03", r"0.03", 217), ("0.1", r"0.1", 243)]                    # (key, label, kept directions M of 383)
PANELS = [("p_bootstrap", "fit_bootstrap", cm.Reds, "D", "Bootstrap null"),
          ("p_gaussian", "fit_gaussian", cm.Blues, "o", "Gaussian null")]
SHADES = np.linspace(0.35, 0.95, len(ARMS))
DODGE = np.linspace(-0.035, 0.035, len(ARMS))          # in log10(N_test)


def curve(d, arm, key, fkey, rng):
    xs, ms, lo, hi = [], [], [], []
    for n in NTES:
        cell = d.get(f"toy2d_K64_lam{arm}_{n}")
        if not cell or not cell[key]:
            continue
        v, fit = np.array(cell[key]), np.array(cell[fkey])
        a, b = med_ci(v, fit, rng)
        xs.append(n); ms.append(np.median(v)); lo.append(a); hi.append(b)
    return np.array(xs), np.array(ms), np.array(lo), np.array(hi)


def main():
    d = json.load(open(os.path.join(DATA, "pvalues", "lambda_toy2d_K64.json")))
    rng = np.random.default_rng(20261022)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4), sharey=True)
    px, pm, _, _ = curve(d, "0.008", "p_point", "fit_gaussian", rng)        # Lambda-independent
    for ax, (key, fkey, cmap, mk, title) in zip(axes, PANELS):
        ax.plot(px, pm, color=GREY, ls=":", lw=1.4, marker="^", ms=4.5, zorder=1,
                label="Point null (any $\\Lambda_{\\max}$)")
        for (arm, lab, M), shade, dx in zip(ARMS, SHADES, DODGE):
            col = cmap(shade)
            xs, ms, lo, hi = curve(d, arm, key, fkey, rng)
            xd = xs * 10 ** dx
            ax.plot(xd, ms, color=col, lw=1.7, marker=mk, ms=4.5, zorder=3,
                    label=rf"$\Lambda_{{\max}}={lab}$  ($M\approx{M}$)")
            ax.errorbar(xd, ms, yerr=[ms - lo, hi - ms], fmt="none", ecolor=col, elinewidth=1, capsize=1.5, zorder=2)
        ax.axhline(0.5, color=GRID, lw=1, ls=":", zorder=0)
        ax.axvline(100_000, color=MUTED, lw=1, ls="--", zorder=0)
        ax.set_xscale("log"); ax.set_xlim(700, 3e5); ax.set_ylim(-0.02, 1.0)
        ax.set_xticks([1000, 10000, 100000]); ax.set_xticklabels(["1k", "10k", "100k"])
        ax.set_title(title, color=INK, fontsize=10.5)
        ax.set_xlabel(r"$N_{\rm test}$", color=INK)
        ax.tick_params(labelsize=8.5)
        style(ax, which="major")
        ax.legend(frameon=False, fontsize=7.6, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.17),
                  ncol=2, handlelength=1.8, columnspacing=1.2)
    axes[0].set_ylabel("null $p$-value (median over sets)", color=INK, fontsize=9.5)
    fig.tight_layout()
    save(fig, "lambda_toy2d_K64")


if __name__ == "__main__":
    main()
