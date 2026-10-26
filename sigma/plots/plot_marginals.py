"""1D marginals of one fit with 68% parameter-uncertainty bands (Figs. 4 and 8): the fitted model (line), the
Gaussian band (draws theta_hat + W z, z ~ N(0, I_M), the Gaussian null's parameter draws) and the bootstrap band
(the bootstrap null's cold refits), against held-out data (points); lower panels: ratio to the fit. Bands and model
are bin-averaged so thin structure compares fairly.

usage: python plot_marginals.py toy2d    -> figures/marginals_toy2d_K64   (Fig. 4; held-out data drawn from the truth)
       python plot_marginals.py jets4d   -> figures/marginals_jets4d_K256 (Fig. 8; needs the jet-embedding files,
                                            see sigma_gof/jets4d.py, SIGMA_JETS_DATA)
Input: data/marginals/<bench>_K<K>_fit0_bands.npz  (theta, gaussian_thetas (200, P), bootstrap_thetas (100, P))
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from common import DATA, ROOT, INK, GRID, save

sys.path.insert(0, os.path.join(ROOT, "sigma_gof"))
from gmm import UGMM

POINT, BLUE_BAND, RED_BAND = "#8a897f", "#b3cdf0", "#f0b9b1"
N_FINE, N_COARSE, SUB = 150, 40, 8
N_HO = 100_000                                   # held-out events shown (bin ranges use the full sample)
DATA_LABEL = r"held-out data ($N = 10^{5}$)"
CASES = {"toy2d": "toy2d_K64", "jets4d": "jets4d_K256"}


def binned_marginal(model, theta, d, edges):
    """Bin-averaged marginal density of dimension d on the given edges."""
    logw, mu, Us = model.unpack(theta)
    s = np.sqrt(np.linalg.inv(np.einsum("kij,klj->kil", Us, Us))[:, d, d])
    x = (edges[:-1, None] + (np.arange(SUB)[None, :] + 0.5) / SUB * np.diff(edges)[:, None]).ravel()
    pdf = (np.exp(logw)[None, :] * np.exp(-0.5 * ((x[:, None] - mu[None, :, d]) / s[None, :]) ** 2)
           / (s[None, :] * np.sqrt(2 * np.pi))).sum(1)
    return pdf.reshape(len(edges) - 1, SUB).mean(1)


def panel(ax, rx, d, ctx, legend, title=True):
    fine, coarse = ctx["fine"][d], ctx["coarse"][d]
    xf, xc = 0.5 * (fine[1:] + fine[:-1]), 0.5 * (coarse[1:] + coarse[:-1])
    n_ho = len(ctx["Xho"])
    hc = np.histogram(ctx["Xho"][:, d], coarse)[0]
    dens, derr = hc / (n_ho * np.diff(coarse)), np.sqrt(hc) / (n_ho * np.diff(coarse))
    best = binned_marginal(ctx["model"], ctx["theta"], d, fine)
    best_c = binned_marginal(ctx["model"], ctx["theta"], d, coarse)
    bands = []
    for thetas, col, lab in ((ctx["gauss"], BLUE_BAND, "Gaussian uncertainty (68%)"),
                             (ctx["boot"], RED_BAND, "Bootstrap uncertainty (68%)")):
        arr = np.array([binned_marginal(ctx["model"], th, d, fine) for th in thetas])
        bands.append((np.percentile(arr, [16, 84], axis=0), col, lab))
    for (lo, hi), col, lab in reversed(bands):      # wider bootstrap band underneath, Gaussian band on top
        z = 2 if col == BLUE_BAND else 1
        ax.fill_between(xf, lo, hi, color=col, alpha=1.0, lw=0, label=lab, zorder=z)
        rx.fill_between(xf, lo / best, hi / best, color=col, alpha=1.0, lw=0, zorder=z)
    ax.plot(xf, best, color=POINT, lw=1.0, ls="-", label="fitted model", zorder=4)
    ax.errorbar(xc, dens, yerr=derr, fmt="x", ms=4, mew=1.0, color=INK, elinewidth=0.9, label=DATA_LABEL, zorder=5)
    rx.axhline(1.0, color=POINT, lw=1.0, ls="-", zorder=4)
    rx.errorbar(xc, dens / best_c, yerr=derr / best_c, fmt="x", ms=4, mew=1.0, color=INK, elinewidth=0.9, zorder=5)
    ax.set_ylabel("density", color=INK); rx.set_ylabel("ratio to fit", color=INK)
    rx.set_xlabel(f"$x_{d + 1}$", color=INK); rx.set_ylim(*ctx["rlim"])
    rx.set_xlim(fine[0], fine[-1])
    if title:
        ax.set_title(f"$K={ctx['K']}$,  $N_{{\\rm train}}={ctx['ntr'] // 1000}k$", color=INK, fontsize=10.5)
    for a in (ax, rx):
        a.tick_params(direction="in", which="both", colors=INK, top=True, right=True)
        for sp in a.spines.values():
            sp.set_color(GRID)
    if legend:
        h, l = ax.get_legend_handles_labels()
        order = [l.index(x) for x in (DATA_LABEL, "fitted model", "Bootstrap uncertainty (68%)",
                                      "Gaussian uncertainty (68%)")]
        ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=9, labelcolor=INK, loc="best")


def main(bench):
    z = np.load(os.path.join(DATA, "marginals", f"{CASES[bench]}_fit0_bands.npz"))
    K, D = int(z["K"]), int(z["d"])
    model = UGMM(K, D)
    if bench == "jets4d":
        import jets4d
        Xho = jets4d.heldout_events()                 # the 1M-event test region
    else:
        from toy2d import generate_2d_gmm_skew
        Xho = generate_2d_gmm_skew(1_000_000, seed=424_242)
    lims = [np.quantile(Xho[:, d], [0.005, 0.995]) for d in range(D)]
    ctx = dict(model=model, theta=z["theta"], K=K, ntr=int(z["n_train"]), Xho=Xho[:N_HO],
               gauss=list(z["gaussian_thetas"]), boot=list(z["bootstrap_thetas"]),
               rlim=(0.8, 1.2) if D == 4 else (0.9, 1.1),
               fine=[np.linspace(*lim, N_FINE + 1) for lim in lims],
               coarse=[np.linspace(*lim, N_COARSE + 1) for lim in lims])
    if D == 2:
        fig, axs = plt.subplots(2, D, figsize=(5 * D, 5.4), sharex="col",
                                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
        for d in range(D):
            panel(axs[0, d], axs[1, d], d, ctx, legend=(d == 0))
    else:                                             # 2 x 2 grid of (marginal, ratio) pairs, one title
        fig = plt.figure(figsize=(10, 11.4))
        gs = fig.add_gridspec(5, 2, height_ratios=[3, 1, 0.62, 3, 1], hspace=0.05, wspace=0.25)
        for d in range(D):
            r, c = 3 * (d // 2), d % 2
            ax = fig.add_subplot(gs[r, c]); rx = fig.add_subplot(gs[r + 1, c], sharex=ax)
            ax.tick_params(labelbottom=False)
            panel(ax, rx, d, ctx, legend=(d == 0), title=False)
        fig.suptitle(f"$K={K}$,  $N_{{\\rm train}}={ctx['ntr'] // 1000}k$", color=INK, fontsize=10.5, y=0.905)
    save(fig, f"marginals_{CASES[bench]}", dpi=140)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "toy2d")
