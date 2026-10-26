"""Empirical null distributions (App. D, Fig. 15): 4D jet embedding, K = 256, N_train = 100k, all 10 fits pooled.
Columns: N_test = 1k, 20k, 100k. Rows: composite statistic with its bootstrap null, composite statistic with its
Gaussian null, point statistic with its point null; filled = null, black = observed. The first column also shows a
chi2_nu fitted by maximum likelihood to each null.

usage: python plot_null_dists.py  -> figures/null_dists_jets4d_K256
Input: data/null_dists/jets4d_K256_{gaussian,bootstrap}_nte{001k,020k,100k}.json
       (gaussian file: observed [{t_comp, t_point}], bank_comp, bank_point; bootstrap file: observed, bank_comp)
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from common import DATA, INK, BLUE, RED, GREY, BLUE_BAND, RED_BAND, style, save

NBINS = 20
COLS = [("001k", r"$N_{\rm test}=1k$"), ("020k", r"$N_{\rm test}=20k$"), ("100k", r"$N_{\rm test}=100k$")]
ROWS = (("Bootstrap null", "bcomp", "t_comp", RED_BAND, RED),
        ("Gaussian null", "gcomp", "t_comp", BLUE_BAND, BLUE),
        ("Point null", "point", "t_point", "#e2e1dc", GREY))


def medp(bank, obs):
    return float(np.median((1 + (bank[None, :] >= obs[:, None]).sum(1)) / (len(bank) + 1)))


def fit_dof(x):
    """Maximum-likelihood degrees of freedom of a unit-scale chi2_nu for the sample x."""
    return minimize_scalar(lambda nu: -chi2.logpdf(x, nu).sum(), bounds=(1.0, 500.0), method="bounded").x


def edges_of(vals):
    v = np.concatenate(vals)
    return np.linspace(max(0.0, np.percentile(v, 0.2) - 5), np.percentile(v, 99.8) + 5, NBINS + 1)


def main():
    n = len(COLS)
    fig, axes = plt.subplots(3, n, figsize=(4.6 * n, 9.4), sharex=False, gridspec_kw={"hspace": 0.16, "wspace": 0.18})
    for c, (tag, title) in enumerate(COLS):
        g = json.load(open(os.path.join(DATA, "null_dists", f"jets4d_K256_gaussian_nte{tag}.json")))
        b = json.load(open(os.path.join(DATA, "null_dists", f"jets4d_K256_bootstrap_nte{tag}.json")))
        obs = {"t_comp": np.array([o["t_comp"] for o in g["observed"]]),
               "t_point": np.array([o["t_point"] for o in g["observed"]])}
        banks = {"point": np.array(g["bank_point"]), "gcomp": np.array(g["bank_comp"]), "bcomp": np.array(b["bank_comp"])}
        e_comp = edges_of([banks["gcomp"], banks["bcomp"], obs["t_comp"]])
        e_point = edges_of([banks["point"], obs["t_point"]])
        for r, (lab, bk, ok, fill, edge) in enumerate(ROWS):
            ax = axes[r, c]
            edges = e_point if ok == "t_point" else e_comp
            kw = dict(bins=edges, density=True)
            x = np.linspace(edges[0], edges[-1], 400)
            ax.set_xlim(edges[0], edges[-1])
            if r < len(ROWS) - 1 and ROWS[r + 1][2] == ok:
                ax.tick_params(labelbottom=False)       # the two composite rows share a range
            ax.hist(banks[bk], histtype="stepfilled", color=fill, lw=0, zorder=1, **kw)
            ax.hist(banks[bk], histtype="step", color=edge, lw=1.6, zorder=3, label=lab, **kw)
            if c == 0:
                nu = fit_dof(banks[bk])
                ax.plot(x, chi2.pdf(x, nu), color=edge, lw=1.5, ls=(0, (1, 1.3)), zorder=4,
                        label=rf"fitted $\chi^2_{{\nu}}$, $\hat\nu = {nu:.1f}$")
                print(f"  {lab}: fitted nu = {nu:.2f}")
            ax.hist(obs[ok], histtype="step", color=INK, lw=1.8, zorder=5, label="observed", **kw)
            ax.set_ylim(0, (1.75 if c == 0 else 1.12) * ax.get_ylim()[1])
            style(ax)
            if c == 0:
                ax.set_ylabel(("point" if ok == "t_point" else "composite") + " statistic\ndensity", color=INK)
                ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
            print(f"  {title} {lab}: median p {medp(banks[bk], obs[ok]):.3f}")
        axes[0, c].set_title(title, color=INK, fontsize=10.5)
        axes[-1, c].set_xlabel("test statistic $t$", color=INK)
        axes[-1, c].set_xlim(e_point[0], e_point[-1])
    save(fig, "null_dists_jets4d_K256", dpi=160)


if __name__ == "__main__":
    main()
