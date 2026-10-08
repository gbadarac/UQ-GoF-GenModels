"""
plot_pvalue_ladder_paper.py
---------------------------
Paper-style p-value vs N_test ladder: a 2-panel figure (M=32, M=128) per
dimension, in the tangent-method paper look (paper_style_bundle/paper_style.py).

This is the publication version of plot_pvalue_vs_ntest.py — SAME discover ->
parse -> pool over the 10 ensemble choices -> median-p + 95%-CI machinery (the
data functions are imported straight from plot_pvalue_vs_ntest.py so there is
one source of truth), but rendered in the paper's serif/colour style and laid
out as the two-on-a-line panels the paper uses, instead of one figure per M.

Three curves per panel (no bootstrap null — we don't run that leg):
    * single model   (Nens=1, point null)        -> black, dashed, squares
    * point null     (ensemble, plug-in at w_hat) -> grey, dotted, triangles
    * Gaussian null  (ensemble, composite null,
                      fit uncertainty propagated)  -> blue, solid, circles

Mapping to the run-tag tokens LRT.py writes:
    composite_null          -> "Gaussian null"  (the Gaussian-budget null that
                               draws the fit weights ~ N(w_hat, Sigma_w))
    ensemble point_null     -> "Point null"     (plug-in at w_hat)
    Nens1   point_null      -> "Single model"   (no ensemble, hence point only)

Error bars = 95% CI of the pooled median (hierarchical bootstrap over choices +
within-choice null/test toys), identical to plot_pvalue_vs_ntest.py.

Usage:
  python plot_pvalue_ladder_paper.py              # both 2D and 4D
  python plot_pvalue_ladder_paper.py --dim 2d
  python plot_pvalue_ladder_paper.py --dim 4d --out /path/fig.pdf
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# paper_style lives in the repo-root bundle; import it as the single style source.
sys.path.insert(0, os.path.join(HERE, "..", "paper_style_bundle"))
import paper_style as ps
# Reuse the data-collection / pooling logic verbatim (no duplication of the stats).
from plot_pvalue_vs_ntest import (parse_tag, find_run_dirs, pooled_pvalue_percentiles)

# M (= ensemble size, Nens) panels to draw, left -> right.
M_PANELS = [32, 128]

# Per-dimension plot furniture. N_train = 100k for both (see project memory).
DIM_CFG = {
    "2d": dict(bases=["results/nf/2d_uniform_ensemble"],
               ntrain=100_000, ntrain_lab="100k",
               ticks=[100, 1000, 10000, 100000], xlim=(70, 3.0e5),
               out="results/nf/2d_uniform_ensemble/figs/pvalue_ladder_paper.pdf"),
    "4d": dict(bases=["results/nf/4d_uniform_ensemble"],
               ntrain=100_000, ntrain_lab="100k",
               ticks=[100, 1000, 10000, 100000], xlim=(70, 1.5e5),
               out="results/nf/4d_uniform_ensemble/figs/pvalue_ladder_paper.pdf"),
}

# series key -> (plot kwargs, error-bar colour, legend label). Order here is the
# legend order (single -> point -> Gaussian, as requested).
SINGLE_STYLE = dict(color=ps.INK, lw=1.5, ls=(0, (5, 2)), marker="s", ms=4.5,
                    mfc=ps.INK, mew=1.0)
SERIES = [
    ("single",   SINGLE_STYLE,        ps.INK,  "Single model"),
    ("point",    ps.LINE["plugin"],   ps.GREY, "Point null"),
    ("gaussian", ps.LINE["gaussian"], ps.BLUE, "Gaussian null"),
]


def gather_rows(bases):
    """(Nens, mode) -> list of (Ntest, median_p, ci_lo, ci_hi), pooled over the
    10 ensemble choices. Thin wrapper over plot_pvalue_vs_ntest's pooling."""
    groups = {}                       # run-tag basename -> [run_dir per choice]
    for base in bases:
        for run_dir in find_run_dirs(base):
            groups.setdefault(os.path.basename(run_dir), []).append(run_dir)
    rows = {}
    for tag_name, run_dirs in sorted(groups.items()):
        tag = parse_tag(tag_name)
        if tag is None:
            continue
        nens, ntest, mode = tag
        res = pooled_pvalue_percentiles(run_dirs)
        if res is None:
            continue
        _q25, med, _q75, lo, hi, _n, _B, _nch = res
        rows.setdefault((nens, mode), []).append((ntest, med, lo, hi))
    return rows


def series_points(rows, key, M):
    """Pull the (Ntest, med, lo, hi) list for one legend series in panel M.
    single = the Nens=1 point-null baseline (shared across panels)."""
    if key == "single":
        pts = rows.get((1, "point"))
    elif key == "point":
        pts = rows.get((M, "point"))
    else:                                     # gaussian = composite null
        pts = rows.get((M, "composite"))
    return np.array(sorted(pts)) if pts else None


def draw_series(ax, pts, style, ecol, label):
    x, med, lo, hi = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    ax.plot(x, med, **style, label=label, zorder=3)
    ax.errorbar(x, med, yerr=[med - lo, hi - med], ecolor=ecol, **ps.ERRORBAR,
                zorder=3)


def make_figure(dim, rows, out):
    cfg = DIM_CFG[dim]
    fig, axes = plt.subplots(1, len(M_PANELS), figsize=(2.9 * len(M_PANELS), 3.9),
                             sharey=True)
    for ax, M in zip(axes, M_PANELS):
        for key, style, ecol, label in SERIES:
            pts = series_points(rows, key, M)
            if pts is None:
                print(f"  [{dim} M={M}] no data for '{label}'")
                continue
            draw_series(ax, pts, style, ecol, label)
        ps.ntrain_marker(ax, cfg["ntrain"])
        ps.half_line(ax)
        ps.ntest_ticks(ax, cfg["ticks"])
        ax.set_xlim(*cfg["xlim"])
        ax.set_ylim(-0.02, 1.0)
        ax.set_title(rf"$M={M}$,  $N_{{\rm train}}={cfg['ntrain_lab']}$",
                     fontsize=ps.TITLE_SIZE)
        ax.set_xlabel(r"$N_{\rm test}$")
        ps.style_axes(ax)
    axes[0].set_ylabel(r"null $p$-value  (median over sets)", color=ps.INK)
    # legend inline in the left panel (matches the reference figure layout).
    axes[0].legend(fontsize=9, loc="upper left", **ps.LEGEND)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ps.save(fig, os.path.splitext(out)[0])
    plt.close(fig)
    print(f"wrote {os.path.splitext(out)[0]}.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", choices=["2d", "4d", "both"], default="both")
    ap.add_argument("--out", default=None,
                    help="Output path (only with a single --dim; default lives in "
                         "the dimension's figs/ dir).")
    args = ap.parse_args()
    dims = ["2d", "4d"] if args.dim == "both" else [args.dim]
    if args.out and len(dims) > 1:
        ap.error("--out is only valid with a single --dim (not 'both').")
    ps.apply()
    for dim in dims:
        cfg = DIM_CFG[dim]
        bases = [os.path.join(HERE, b) for b in cfg["bases"]]
        rows = gather_rows(bases)
        out = args.out or os.path.join(HERE, cfg["out"])
        make_figure(dim, rows, out)


if __name__ == "__main__":
    main()
