"""
plot_pvalue_vs_ntest.py
-----------------------
NPLM p-value vs N_test — frozen vs constrained (and single) — in the reference
group-standard style: a median p-value line per configuration with a single
light IQR (25-75%) toy-spread band and joint-bootstrap 95% CI error bars on the
median, plus the nominal (p=0.5) and bank-floor (p=1/(B+1)) reference lines.

Design (mirrors the reference repo's per-run -> merge -> plot split):
  * LRT.py is UNCHANGED — this reads only the seed*_T.npy files it already writes.
  * analyse_LRT_output.py is UNCHANGED — that stays the single-run diagnostic;
    this is the cross-run scan aggregation + plot.

What it does:
  1. discovers run-tag dirs under the given --base dir(s),
  2. parses (Nens, Ntest, mode = frozen | constrained) from each folder name,
  3. per run computes, for each TEST toy i, the empirical p-value against the
     NULL toys:  p_i = (#{T_null >= T_test,i} + 1) / (B + 1)
     (identical estimator to analyse_LRT_output.py, just per-toy instead of the
     single median), then the {25, 50, 75} percentiles of {p_i} plus a joint
     (double) bootstrap 95% CI of the median — resampling BOTH null and test
     toys so the CI includes the finite-B Monte-Carlo error of the null tail,
     not just the test-toy spread,
  4. plots one curve per (Nens, mode): median + a single light IQR band +
     median CI error bars; distinct markers (constrained=o, frozen=s,
     single=triangle); log x-axis; nominal p=0.5 and bank-floor reference lines.

NOTE on the median line: because the empirical p-value is a monotonic function
of T_obs, the median of the per-toy p-values equals the p-value of the median
test T — i.e. this p50 line is exactly the `p-value` number analyse_LRT_output.py
already prints per run. The bands are the extra piece (they need the full per-toy
p-value distribution, which the single-run script does not emit).

Caveat 1: with the 10-fit NF layout (choice*/ subdirs) the band is POOLED over the ensemble
choices, so it captures the fit-to-fit (model-choice) spread AND the test-toy spread — the
honest experiment-to-experiment band. For a LEGACY single-run cell (one dir, e.g. kernels) the
"pool" is one fit, so the band is just that fit's test-toy IQR of correlated bootstrap resamples
and STILL UNDERESTIMATES the true spread — label those as such.

Caveat 2: p is discretized on a 1/(B+1) grid. When the median test T exceeds
every null T, all p_i pin to the bank floor 1/(B+1): the band and CI then
collapse to zero width — this means "p unresolved below the floor, need more
null toys B", NOT an infinitely precise p. The plot annotates the bank floor;
read a zero-width bar there as an upper limit, not a measurement.

Usage:
  python plot_pvalue_vs_ntest.py \
     --base LRT/results/kernels/2d_ensemble LRT/results/kernels/2d_single_model \
     --out  LRT/results/figs/pvalue_vs_ntest.pdf \
     [--csv LRT/results/figs/pvalue_vs_ntest.csv] [--title "2d_gmm kernels"]
"""
import os, glob, re, argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One light IQR band (25-75) + median + bootstrap CI of the median. No nested bands.
QS = [25.0, 50.0, 75.0]
N_BOOT = 2000
RNG = np.random.default_rng(20260825)

# Axis furniture (spines, reference lines, annotations) stays muted grey; the
# three data curves are blue / red / green (composite / point-null / single).
INK, GRID, MUTED = "#33322e", "#c9c8c0", "#8a897f"
BLUE, RED, GREEN = "#2a78d6", "#d62728", "#2ca02c"
# mode -> (color, marker, legend label). Group terminology: constrained w = COMPOSITE-null
# (weight nuisances profiled); frozen w = POINT-null (plug-in at w_hat, density treated as exact).
MODE_STYLE = {
    "composite": (BLUE, "o", "composite (uncertainty propagated)"),
    "point":     (RED,  "s", r"point null (plug-in at $\hat{w}$)"),
}


def collect_T(run_dir, mode):
    """Load all finite seed*_T.npy under run_dir/<mode>/seed*/ into a 1-D array."""
    out = []
    for sd in sorted(glob.glob(os.path.join(run_dir, mode, "seed*"))):
        f = os.path.join(sd, os.path.basename(sd) + "_T.npy")
        if os.path.exists(f):
            v = float(np.load(f))
            if np.isfinite(v):
                out.append(v)
    return np.array(out, dtype=float)


def _pertoy_p(run_dir, min_toys=5):
    """Per-obs-toy empirical p-values for ONE fit (choice), each vs THAT fit's OWN null:
    p_i = (#{T_null >= T_test,i} + 1)/(B + 1). Returns (p, tc, tt) or None if too few toys."""
    tc = collect_T(run_dir, "calibration")
    tt = collect_T(run_dir, "test")
    if len(tc) < min_toys or len(tt) < min_toys:
        return None
    B = len(tc)
    k = (tc[None, :] >= tt[:, None]).sum(axis=1)          # (n_test,)
    return (k + 1.0) / (B + 1.0), tc, tt


def pooled_pvalue_percentiles(run_dirs, min_toys=5):
    """POOLED p-value band over the ensemble choices (Sean's 10-fit recipe).

    Each choice's obs toys are turned into p-values against THAT choice's OWN null
    (self-calibration), then ALL the per-toy p-values are POOLED across the choices and the
    {25,50,75} percentiles taken. This pooled band captures BOTH the fit-to-fit (model-choice)
    spread AND the test-toy spread — the honest experiment-to-experiment band that supersedes
    the single-fit test-toy band. With one run_dir it reduces to the old single-fit percentiles.

    ci_lo/ci_hi: a HIERARCHICAL bootstrap of the POOLED median — resample the choices
    (fit-to-fit), and within each resampled choice resample its null + obs toys (test-toy +
    finite-B) — so the CI reflects all three error sources.

    Returns (q25, med, q75, ci_lo, ci_hi, n_test_total, B_typ, n_choices) or None."""
    fits = []
    for rd in run_dirs:
        res = _pertoy_p(rd, min_toys)
        if res is not None:
            fits.append(res)
    if not fits:
        return None
    pooled = np.concatenate([p for p, _, _ in fits])
    q25, med, q75 = np.percentile(pooled, QS)
    n_ch = len(fits)
    med_boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ch = RNG.integers(0, n_ch, n_ch)                  # resample choices (fit-to-fit)
        ps = []
        for ci in ch:
            _, tc, tt = fits[ci]
            B, nt = len(tc), len(tt)
            cb = tc[RNG.integers(0, B, B)]                # null tail resample (finite-B)
            tb = tt[RNG.integers(0, nt, nt)]              # test-toy resample
            kk = (cb[None, :] >= tb[:, None]).sum(axis=1)
            ps.append((kk + 1.0) / (B + 1.0))
        med_boot[b] = np.median(np.concatenate(ps))
    lo, hi = np.percentile(med_boot, [2.5, 97.5])
    B_typ = int(np.median([len(tc) for _, tc, _ in fits]))
    return q25, med, q75, lo, hi, len(pooled), B_typ, n_ch


def parse_tag(name):
    """Parse (Nens, Ntest, mode) from a run-tag folder name; None if it isn't one."""
    m_n = re.search(r"Nens(\d+)", name)
    m_t = re.search(r"Ntest(\d+)", name)
    if not (m_n and m_t):
        return None
    name = name.rstrip("/")
    # New tokens (point_null / composite_null); keep back-compat with the old frozen/constrained
    # names so legacy dirs still parse.
    if name.endswith("composite_null") or name.endswith("constrained"):
        mode = "composite"
    elif "point_null" in name or "frozen_weights" in name:
        mode = "point"
    else:
        return None
    return int(m_n.group(1)), int(m_t.group(1)), mode


def find_run_dirs(base):
    """Run-tag dirs under a base. Legacy/kernels layout: directly under <base>/. 10-fit NF
    layout: nested under <base>/choice<k>/. Return both — grouping the 10 choices of a cell
    (identical run-tag name) happens by name in main()."""
    dirs = []
    for pat in (os.path.join(base, "*"), os.path.join(base, "choice*", "*")):
        for d in sorted(glob.glob(pat)):
            if os.path.isdir(d) and parse_tag(os.path.basename(d)) is not None:
                dirs.append(d)
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True,
                    help="One or more dirs whose immediate subdirs are run-tag dirs "
                         "(e.g. .../2d_ensemble .../2d_single_model).")
    ap.add_argument("--out", default=None,
                    help="Output path. Default: <first --base>/figs/pvalue_vs_ntest.pdf "
                         "(figures live next to the runs they summarize).")
    ap.add_argument("--csv", default=None, help="Optional summary.csv dump.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--alpha_level", type=float, default=0.05)
    ap.add_argument("--nens", nargs="+", type=int, default=None,
                    help="Only include these ensemble sizes (the single model, Nens=1, is "
                         "always kept as the baseline). Default: all sizes found.")
    ap.add_argument("--min_ntest", type=int, default=0,
                    help="Skip runs with Ntest below this — drops off-scan/debug runs "
                         "(e.g. the Ntest=1000/M40 test).")
    ap.add_argument("--mode", choices=["all", "point", "composite"], default="all",
                    help="Restrict ENSEMBLE curves to one null mode (the single model is "
                         "always kept as the baseline). 'composite'/'point' give a clean "
                         "per-size colour scan of that one mode; 'all' (default) keeps both "
                         "(→ the per-size single/point/composite comparison when one --nens).")
    args = ap.parse_args()
    if args.out is None:                 # figs/ default lives inside the primary base dir
        args.out = os.path.join(args.base[0], "figs", "pvalue_vs_ntest.pdf")

    # Group run-tag dirs by their NAME so the choices of a cell (choice0/<tag>, choice1/<tag>,
    # ... — identical <tag> by construction, since choice is an OUT_BASE axis not a run_tag
    # token) pool together. Legacy single-run dirs group as a pool of one.
    groups = {}     # run-tag basename -> list of run_dirs (one per choice)
    for base in args.base:
        for run_dir in find_run_dirs(base):
            groups.setdefault(os.path.basename(run_dir), []).append(run_dir)

    # (Nens, mode) -> list of (Ntest, q25, med, q75, ci_lo, ci_hi)
    rows = {}
    Bs = []
    for tag_name, run_dirs in sorted(groups.items()):
        tag = parse_tag(tag_name)
        if tag is None:
            continue
        nens, ntest, mode = tag
        if ntest < args.min_ntest:
            continue
        if args.nens is not None and nens != 1 and nens not in set(args.nens):
            continue
        if args.mode != "all" and nens != 1 and mode != args.mode:
            continue
        res = pooled_pvalue_percentiles(run_dirs)
        if res is None:
            continue
        q25, med, q75, lo, hi, n_test, B, n_ch = res
        rows.setdefault((nens, mode), []).append((ntest, q25, med, q75, lo, hi))
        Bs.append(B)
        print(f"{mode:11s} Nens={nens:<4d} Ntest={ntest:<7d}  "
              f"med={med:.3f}  IQR=[{q25:.3f},{q75:.3f}]  CI=[{lo:.3f},{hi:.3f}]  "
              f"(fits={n_ch}, pooled n_test={n_test}, B~{B})")

    if not rows:
        raise SystemExit("No usable runs found (need calibration/ + test/ T.npy under --base).")

    out_base = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    # --- ensemble curves: framing auto-selected ---
    #   * ONE null mode across MANY sizes (a size scan: 4D point-only, or a
    #     --mode composite 2D run) -> colour by ENSEMBLE SIZE (Nens).
    #   * MULTIPLE modes (the per-ensemble single/point/composite comparison at one
    #     --nens) -> colour by NULL MODE (composite=blue, point=red) — the script's
    #     original framing (do NOT colour those by size, or the two modes of a single
    #     size collapse to one colour).
    # Both draw the light IQR (25-75%) toy-spread band + median + bootstrap-CI bars. In a
    # size scan the per-size bands overlap (colour-matched, lower alpha) — busy but shows
    # the spread; the per-ensemble Nens* plots give one clean band per size.
    ens_sizes = sorted({n for (n, m) in rows if n != 1 and m in MODE_STYLE})
    ens_modes = sorted({m for (n, m) in rows if n != 1 and m in MODE_STYLE})
    size_scan = (len(ens_modes) == 1 and len(ens_sizes) > 1)
    cmap = plt.cm.plasma
    ecol = {n: cmap(0.12 + 0.76 * i / max(1, len(ens_sizes) - 1))
            for i, n in enumerate(ens_sizes)}
    for (nens, mode), pts in sorted(rows.items()):
        if nens == 1 or mode not in MODE_STYLE:
            continue
        pts = np.array(sorted(pts))                        # sort by Ntest
        x, q25, med, q75, lo, hi = (pts[:, 0], pts[:, 1], pts[:, 2],
                                    pts[:, 3], pts[:, 4], pts[:, 5])
        if size_scan:
            color, mk, lab, band_alpha = (ecol[nens], MODE_STYLE[mode][1],
                                          rf"$N_{{\rm ens}}={nens}$", 0.10)
        else:
            color, mk, lab = MODE_STYLE[mode]
            band_alpha = 0.16
        if len(x) > 1:
            ax.fill_between(x, q25, q75, color=color, alpha=band_alpha, lw=0)
        ax.plot(x, med, color=color, lw=2.0, marker=mk, ms=6, label=lab, zorder=3)
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none",
                    ecolor=color, elinewidth=1.1, capsize=3, zorder=4)

    # --- single-model baseline: muted dashed line + its own light IQR band ---
    for (nens, mode), pts in sorted(rows.items()):
        if nens != 1:
            continue
        pts = np.array(sorted(pts))
        x, q25, med, q75 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
        if len(x) > 1:
            ax.fill_between(x, q25, q75, color=GREEN, alpha=0.12, lw=0)
        ax.plot(x, med, color=GREEN, lw=1.6, ls=(0, (5, 3)), marker="^", ms=6,
                label="single model (no ensemble)", zorder=2)

    # --- reference lines: nominal p=0.5, bank floor p=1/(B+1), N_test=N_train ---
    Btyp = int(np.median(Bs)) if Bs else 100
    floor = 1.0 / (Btyp + 1)
    ax.axhline(0.5,   color=GRID, lw=1.0, ls=":",   zorder=0)
    ax.axhline(floor, color=GRID, lw=1.0, ls="--",  zorder=0)
    ax.axvline(1e5,   color=GRID, lw=1.0, ls="-.",  zorder=0)
    ax.text(2.15e5, 0.5,           " nominal",    color=MUTED, fontsize=8, va="center")
    ax.text(2.15e5, floor + 0.012, " bank floor", color=MUTED, fontsize=8, va="bottom")
    ax.text(1e5, 0.62, r"$N_{\rm test}=N_{\rm train}$", rotation=90,
            color=MUTED, fontsize=8, va="center", ha="right")

    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    # N_test scan spans 100 -> 200000 (oversampling factor 1e-3 .. 2 of N_train=1e5).
    ax.set_xlim(80, 3.0e5)
    ax.set_xticks([100, 200, 1000, 2000, 10000, 20000, 50000, 100000, 200000])
    ax.set_xticklabels(["100", "200", "1k", "2k", "10k", "20k", "50k", "100k", "200k"], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel(r"$N_{\rm test}$", color=INK, fontsize=13)
    ax.set_ylabel(r"$p$-value   (pooled median, 25–75% over fits)", color=INK, fontsize=13)
    if args.title:
        ax.set_title(args.title, color=INK, fontsize=10.5)
    ax.tick_params(direction="in", which="both", colors=INK, top=True, right=True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.tight_layout()

    for ext in (".png", ".pdf"):
        fig.savefig(out_base + ext, dpi=200, bbox_inches="tight")
    plt.close()
    print("wrote", out_base + ".png", "and", out_base + ".pdf")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["nens", "mode", "ntest", "p25", "p50", "p75", "ci_lo", "ci_hi"])
            for (nens, mode), pts in sorted(rows.items()):
                for row in sorted(pts):
                    w.writerow([nens, mode, *(f"{v:.6g}" for v in row)])
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
