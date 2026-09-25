#!/usr/bin/env python3
"""
plot_10fit_aggregate.py
-----------------------
Aggregate the 10 ensemble-CHOICE marginal fits of each configuration into ONE
figure showing the median density + the 25-75% inter-quartile band ACROSS the
fits (the honest fit-to-fit spread), overlaid on the target histogram.

This is the plotting half of the 10-fit fit-to-fit-band deliverable
([[project_paper_campaign_plan]]). The per-choice runs already wrote, per fit,
per feature:  <config>/choice<k>/wifi_ensemble_plots/marginal_feature_<i>_data.npz
with keys f_binned (the model marginal f=sum_i w_i m_i on a dense grid), f_err
(that ONE fit's WiFi covariance band, NOT used here) and bin_centers (the grid,
shared across choices). We stack f_binned over the 10 choices and take the
25/50/75 percentiles per grid point.

Band convention deliberately MATCHES the p-value-vs-N_test plot
(LRT/plot_pvalue_vs_ntest.py): median line + light 25-75 IQR band. The per-fit
plots use each fit's own WiFi cov band, which underestimates the true
experiment-to-experiment spread under flow misspecification (~4-5x too small in
4D); this aggregate makes the band honest.

Reads only the npz files + the target data_train.npy the runs used; it does NOT
reload any flow, so it is fast and needs no GPU.

Usage (from repo root or anywhere):
  python Uncertainty_Modeling/wifi/Fit_Weights/plot_10fit_aggregate.py \
      [--results_root <.../results_fit_weights_NF_uniform>] \
      [--data_2d <data_train.npy>] [--data_4d <data_train.npy>] \
      [--bins_2d 60] [--bins_4d 40]
Writes ensemble_marginals_aggregate.png into each config directory.
"""
import os, sys, re, glob, argparse
import numpy as np
import torch
from scipy import stats

# import the shared plotter from the wifi package (one level up)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils_NF_wifi import plot_ensemble_marginals_aggregate


# Analytic 1D marginals of the 2D toy (features are generated INDEPENDENTLY, so these
# ARE the exact marginals). Verbatim params from
# data/2d_gmm_toymodel/generate_2d_gaussian_heavy_tail_target_data.py:
#   feature 1 (x0): 50/50 mix N(-0.70, 0.12^2) + N(-0.30, 0.12^2)
#   feature 2 (x1): skew-normal(loc=1.0, scale=0.75, alpha=8.0)
def truth_2d(i, grid):
    if i == 0:
        return 0.5 * stats.norm.pdf(grid, -0.70, 0.12) + 0.5 * stats.norm.pdf(grid, -0.30, 0.12)
    return stats.skewnorm.pdf(grid, 8.0, loc=1.0, scale=0.75)

# repo root = .../MonoJet_NPLM_analysis (three levels up: Fit_Weights -> wifi ->
# Uncertainty_Modeling -> repo root)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# per dimensionality group: (results subdir, target data, hist bins, dataset label)
GROUPS = {
    "2d_toymodel": dict(
        data=os.path.join(REPO_ROOT, "data/2d_gmm_toymodel/"
             "2d_gmm_skew_Ntrain100000_Ntest100000_seed42/data_train.npy"),
        bins=40, label="2D toy model", truth_fn=truth_2d,
    ),
    "4d_embedding": dict(
        data=os.path.join(REPO_ROOT, "data/4d_embeddings/"
             "4d_embedding_qcd_Ntrain100000_Ntest100000_seed42/data_train.npy"),
        bins=40, label="4D embedding", truth_fn=None,   # no analytic truth for the embedding
    ),
}


def _load_config(config_dir, n_features):
    """Stack the per-choice model marginals for one configuration.

    Returns (curves_per_feature, grids_per_feature, ferr_per_feature, n_fits) or
    None if no complete choice is found."""
    choice_dirs = sorted(
        glob.glob(os.path.join(config_dir, "choice*")),
        key=lambda p: int(re.search(r"choice(\d+)", os.path.basename(p)).group(1)),
    )
    curves = [[] for _ in range(n_features)]
    ferrs = [[] for _ in range(n_features)]
    grids = [None] * n_features
    n_fits = 0
    for cd in choice_dirs:
        npzs = [os.path.join(cd, "wifi_ensemble_plots", f"marginal_feature_{i+1}_data.npz")
                for i in range(n_features)]
        if not all(os.path.isfile(p) for p in npzs):
            print(f"  [skip] {os.path.basename(cd)}: missing marginal npz")
            continue
        for i, p in enumerate(npzs):
            d = np.load(p)
            g = d["bin_centers"]
            if grids[i] is None:
                grids[i] = g
            elif not np.allclose(g, grids[i]):
                raise ValueError(
                    f"grid mismatch in {p}: choices must share a grid to stack")
            curves[i].append(d["f_binned"])
            ferrs[i].append(d["f_err"])
        n_fits += 1
    if n_fits == 0:
        return None
    curves = [np.asarray(c) for c in curves]      # each (n_fits, n_curve)
    ferrs = [np.asarray(c) for c in ferrs]        # each (n_fits, n_curve)
    return curves, grids, ferrs, n_fits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", default=os.path.join(
        os.path.dirname(__file__), "results_fit_weights_NF_uniform"),
        help="dir holding the 2d_toymodel/ and 4d_embedding/ group subdirs")
    ap.add_argument("--data_2d", default=None, help="override 2D target data_train.npy")
    ap.add_argument("--data_4d", default=None, help="override 4D target data_train.npy")
    ap.add_argument("--bins_2d", type=int, default=None)
    ap.add_argument("--bins_4d", type=int, default=None)
    ap.add_argument("--no_fit2fit", dest="show_fit2fit", action="store_false",
                    help="hide the fit-to-fit 25-75%% overlay; show only the propagated "
                         "+-1sigma test band. Default: BOTH are drawn (propagated +-1sigma "
                         "filled = the test uncertainty; fit-to-fit grey dashed envelope = "
                         "the experiment-to-experiment spread, the p-value plot's band)")
    ap.set_defaults(show_fit2fit=True)
    ap.add_argument("--show_truth", action="store_true",
                    help="overlay the analytic ground-truth line where known (2D toy). "
                         "OFF by default; the ratio panel is ALWAYS to the median fit so "
                         "2D and 4D stay structurally identical either way")
    args = ap.parse_args()

    overrides = {"2d_toymodel": (args.data_2d, args.bins_2d),
                 "4d_embedding": (args.data_4d, args.bins_4d)}

    made = 0
    for group, cfg in GROUPS.items():
        group_dir = os.path.join(args.results_root, group)
        if not os.path.isdir(group_dir):
            print(f"[{group}] no such dir, skipping: {group_dir}")
            continue

        data_path, bins = overrides[group]
        data_path = data_path or cfg["data"]
        bins = bins or cfg["bins"]
        if not os.path.isfile(data_path):
            print(f"[{group}] target data missing, skipping group: {data_path}")
            continue
        x_data = torch.from_numpy(np.load(data_path))
        n_features = x_data.shape[1]
        feature_names = [f"feature {i+1}" for i in range(n_features)]
        print(f"[{group}] target {os.path.basename(data_path)} "
              f"{tuple(x_data.shape)}, bins={bins}")

        config_dirs = sorted(glob.glob(os.path.join(group_dir, "*ensemblecomponents*")))
        for config_dir in config_dirs:
            name = os.path.basename(config_dir)
            m = re.search(r"ensemblecomponents(\d+)", name)
            n_components = int(m.group(1)) if m else None
            loaded = _load_config(config_dir, n_features)
            if loaded is None:
                print(f"  [{name}] no complete choices, skipped")
                continue
            curves, grids, ferrs, n_fits = loaded
            out = plot_ensemble_marginals_aggregate(
                curves, grids, ferrs, x_data, feature_names, config_dir,
                bins=bins, n_components=n_components, n_fits=n_fits,
                truth_fn=(cfg["truth_fn"] if args.show_truth else None),
                dataset_label=cfg["label"], show_fit2fit=args.show_fit2fit,
            )
            print(f"  [{name}] M={n_components}, {n_fits} fits -> {os.path.relpath(out, args.results_root)}")
            made += 1

    print(f"\nDone. {made} aggregate figure(s) written.")


if __name__ == "__main__":
    main()
