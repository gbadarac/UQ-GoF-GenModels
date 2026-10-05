import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import os

import mplhep as hep

# Use CMS style for plots
hep.style.use("CMS")

# ── Shared marginal-plot style (group-standard format) ────────────────────────
# ink = axis/title text, muted = annotations, grey = data points, blue = model+band.
INK, MUTED, GREY, BLUE = "#33322e", "#8a897f", "#9a9a9a", "#2a78d6"


def _draw_marginal_feature(ax_main, ax_ratio, model_x, f_binned, f_err, valid_curve,
                           data_x, data_y, data_err, feature_label,
                           leftmost=False, show_legend=False):
    """One feature's density (top) + band/mean ratio (bottom) in the shared group
    format: grey filled-histogram target, solid-blue ensemble model f(x)=Σ wᵢfᵢ, one
    blue 1σ band; ink axis text, muted annotations. y-labels only on the leftmost
    column, legend only where show_legend. Lower panel = the 1σ band relative to the
    mean (band/mean = relative uncertainty; no ground-truth ratio, 4D has no truth)."""
    band_lo, band_hi = f_binned - f_err, f_binned + f_err

    # density panel
    ax_main.fill_between(data_x, data_y, step='mid', color=GREY, alpha=0.35, lw=0,
                         zorder=1, label="target")
    ax_main.fill_between(model_x, band_lo, band_hi, color=BLUE, alpha=0.30, lw=0,
                         zorder=2, label=r"$\pm 1\sigma$")
    ax_main.plot(model_x[valid_curve], f_binned[valid_curve], '-', color=BLUE,
                 lw=1.8, zorder=3, label=r"$f(x)=\sum_i w_i f_i(x)$")
    if leftmost:
        ax_main.set_ylabel("density", color=INK, fontsize=17)
    if show_legend:
        ax_main.legend(frameon=False, fontsize=13, labelcolor=INK)
    ax_main.tick_params(colors=INK)

    # ratio panel: band / mean (relative uncertainty; no ground-truth ratio, 4D has none)
    f_safe = np.where(f_binned > 0, f_binned, np.nan)
    r_lo, r_hi = band_lo / f_safe, band_hi / f_safe
    good = ~np.isnan(r_hi)
    ax_ratio.fill_between(model_x[good], r_lo[good], r_hi[good], color=BLUE, alpha=0.30, lw=0)
    ax_ratio.axhline(1.0, color=MUTED, ls=':', lw=1)
    ax_ratio.set_ylim(0.9, 1.1)
    ax_ratio.set_xlabel(feature_label, color=INK, fontsize=17)
    if leftmost:
        ax_ratio.set_ylabel("band / mean", color=INK, fontsize=14)
    ax_ratio.tick_params(colors=INK)
    ax_ratio.grid(True, ls='--', lw=0.5, alpha=0.5)


def probs(weights, model_probs):
    """
    Args:
        weights: torch tensor of shape (M,) with requires_grad=True
        model_probs: torch tensor of shape (N, M), where each column is f_i(x) for all x
    Returns:
        p(x) ≈ ∑ w_i * f_i(x) for each x (shape N,)
    """
    return (model_probs * weights).sum(dim=1)  # shape: (N,)

def log_likelihood(weights, model_probs):
    p_x = probs(weights, model_probs) + 1e-8  # prevent log(0)
    ll = torch.log(p_x).mean() #averaging the loss over all datapoints 
    return ll 

def ensemble_pred(weights, model_probs):
    weights = weights.to(model_probs.device)
    model_vals = (model_probs * weights).sum(dim=1)
    return model_vals.cpu().numpy()

def ensemble_unc(cov_w, model_probs):
    # cov_w is (M-1, M-1); propagate through ∂f/∂u_j = f_j - f_M
    if hasattr(model_probs, 'cpu'):
        model_probs_np = model_probs.cpu().clone().numpy()
    else:
        model_probs_np = np.asarray(model_probs)
    if hasattr(cov_w, 'numpy'):
        cov_w = cov_w.numpy()
    g = model_probs_np[:, :-1] - model_probs_np[:, -1:]   # (N, M-1)
    sigma_sq = np.einsum('ni,ij,nj->n', g, cov_w, g)
    return np.sqrt(np.maximum(sigma_sq, 0.0))

def plot_ensemble_marginals_2d(f_i_models, x_data, weights, cov_w, feature_names, outdir,
                               bins=60, n_curve=300, n_components=None,
                               estimator_label="Normalizing Flow", dataset_label="2D toy model"):
    """All feature marginals in ONE figure (2 rows x n_features cols: density on top,
    band/mean ratio below), with a title stating the density-estimator type and the
    number of ensemble components.

    The green target HISTOGRAM (``bins``) and the red model curve + uncertainty band
    (evaluated on a dense ``n_curve`` grid) are DECOUPLED: the model marginal is drawn
    smoothly regardless of the histogram binning, so wiggles in the red line are real
    density structure, not an artifact of coarse bins. Per-feature marginal_feature_*.npz
    (keys f_binned/f_err/bin_centers, now on the dense grid) are still written for the
    downstream LRT / NPLM consumers.
    """
    x = x_data.cpu().numpy()
    num_features = x.shape[1]

    fig, axes = plt.subplots(
        2, num_features, figsize=(6 * num_features, 9),
        gridspec_kw={'height_ratios': [3, 1]}, squeeze=False,
    )

    for i in range(num_features):
        ax_main, ax_ratio = axes[0, i], axes[1, i]
        feature_label = feature_names[i]
        input_feature = x[:, i]

        # Bin over full data range with margin
        margin = 0.05 * (np.max(input_feature) - np.min(input_feature))
        low, high = np.min(input_feature) - margin, np.max(input_feature) + margin

        # --- target histogram (thin bins, decoupled from the model curve) ---
        bin_edges = np.linspace(low, high, bins + 1)
        bin_centers_hist = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)
        hist_target_counts, _ = np.histogram(input_feature, bins=bin_edges)
        N_target = np.sum(hist_target_counts)
        hist_target = hist_target_counts / (N_target * bin_widths)
        err_target = np.sqrt(hist_target_counts) / (N_target * bin_widths)

        # ------------------------------------------
        # TRUE 1D marginal of each ensemble member (2D case), on a DENSE grid
        # ------------------------------------------
        # A normalizing flow's marginal has no closed form, so we marginalise the
        # joint density NUMERICALLY: for every eval point c of feature i we integrate
        #     f_m^marg(c) = ∫ f_m(x_i = c, x_j) dx_j
        # over the OTHER feature x_j on a fine grid. The eval points x_eval are a dense
        # (n_curve) grid independent of the histogram bins -> smooth red curve + band.
        j = 1 - i                                    # the other feature (2D only)
        xj = x[:, j]
        margin_j = 0.05 * (xj.max() - xj.min())
        n_int = 300
        xj_grid = np.linspace(xj.min() - margin_j, xj.max() + margin_j, n_int)
        dxj = xj_grid[1] - xj_grid[0]

        x_eval = np.linspace(low, high, n_curve)
        dx_eval = x_eval[1] - x_eval[0]

        # (n_curve, n_int, 2): every eval point of feature i paired with every x_j node
        CC, GG = np.meshgrid(x_eval, xj_grid, indexing="ij")          # (n_curve, n_int)
        grid = np.empty((n_curve, n_int, 2), dtype=np.float32)
        grid[:, :, i] = CC
        grid[:, :, j] = GG
        grid_flat = torch.from_numpy(grid.reshape(n_curve * n_int, 2)).float().to(
            next(f_i_models[0].parameters()).device
        )

        with torch.no_grad():
            probs_grid = torch.stack(
                [torch.exp(flow.log_prob(grid_flat)) for flow in f_i_models],
                dim=1  # (n_curve*n_int, M)
            )

        # Integrate out x_j (Riemann sum) -> per-member marginal at each eval point
        probs_marg = probs_grid.view(n_curve, n_int, -1).sum(dim=1) * dxj   # (n_curve, M)

        f_binned = ensemble_pred(weights, probs_marg)          # (n_curve,)
        f_err = ensemble_unc(cov_w, probs_marg)                # (n_curve,)

        # Members are already ~unit-normalised; renormalise defensively for the
        # finite grid so the plotted marginal integrates to 1 over the eval range.
        N = np.sum(f_binned * dx_eval)
        f_binned /= N
        f_err /= N

        # ------------------
        # Save marginals as npz (dense grid; keys unchanged for downstream consumers)
        # ------------------
        out_marginal = os.path.join(outdir, f"marginal_feature_{i+1}_data.npz")
        np.savez_compressed(out_marginal,
                            f_binned=f_binned,
                            f_err=f_err,
                            bin_centers=x_eval)

        # Model curve only across the data support (where the target has events)
        valid_curve = (x_eval >= input_feature.min()) & (x_eval <= input_feature.max())

        _draw_marginal_feature(
            ax_main, ax_ratio, x_eval, f_binned, f_err, valid_curve,
            bin_centers_hist, hist_target, err_target, feature_label,
            leftmost=(i == 0), show_legend=(i == 0),
        )

    # Figure title (group format): estimator + ensemble size, ink color. The 10-fit
    # AGGREGATE plotter adds a muted "mean over 10 ensemble fits" subtitle; a single
    # per-fit plot stays single-line (it is one ensemble choice).
    title = f"{dataset_label} marginals for {estimator_label}"
    if n_components is not None:
        comp_word = "component" if n_components == 1 else "components"
        title += f" ({n_components} ensemble {comp_word})"
    fig.suptitle(title, color=INK, fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = os.path.join(outdir, "ensemble_marginals.png")
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)

def plot_ensemble_marginals_nd(f_i_models, x_data, weights, cov_w, feature_names, outdir,
                            bins=40, n_samples=30000, device=None,
                            estimator_label="Normalizing Flow", dataset_label=None,
                            n_components=None):
    """1D marginals of the WiFi NF ensemble for ANY dimensionality D >= 2.

    The marginal of feature j is computed EXACTLY via the linearity of the mixture:
        p_marg(x_j) = int sum_i w_i f_i(x) dx_{-j} = sum_i w_i * m_i(x_j)
    where m_i(x_j) = int f_i dx_{-j} is member i's own marginal. Each member is a
    proper normalized flow, so m_i is obtained UNBIASED by sampling that member and
    histogramming feature j (exactly what the training-side sample plot does).

    This replaces the old "data-proposal" estimator (hold the other features at data
    values and average the joint density), which actually computes
    int f(x_j, x_-j) p_data(x_-j) dx_-j -- equal to the true marginal ONLY when x_j is
    independent of the other features. That bias is invisible for a product density
    (the 2D toy) but severe once the features are correlated (the 4D embedding), where
    it produces a spurious spike at the mode.

    Reduces to the exact single-member sample marginal for M = 1 (band collapses to 0).
    """
    x = x_data.cpu().numpy()
    N, D = x.shape
    weights = weights.detach().cpu().double()
    cov_w = torch.from_numpy(cov_w).double()

    # Sample each member ONCE (reused across all D features). The flows are passed on
    # CPU; move each to the sampling device (GPU if available) and back, since
    # autoregressive sampling is slow on CPU. Reproducible per member.
    dev = torch.device(device) if device is not None else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    member_samples = []
    with torch.no_grad():
        for j, flow in enumerate(f_i_models):
            torch.manual_seed(1234 + j)
            flow.to(dev)
            member_samples.append(flow.sample(n_samples).cpu().numpy())  # (n_samples, D)
            flow.to("cpu")

    fig, axes = plt.subplots(2, D, figsize=(6 * D, 9),
                             gridspec_kw={'height_ratios': [3, 1]}, squeeze=False)
    for i in range(D):
        ax_main, ax_ratio = axes[0, i], axes[1, i]
        feature_label = feature_names[i]
        xi = x[:, i]

        # binning on the data support
        margin = 0.05 * (xi.max() - xi.min())
        low, high = xi.min() - margin, xi.max() + margin
        bin_edges = np.linspace(low, high, bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = np.diff(bin_edges)

        # target histogram + errors
        hist_counts, _ = np.histogram(xi, bins=bin_edges)
        N_target = hist_counts.sum()
        hist_target = hist_counts / (N_target * bin_widths)
        err_target = np.sqrt(hist_counts) / (N_target * bin_widths)

        # ---- Exact per-member marginal via sampling (unbiased under correlation) ----
        # v_mat[b, j] = m_j(bin_center_b): member j's marginal density of feature i, from
        # that member's own samples. Normalized by the TOTAL sample count (samples outside
        # the plot range are dropped, so each column integrates to ~1 over the shown
        # support); the weighted sum f_binned is renormalized to unit area just below.
        B = len(bin_centers)
        M = len(f_i_models)
        v_mat = torch.empty((B, M), dtype=torch.float64)
        for j, s in enumerate(member_samples):
            counts, _ = np.histogram(s[:, i], bins=bin_edges)
            v_mat[:, j] = torch.from_numpy(counts / (len(s) * bin_widths)).double()

        # Ensemble mean and uncertainty at each center
        w_col = weights.view(-1, 1)                           # (M,1)
        f_binned = (v_mat @ weights).numpy()                  # (B,)
        # sigma^2(c) = g(c)^T Cov_u g(c),  g_j = v_j - v_M  (M-1 free weights)
        g_mat = v_mat[:, :-1] - v_mat[:, -1:]                # (B, M-1)
        sigma2 = (g_mat @ cov_w @ g_mat.T).diagonal().numpy()
        f_err = np.sqrt(np.maximum(sigma2, 0.0))              # (B,)

        # Normalize to unit area over the scan dimension
        area = np.sum(f_binned * bin_widths)
        if area > 0:
            f_binned /= area
            f_err    /= area

        print(f"[feat {i}] f_binned min/max:", f_binned.min(), f_binned.max())
        print(f"[feat {i}] f_err min/max:", f_err.min(), f_err.max())

        # save npz
        out_marginal = os.path.join(outdir, f"marginal_feature_{i+1}_data.npz")
        np.savez_compressed(out_marginal, f_binned=f_binned, f_err=f_err, bin_centers=bin_centers)

        valid_bins = hist_target > 0
        _draw_marginal_feature(
            ax_main, ax_ratio, bin_centers, f_binned, f_err, valid_bins,
            bin_centers, hist_target, err_target, feature_label,
            leftmost=(i == 0), show_legend=(i == 0),
        )

    # Figure title (group format), ink color. The 10-fit AGGREGATE plotter adds a
    # muted "mean over 10 ensemble fits" subtitle; a single per-fit plot stays
    # single-line (it is one ensemble choice).
    ds_label = dataset_label if dataset_label is not None else f"{D}D"
    title = f"{ds_label} marginals for {estimator_label}"
    if n_components is not None:
        comp_word = "component" if n_components == 1 else "components"
        title += f" ({n_components} ensemble {comp_word})"
    fig.suptitle(title, color=INK, fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = os.path.join(outdir, "ensemble_marginals.png")
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


# canonical palette (shared with the p-value / test-statistic plot):
#   BLUE  #2a78d6 = Gaussian composite band   RED #c0392b = replicate/bootstrap composite
#   GREY  #9a9a9a = plug-in / fitted-model line   GRID #c9c8c0 = light reference lines + spines
#   MUTED #8a897f = annotation text            INK  #33322e = axis labels / titles / data
ORANGE = "#eb6834"
RED = "#c0392b"
GRID = "#c9c8c0"
# opaque band tints (NO alpha — overlapping alpha bands wash out to grey); Sean's paper_style
BLUE_BAND = "#b3cdf0"   # Gaussian uncertainty band
RED_BAND = "#f0b9b1"    # fit-to-fit band (diagnostic overlay only)


def _fmt_N(n):
    """LaTeX event count for legends: 100000 -> '10^{5}', 1900000 -> '1.9\\times10^{6}'."""
    import math
    n = int(n)
    if n <= 0:
        return str(n)
    e = int(round(math.log10(n)))
    if abs(n - 10 ** e) <= 0.01 * 10 ** e:
        return f"10^{{{e}}}"
    e = int(math.floor(math.log10(n)))
    return f"{n / 10 ** e:.1f}\\times10^{{{e}}}"


def _draw_marginal_feature_aggregate(ax_main, ax_ratio, grid, median, band_lo, band_hi,
                                     wifi_lo, wifi_hi, data_x, data_y, data_err,
                                     truth, feature_label, leftmost=False, show_legend=False,
                                     ratio_ylim=(0.8, 1.2), show_fit2fit=False,
                                     show_gaussian=True, data_label="held-out data",
                                     fit_label=r"fitted model  $f(x)=\sum_i w_i f_i(x)$"):
    """One feature's WiFi-NF density (top) + ratio-to-fit (bottom), in Sean's paper style
    (paper_style_bundle/plot_marginals_paper.py):
      * held-out data -> black 'x' markers with Poisson error bars,
      * fitted model  -> MUTED grey solid line = the WiFi ensemble median f(x)=sum_i w_i f_i(x),
      * "Gaussian uncertainty (68%)" -> opaque BLUE_BAND fill = the weight-covariance Sigma_w
        propagated to the density (median +- mean f_err over the fits); the density analogue of
        the LRT's N(w_hat, Sigma_w) weight prior, and the ONLY uncertainty an NF actually
        propagates into the test. `show_gaussian=False` (single model) -> no band at all.
      * `show_fit2fit` (DIAGNOSTIC, off in the paper) adds the fit-to-fit 25-75% spread across
        the 10 choices as an opaque RED_BAND — NOT propagable into the test; kept only to show
        it's ~= the Gaussian band in 2D but >> it in 4D.
    Bands are opaque tints (no alpha); wider (red) drawn first, Gaussian (blue) on top.
    Bottom panel: everything / the fitted model ("ratio to fit").
    """
    ref = median                                     # ratio reference: the fitted model
    ref_at_data = np.interp(data_x, grid, ref)
    # trim to where the fit is well-defined (signed-weight median dips ~0 in far tails ->
    # ratio blows up); mask data to the same x-range so band + points end together.
    thr = 1e-3 * np.nanmax(ref)
    ok = ref > thr
    xlo, xhi = float(grid[ok].min()), float(grid[ok].max())
    dm = (data_x >= xlo) & (data_x <= xhi) & (ref_at_data > thr)

    # ---- top panel ----
    if show_fit2fit:
        ax_main.fill_between(grid, band_lo, band_hi, color=RED_BAND, lw=0, zorder=1,
                             label="fit-to-fit (25-75%)")
    if show_gaussian:
        ax_main.fill_between(grid, wifi_lo, wifi_hi, color=BLUE_BAND, lw=0, zorder=2,
                             label="Gaussian uncertainty (68%)")
    if truth is not None:
        ax_main.plot(grid, truth, color=INK, lw=1.2, zorder=3, label="ground truth")
    ax_main.plot(grid, median, color=MUTED, lw=1.0, ls="-", zorder=4, label=fit_label)
    ax_main.errorbar(data_x[dm], data_y[dm], yerr=data_err[dm], fmt="x", ms=4, mew=1.0,
                     color=INK, ecolor=INK, elinewidth=0.9, capsize=0, zorder=5,
                     label=data_label)
    if leftmost:
        ax_main.set_ylabel("density", color=INK, fontsize=12)
    if show_legend:
        h, l = ax_main.get_legend_handles_labels()
        pref = [data_label, fit_label, "Gaussian uncertainty (68%)",
                "fit-to-fit (25-75%)", "ground truth"]
        idx = [l.index(p) for p in pref if p in l]
        ax_main.legend([h[i] for i in idx], [l[i] for i in idx],
                       frameon=False, fontsize=9, labelcolor=INK, loc="best")

    ax_main.set_xlim(xlo, xhi)          # sharex='col' -> trims the ratio panel too

    # ---- ratio panel: everything / the fitted model ----
    # `ax_ratio is None` -> no ratio panel at all (the single-model plots: there is no band
    # and no fit-to-fit spread, so the residual view adds nothing). The feature label then
    # moves up onto the density panel.
    if ax_ratio is None:
        ax_main.set_xlabel(feature_label, color=INK, fontsize=13)
        return
    ax_ratio.axhline(1.0, color=MUTED, ls="-", lw=1.0, zorder=4)   # = fitted model
    if show_fit2fit:
        ax_ratio.fill_between(grid[ok], band_lo[ok] / ref[ok], band_hi[ok] / ref[ok],
                              color=RED_BAND, lw=0, zorder=1)
    if show_gaussian:
        ax_ratio.fill_between(grid[ok], wifi_lo[ok] / ref[ok], wifi_hi[ok] / ref[ok],
                              color=BLUE_BAND, lw=0, zorder=2)
    if truth is not None:
        ax_ratio.plot(grid[ok], truth[ok] / ref[ok], color=INK, lw=1.0, zorder=3)
    ax_ratio.errorbar(data_x[dm], data_y[dm] / ref_at_data[dm],
                      yerr=data_err[dm] / ref_at_data[dm], fmt="x", ms=4, mew=1.0,
                      color=INK, ecolor=INK, elinewidth=0.9, capsize=0, zorder=5)
    ax_ratio.set_ylim(*ratio_ylim)
    ax_ratio.set_xlabel(feature_label, color=INK, fontsize=13)
    if leftmost:
        ax_ratio.set_ylabel("ratio to fit", color=INK, fontsize=12)


def plot_ensemble_marginals_aggregate(curves_per_feature, grids_per_feature,
                                      ferr_per_feature, x_data, feature_names, outdir,
                                      bins=40, n_components=None, n_fits=None, truth_fn=None,
                                      estimator_label="Normalizing Flow", dataset_label=None,
                                      qs=(25.0, 50.0, 75.0), ratio_ylim=(0.8, 1.2),
                                      show_fit2fit=False, show_gaussian=True, n_train=None):
    """Aggregate the per-fit ensemble marginals of the 10 ensemble CHOICES into ONE
    figure per configuration (paper template: data points + median + propagated band).

    Central line = MEDIAN over the fits; the primary BLUE band = the WiFi weight-cov
    +-1sigma band PROPAGATED INTO THE TEST (median +- mean f_err over the fits; f_err =
    ensemble_unc(Sigma_w), the density-level version of the LRT's N(w_hat, Sigma_w)
    weight prior) -> the plot is a coverage check that the target sits inside the
    uncertainty the test uses. With ``show_fit2fit`` the fit-to-fit 25-75% IQR across
    the fits is overlaid as a grey dashed envelope (the p-value plot's convention; a
    DIFFERENT quantity = experiment-to-experiment spread of the point estimate).

    Args:
        curves_per_feature: list (len = n_features) of (n_fits, n_curve) arrays, the
            stacked per-fit model marginals f(x)=sum_i w_i m_i on the shared grid.
        grids_per_feature: list of (n_curve,) grids (one per feature; shared by fits).
        ferr_per_feature:  list of (n_fits, n_curve) per-fit WiFi cov bands (f_err).
        x_data: (N, D) target sample -> binned points with Poisson error bars.
        truth_fn: optional callable(feature_index, grid)->analytic truth density; when
            None (e.g. 4D embedding) the ratio panel is taken w.r.t. the median fit.
        qs: the (low, mid, high) percentiles; defaults to the 25/50/75 house standard.
    """
    # serif fonts to match the paper body + Sean's p-value plots. MUST come AFTER mplhep's CMS
    # style (imported at module load) so serif wins. Ref: paper_style_bundle/paper_style.py.
    # Reset label locations to center too (CMS puts them at the axis ends).
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                         "xaxis.labellocation": "center", "yaxis.labellocation": "center"})

    x = x_data.cpu().numpy() if hasattr(x_data, "cpu") else np.asarray(x_data)
    num_features = len(curves_per_feature)
    q_lo, q_mid, q_hi = qs
    data_label = f"held-out data ($N={_fmt_N(x.shape[0])}$)"

    # per-column title, Sean's compact format: "$N_{\rm train}=100k$  (...)". single model (M=1)
    # has no ensemble/weights -> label as a single model.
    single = (n_components == 1)
    title_bits = []
    if n_train is not None:
        title_bits.append(rf"$N_{{\rm train}}={int(n_train) // 1000}k$")
    tag = ("single NF model" if single
           else r"$w_i f_i$ NF ensemble" + (rf", $M={n_components}$" if n_components is not None else ""))
    col_title = (title_bits[0] + f"  ({tag})") if title_bits else f"({tag})"
    fit_label = (r"fitted model  $f(x)=f_i(x)$" if single
                 else r"fitted model  $f(x)=\sum_i w_i f_i(x)$")

    # 2D -> one row of (marginal,ratio) columns (the two-panel template); 4D -> Sean's 2x2 grid
    # of (marginal,ratio) pairs (paper_style_bundle/example_marginal_grid.py). The grid is the
    # SAME WIDTH as the 2D figure, so both render at the same font size at a fixed paper-column
    # width -- a 1x4 strip would get squeezed ~2x more and come out with smaller fonts.
    # Single-model plots (M=1) carry no band and no fit-to-fit spread, so the residual view adds
    # nothing -> drop the ratio panel entirely for those (show_ratio=False): a density-only figure.
    grid_layout = (num_features == 4)
    show_ratio = not single

    def _style_ax(ax):            # light-grey thin spines + inward ticks (not the heavy CMS default)
        for sp in ax.spines.values():
            sp.set_color(GRID)
            sp.set_linewidth(0.9)
        ax.tick_params(direction="in", which="both", colors=INK, top=True, right=True)
        ax.minorticks_off()

    if grid_layout:
        from matplotlib.ticker import MaxNLocator
        if show_ratio:
            fig = plt.figure(figsize=(10, 11.4))
            # row index 2 (height 0.62) is a spacer between the two (marginal,ratio) row-pairs
            gs = fig.add_gridspec(5, 2, height_ratios=[3, 1, 0.62, 3, 1], hspace=0.05, wspace=0.25)
            main_axes, ratio_axes = [], []
            for d in range(num_features):
                r, c = 3 * (d // 2), d % 2
                ax = fig.add_subplot(gs[r, c])
                rx = fig.add_subplot(gs[r + 1, c], sharex=ax)
                main_axes.append(ax)
                ratio_axes.append(rx)
        else:
            # density-only 2x2 grid, same 10-inch width as the ratio version
            fig = plt.figure(figsize=(10, 8.8))
            gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)
            main_axes = [fig.add_subplot(gs[d // 2, d % 2]) for d in range(num_features)]
            ratio_axes = [None] * num_features
    else:
        nrows = 2 if show_ratio else 1
        fig, axes = plt.subplots(
            nrows, num_features, figsize=(5 * num_features, 5.4 if show_ratio else 4.0),
            sharex='col',
            gridspec_kw=({'height_ratios': [3, 1], 'hspace': 0.05} if show_ratio else None),
            squeeze=False,
        )
        main_axes = [axes[0, i] for i in range(num_features)]
        ratio_axes = ([axes[1, i] for i in range(num_features)] if show_ratio
                      else [None] * num_features)

    for ax in main_axes + [a for a in ratio_axes if a is not None]:
        _style_ax(ax)

    for i in range(num_features):
        ax_main, ax_ratio = main_axes[i], ratio_axes[i]
        curves = np.asarray(curves_per_feature[i])        # (n_fits, n_curve)
        grid = np.asarray(grids_per_feature[i])           # (n_curve,)

        lo = np.percentile(curves, q_lo, axis=0)
        mid = np.percentile(curves, q_mid, axis=0)
        hi = np.percentile(curves, q_hi, axis=0)

        # typical single-fit WiFi +-1sigma band = mean f_err across fits, about the median
        mean_ferr = np.asarray(ferr_per_feature[i]).mean(axis=0)
        wifi_lo, wifi_hi = mid - mean_ferr, mid + mean_ferr

        # target sample as binned points with Poisson errors
        input_feature = x[:, i]
        margin = 0.05 * (np.max(input_feature) - np.min(input_feature))
        low, high = np.min(input_feature) - margin, np.max(input_feature) + margin
        bin_edges = np.linspace(low, high, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)
        counts, _ = np.histogram(input_feature, bins=bin_edges)
        N_tot = np.sum(counts)
        data_y = counts / (N_tot * bin_widths)
        data_err = np.sqrt(counts) / (N_tot * bin_widths)

        truth = truth_fn(i, grid) if truth_fn is not None else None

        _draw_marginal_feature_aggregate(
            ax_main, ax_ratio, grid, mid, lo, hi, wifi_lo, wifi_hi,
            bin_centers, data_y, data_err, truth, feature_names[i],
            # grid: every pair is its own column pair -> y-labels on all (Sean's template);
            # strip: only the leftmost column carries the y-labels.
            leftmost=(True if grid_layout else (i == 0)),
            show_legend=(i == 0), ratio_ylim=ratio_ylim,
            show_fit2fit=show_fit2fit, show_gaussian=show_gaussian,
            data_label=data_label, fit_label=fit_label,
        )
        if grid_layout:
            # sparse ticks on the half-width panels; hide the density-panel x labels only when a
            # ratio panel sits below it (otherwise the density panel is its column's bottom row).
            (ax_ratio if show_ratio else ax_main).xaxis.set_major_locator(MaxNLocator(5))
            if show_ratio:
                ax_main.tick_params(labelbottom=False)         # x labels only on the ratio panel
        else:
            ax_main.set_title(col_title, color=INK, fontsize=10.5, pad=6)

    if grid_layout:
        fig.suptitle(col_title, color=INK, fontsize=10.5,
                     y=(0.905 if show_ratio else 0.95))        # one title for the 2x2 grid
    else:
        fig.tight_layout(h_pad=0.3)
    base = os.path.join(outdir, "ensemble_marginals_aggregate")
    for ext in ("png", "pdf"):                       # png for quick viewing, pdf (vector) for the paper
        fig.savefig(f"{base}.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base + ".png"


def plot_ensemble_marginals_M_grid(configs, x_data, feature_names, outdir,
                                   bins=40, dataset_label=None, n_train=None,
                                   qs=(25.0, 50.0, 75.0), ratio_ylim=(0.8, 1.2),
                                   fname="marginals_M_grid"):
    """Compress the per-M marginal figures into ONE comparison panel (Gaia's request):
    ROWS = feature marginals, COLUMNS = ensemble size M (e.g. 1, 32, 128). Each cell is a
    (density, ratio-to-fit) pair: held-out data (black 'x') + that config's fitted-median
    marginal (grey) + Gaussian 68% band (blue). So the whole M-progression of one marginal
    reads across a row at a shared per-row density + x scale.

    The SINGLE-MODEL column (M=1) has no weight covariance -> no band -> the ratio panel
    carries no information, so its density is the SAME size as the others and the ratio row
    below it is simply left blank (white space).

    This is a TRANSPOSE of the per-config aggregate (which fixes M and shows all features):
    here M runs along the columns so the comparison is M=1 -> M=128 for a fixed marginal.

    configs: list of (M, curves_per_feature, grids_per_feature, ferr_per_feature), any order
        (sorted here by M). curves_per_feature[i] = (n_fits, n_curve) stacked over the 10 fits;
        grids_per_feature[i] = (n_curve,); ferr_per_feature[i] = (n_fits, n_curve) WiFi cov band.
        Exactly the per-config arrays plot_ensemble_marginals_aggregate already consumes.
    Writes <outdir>/<fname>.{png,pdf}; returns the PNG path.
    """
    from matplotlib.ticker import MaxNLocator
    # serif to match the paper body + Sean's plots; AFTER mplhep's CMS import so serif wins.
    # Also reset label locations to CENTER: mplhep's CMS style puts axis labels at the axis
    # ends (xaxis->right, yaxis->top), which would push x_i to the right and "ratio" to the top.
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                         "xaxis.labellocation": "center", "yaxis.labellocation": "center"})

    x = x_data.cpu().numpy() if hasattr(x_data, "cpu") else np.asarray(x_data)
    n_features = len(feature_names)
    configs = sorted(configs, key=lambda c: c[0])        # columns ordered by M
    n_cols = len(configs)
    _, q_mid, _ = qs
    data_label = f"held-out data ($N={_fmt_N(x.shape[0])}$)"

    def _style(ax):                                      # light-grey thin spines + inward ticks
        for sp in ax.spines.values():
            sp.set_color(GRID)
            sp.set_linewidth(0.9)
        # length=3 shortens the ticks (mplhep's CMS style makes them long ~10pt)
        ax.tick_params(direction="in", which="both", colors=INK, top=True, right=True,
                       labelsize=7.5, length=3)
        ax.minorticks_off()

    # each feature = a (density[h=3], ratio[h=1]) row-pair; a thin spacer row between pairs
    height_ratios = []
    for i in range(n_features):
        height_ratios += [3, 1]
        if i < n_features - 1:
            height_ratios += [0.55]
    fig = plt.figure(figsize=(2.75 * n_cols, 2.95 * n_features))
    gs = fig.add_gridspec(len(height_ratios), n_cols, height_ratios=height_ratios,
                          hspace=0.06, wspace=0.08)

    top_left_density = None                              # panel above the legend slot
    for i in range(n_features):                          # row = one feature's marginal
        rd, rr = 3 * i, 3 * i + 1                         # density row, ratio row
        # target histogram for this feature (shared across all M columns)
        xi = x[:, i]
        margin = 0.05 * (xi.max() - xi.min())
        low, high = xi.min() - margin, xi.max() + margin
        edges = np.linspace(low, high, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        counts, _ = np.histogram(xi, bins=edges)
        N_tot = counts.sum()
        data_y = counts / (N_tot * widths)
        data_err = np.sqrt(counts) / (N_tot * widths)

        # per-column median + band, and the common per-row x/y limits
        cols, xlo_row, xhi_row, ymax_row = [], np.inf, -np.inf, float(data_y.max())
        for (M, cpf, gpf, fpf) in configs:
            grid = np.asarray(gpf[i])
            mid = np.percentile(np.asarray(cpf[i]), q_mid, axis=0)
            ferr = np.asarray(fpf[i]).mean(axis=0)
            thr = 1e-3 * np.nanmax(mid)
            ok = mid > thr
            xlo_row = min(xlo_row, float(grid[ok].min()))
            xhi_row = max(xhi_row, float(grid[ok].max()))
            ymax_row = max(ymax_row, float(mid[ok].max()))
            cols.append((M, grid, mid, ferr, ok))

        # leftmost column with a ratio panel carries the "ratio to fit" y-label (M=1 has none)
        first_ratio_j = next((j for j, c in enumerate(cols) if c[0] > 1), None)

        for j, (M, grid, mid, ferr, ok) in enumerate(cols):
            has_ratio = (M > 1)                          # single model: no band -> no ratio panel
            axd = fig.add_subplot(gs[rd, j])             # SAME-size density in every column
            if has_ratio:
                axr = fig.add_subplot(gs[rr, j], sharex=axd)
                _style(axr)
            else:
                axr = None                               # M=1: leave the ratio row below blank
            _style(axd)
            if i == 0 and j == 0:
                top_left_density = axd                    # sits above the legend slot

            # ---- density panel ----
            if has_ratio:
                axd.fill_between(grid[ok], (mid - ferr)[ok], (mid + ferr)[ok],
                                 color=BLUE_BAND, lw=0, zorder=2,
                                 label="Gaussian uncertainty (68%)")
            axd.plot(grid[ok], mid[ok], color=MUTED, lw=1.0, ls="-", zorder=4,
                     label="fitted model")
            axd.errorbar(centers, data_y, yerr=data_err, fmt="x", ms=3, mew=0.8,
                         color=INK, ecolor=INK, elinewidth=0.7, capsize=0, zorder=5,
                         label=data_label)
            axd.set_xlim(xlo_row, xhi_row)
            axd.set_ylim(0, 1.25 * ymax_row)
            axd.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
            axd.yaxis.set_major_locator(MaxNLocator(nbins=4))
            if i == 0:
                axd.set_title(rf"$M={M}$", color=INK, fontsize=10.5, pad=4)
            if j == 0:
                axd.set_ylabel("density", color=INK, fontsize=10)
            else:
                axd.tick_params(labelleft=False)
            if has_ratio:                                # x labels live on the ratio panel below
                axd.tick_params(labelbottom=False)

            # ---- ratio-to-fit panel (everything / the fitted median) ----
            if has_ratio:
                ref = mid
                ref_at = np.interp(centers, grid, ref)
                dm = (centers >= xlo_row) & (centers <= xhi_row) & (ref_at > 1e-3 * np.nanmax(ref))
                axr.axhline(1.0, color=MUTED, ls="-", lw=1.0, zorder=4)
                axr.fill_between(grid[ok], ((mid - ferr) / ref)[ok], ((mid + ferr) / ref)[ok],
                                 color=BLUE_BAND, lw=0, zorder=2)
                axr.errorbar(centers[dm], (data_y / ref_at)[dm], yerr=(data_err / ref_at)[dm],
                             fmt="x", ms=3, mew=0.8, color=INK, ecolor=INK, elinewidth=0.7,
                             capsize=0, zorder=5)
                axr.set_xlim(xlo_row, xhi_row)
                axr.set_ylim(*ratio_ylim)
                axr.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
                axr.yaxis.set_major_locator(MaxNLocator(nbins=3))
                if j == first_ratio_j:                   # leftmost ratio carries the y-label/ticks
                    axr.set_ylabel("ratio", color=INK, fontsize=8, labelpad=1)
                else:
                    axr.tick_params(labelleft=False)

            # x-axis label identifying the feature, on the leftmost RATIO panel's bottom
            # (perpendicular to its "ratio" y-label -> no collision with the M=1 column). If no
            # ratio column exists at all, fall back to the leftmost density.
            if (has_ratio and j == first_ratio_j) or (first_ratio_j is None and j == 0):
                bottom_ax = axr if has_ratio else axd
                bottom_ax.tick_params(axis="x", pad=1.5)       # pull the tick numbers in
                bottom_ax.set_xlabel(rf"$x_{{{i + 1}}}$", color=INK, fontsize=11, labelpad=1)

    # legend in the blank white slot below the top-left (M=1) density: gs[1,0] is the ratio row
    # of the first feature's first column, which is empty when that column is M=1. This never
    # overlaps a curve (unlike an in-panel legend on the bimodal). Handles from a density cell
    # that has a band so all three entries show.
    h, l = [], []
    for a in fig.axes:
        hh, ll = a.get_legend_handles_labels()
        if "Gaussian uncertainty (68%)" in ll:
            h, l = hh, ll
            break
    order = [l.index(p) for p in (data_label, "fitted model", "Gaussian uncertainty (68%)")
             if p in l]
    if order and configs[0][0] == 1:                     # first column is M=1 -> gs[1,0] is blank
        if top_left_density is not None:                 # its x-tick numbers hang into the slot
            top_left_density.tick_params(axis="x", labelbottom=False)
        legend_ax = fig.add_subplot(gs[1, 0])
        legend_ax.set_axis_off()
        legend_ax.legend([h[k] for k in order], [l[k] for k in order], frameon=False,
                         fontsize=8, labelcolor=INK, loc="center",
                         handlelength=1.4, handletextpad=0.5, borderaxespad=0.2)
    elif order:                                          # no blank slot -> top strip fallback
        fig.legend([h[k] for k in order], [l[k] for k in order], frameon=False, fontsize=9,
                   labelcolor=INK, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.945))

    m_list = ", ".join(str(c[0]) for c in configs)
    ntr = rf",  $N_{{\rm train}}={int(n_train) // 1000}k$" if n_train is not None else ""
    # fixed ABSOLUTE top margin (inches), not a fraction: a fraction leaves ~2x the title->plots
    # gap on the taller 4D figure (4 rows) vs 2D (2 rows). Anchored to the 2D spacing.
    figh = fig.get_figheight()
    fig.suptitle(rf"$M = {m_list}$" + ntr, color=INK, fontsize=12, y=1 - 0.24 / figh)
    fig.subplots_adjust(top=1 - 0.65 / figh, left=0.07, right=0.985)
    base = os.path.join(outdir, fname)
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base + ".png"
