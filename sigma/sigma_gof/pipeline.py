"""
SIGMA goodness-of-fit pipeline: fit a Gaussian mixture, build its
Fisher-whitened tangent block plus a dictionary of J Gaussian kernels, and
calibrate the composite and point profile-likelihood-ratio statistics with a
Gaussian null, a parametric-bootstrap null (cold EM refits), or the point
(plug-in) null.

Stages (each fit b lives in $SIGMA_RUNS/<run>/m{b:02d}/, default <snapshot root>/runs):

  fit        train data for fit b; EM fit at fixed K; reference set R0 from
             the fit; Fisher whitening W (with the LAM_MAX cut); kernel
             dictionary; saves model.npz + model_info.json.
  gaussian   null bank for one N_test: composite toys drawn from the exact
             mixture at theta_hat + W z, z ~ N(0, I_M) (the Gaussian
             approximation to the fit's sampling law), point toys from
             theta_hat. Writes {tag}_bank.npz.
  bootstrap  parametric-bootstrap null bank for several N_test layers from
             ONE refit per toy: draw a fit-sized sample from p_theta_hat,
             refit cold (best of SIGMA_BOOT_NINIT k-means++ EM starts),
             rebuild R0 / whitening / dictionary at the refit, draw the
             toy's test data from p_theta_hat and evaluate the composite
             statistic with the refit's own features. The point bank uses
             the real fit's features. Writes {tag}_bank.npz per layer (and
             per extra Lambda arm, see SIGMA_BOOT_LAMS).
  observed   t and p for every observed set of (b, N_test) against the
             bank in the run dir; with SIGMA_OBS_TONLY=1 only t (no bank
             needed).
  fill       attach a (complete) bank to a t-only observed file and write
             the normal observed file.

Usage:
  python pipeline.py fit       --b 0 [--s-ref N] [--smoke]
  python pipeline.py gaussian  --b 0 --nte 25000 [--toys 250] [--chunk c --nchunks n]
  python pipeline.py bootstrap --b 0 --ntes 1000 10000 100000 [--toys 250] [--chunk c --nchunks n]
  python pipeline.py observed  --b 0 --nte 25000
  python pipeline.py fill      --b 0 --nte 25000 [--src <t-only file or run dir>]

Run directory: runs/<bench>_K{K}[_Ntr{n}k]{SIGMA_SUFFIX}, e.g. toy2d_K64.
The gaussian and bootstrap banks share the file name {tag}_bank.npz, so give
each null its own SIGMA_SUFFIX (with the fit's model.npz linked into it).

Environment (defaults in parentheses):
  SIGMA_BENCH         toy2d | jets4d (toy2d)
  SIGMA_K             number of mixture components (required for `fit`)
  SIGMA_NTRAIN        training-sample size (100000); other values add _Ntr{n}k
  SIGMA_SUFFIX        free run-name suffix ("")
  SIGMA_J             number of dictionary kernels (64)
  SIGMA_KERNEL_SIGMA  > 0: all kernels isotropic with this width in raw data
                      units; 0: two widths 0.25 / 0.60 x median pairwise
                      distance in standardized coordinates (0)
  SIGMA_LAM_MAX       whitening cut on the covariance eigenvalues (0.008)
  SIGMA_NINIT         k-means++ EM starts of the real fit (1)
  SIGMA_BOOT_NINIT    k-means++ EM starts of every bootstrap refit; must equal
                      the fit's n_init (1)
  SIGMA_BOOT_LAMS     extra Lambda arms of the bootstrap: "lam:suffix,..." ("")
  SIGMA_NSETS         observed sets per (fit, N_test) (100)
  SIGMA_OBS_FRESH     toy2d: 1 = observed sets seeded per fit (distinct across
                      fits); 0 = the same sets for every fit (0)
  SIGMA_OBS_TONLY     observed stage writes t only (0)
  SIGMA_S_FISHER      reference rows used for the Fisher matrix (1000000)
  SIGMA_NZ_MIN        floor on the normalisation-slice size (0)
  SIGMA_BANK_DTYPE    float64 | float32 feature bank (float64)
  SIGMA_EM_TOL, SIGMA_REG_COVAR (gmm.py), SIGMA_CHUNK (tangent.py, teststat.py),
  SIGMA_JETS_DATA (jets4d.py)
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from toy2d import generate_2d_gmm_skew
from gmm import UGMM, fit_em, mle_stationarity, EM_TOL, REG_COVAR
from tangent import (fisher_whiten, rebuild_whitening, build_dictionary,
                     raw_width_dictionary, FeatureMap, _chunks)
from teststat import t_statistics, p_addone

PROJECT_ROOT = os.path.dirname(THIS_DIR)            # the snapshot root (holds sigma_gof/, runs/, data/)
RUNS = os.environ.get("SIGMA_RUNS", os.path.join(PROJECT_ROOT, "runs"))
# The mixture capacity is fixed a priori (no selection rule).
FIXED_K = int(os.environ["SIGMA_K"]) if os.environ.get("SIGMA_K") else None
N_TRAIN = int(os.environ.get("SIGMA_NTRAIN", "100000"))
# k-means++ EM starts for the real fit (best of n). Bootstrap refits must use
# the same estimator: SIGMA_BOOT_NINIT must equal it (check_estimator).
N_INIT = int(os.environ.get("SIGMA_NINIT", "1"))
BOOT_NINIT = int(os.environ.get("SIGMA_BOOT_NINIT", "1"))
BENCH = os.environ.get("SIGMA_BENCH", "toy2d").lower()
if BENCH not in ("toy2d", "jets4d"):
    raise SystemExit(f"SIGMA_BENCH must be toy2d or jets4d, not {BENCH!r}")
_SUFFIX = ("" if FIXED_K is None else f"_K{FIXED_K}") + \
          ("" if N_TRAIN == 100_000 else f"_Ntr{N_TRAIN // 1000}k") + \
          os.environ.get("SIGMA_SUFFIX", "")
_BASE = BENCH
OUT_ROOT = os.path.join(RUNS, _BASE + _SUFFIX)

# ── constants ────────────────────────────────────────────────────────
N_BANK = 200_000            # toy2d observed sets are prefixes of draws of this size
# observed sets per (fit, N_test)
N_SETS = int(os.environ.get("SIGMA_NSETS", "100"))
MASTER = 779                # seed base of toys and toy2d observed sets
SEED_MODEL_BASE = 900_000   # + b: training sample / EM seed of fit b
SEED_R0_BASE = 950_000      # + b: reference set R0 of fit b

S_REF = 2_000_000           # reference-set size, >= 10 x largest N_test
# The Fisher matrix that defines the tangent space and the lambda cut is
# estimated on the FIRST S_FISHER rows of R0, while the normalisation slices
# stay at 10 x N_test (slice_size). Kernel standardisation, clip bounds and
# the dictionary use the full R0.
S_FISHER = int(os.environ.get("SIGMA_S_FISHER", "1000000"))
J_KERNELS = int(os.environ.get("SIGMA_J", "64"))
# Linearization radius. The Gaussian toys draw delta_theta = W z, so
# sqrt(lambda_i) IS the 1-sigma excursion along direction i in raw theta
# units, and the cut must bound that excursion: the tangent expansion's
# validity is set by the CURVATURE of log p in theta, which is a property of
# the model, not of the sample size. So this is an ABSOLUTE cap on lambda.
#
# Note that lambda_i = 1/(N_fit * f_i) with f_i the per-event Fisher
# information, so at fixed model every excursion shrinks like 1/sqrt(N_fit)
# and large N_fit is inherently safer. Directions that stay wide come from
# near-degeneracies of the mixture (very small f_i); the driver is model
# redundancy, and N_fit enters only indirectly, via K growing with more data.
# Do NOT rescale this with N_fit: 1/(N_fit * f_min) has the scaling
# backwards — it tightens the cut exactly where excursions are already
# smallest.
#
# Value: a loose cut (0.8, excursions ~0.9 in theta units against parameters
# of order 1) destroys the Gaussian null through linearization inflation.
# The null sits on a stable plateau for LAM_MAX from 0.05 down to 0.001
# while the number of kept directions M falls, i.e. the dropped directions
# are pure artifact; 0.008 (max excursion 0.089) is the geometric centre of
# that plateau. Re-run such a scan for any new model/benchmark.
LAM_MAX = float(os.environ.get("SIGMA_LAM_MAX", "0.008"))
# > 0: one kernel width for all J kernels, isotropic in RAW data units,
# instead of the two widths 0.25 / 0.60 x q50 isotropic in standardized
# coordinates; 0 = off. Centers stay the k-means centers of the standard
# construction (mapped back to raw coordinates). Applied wherever a
# dictionary is built (real fits and every bootstrap refit).
KERNEL_SIGMA = float(os.environ.get("SIGMA_KERNEL_SIGMA", "0"))
LAM_ALPHA = 1.0
# Sup-norm clip on the kernel tilt. A=1 binds on part of the kernel
# coordinates at large N_test, where it distorts t_point and makes the
# box-constrained solve much more expensive. A=10 does not bind in practice
# and reproduces the L2-only optimum, so it is a pure runaway guard for
# pathological directions rather than an active regularizer.
CLIP_A = 10.0
N_TOYS = 250
# Normalisation slice: every stage estimates Z(beta) (and the solver's
# Hessian) on the first slice_size(nte) rows of R0, S = Z_OVERSAMPLE x
# N_test: residual MC noise ~ N_test/S is a fixed 10% at every N_test, the
# same slice serves the observed data and every toy so it largely cancels in
# the rank comparison, and the per-toy solve cost scales with S.
# SIGMA_NZ_MIN (default 0 = off) is an opt-in floor: at very small N_test
# the slice can be smaller than M+J, so n*Cov_S(B) is rank-deficient and only
# the ridge keeps the solve well-posed (toys self-calibrate either way).
Z_OVERSAMPLE = 10
N_Z_MIN = int(os.environ.get("SIGMA_NZ_MIN", "0"))
BANK_DTYPE = os.environ.get("SIGMA_BANK_DTYPE", "float64").lower()
# Seed slot of the null banks and the toys between partial saves.
BANK_SEED_SLOT = 0
BANK_CKPT = 10


def slice_size(nte, n_ref):
    return min(int(n_ref), max(Z_OVERSAMPLE * int(nte), N_Z_MIN))


def nte_tag(nte):
    """File tag for an N_test value: 'nte025k' for multiples of 1000,
    'nte100' otherwise."""
    nte = int(nte)
    return f"nte{nte // 1000:03d}k" if nte % 1000 == 0 else f"nte{nte}"


def model_dir(b):
    return os.path.join(OUT_ROOT, f"m{b:02d}")


def train_data(b, n):
    """Training sample of fit b."""
    if BENCH == "jets4d":
        import jets4d
        return jets4d.train_data(b, n)
    return generate_2d_gmm_skew(n, seed=SEED_MODEL_BASE + b)


def observed_data(s, nte, b, fresh):
    """Observed set s at N_test = nte for fit b. toy2d: the first nte events
    of a draw of max(N_BANK, nte) (the generator is not prefix-stable, so sets
    up to N_BANK are prefixes of one draw and larger N_test get a full-size
    draw); with fresh=True the seed also depends on b, so sets are distinct
    across fits. jets4d: see jets4d.observed_data."""
    if BENCH == "jets4d":
        import jets4d
        return jets4d.observed_data(s, nte, b)
    return generate_2d_gmm_skew(
        max(N_BANK, int(nte)),
        seed=MASTER * 1_000 + s + (7_000_000 * (b + 1) if fresh else 0))[:nte]


def load_model(b):
    z = np.load(os.path.join(model_dir(b), "model.npz"))
    model = UGMM(int(z["K"]), int(z["d"]))
    return model, z


def check_estimator(z):
    """Bootstrap refits must use the real fit's estimator: refuse to run if
    SIGMA_EM_TOL / SIGMA_REG_COVAR / SIGMA_BOOT_NINIT differ from what the
    model was fitted with."""
    bad = []
    if float(z["em_tol"]) != EM_TOL:
        bad.append(f"em_tol={float(z['em_tol'])} vs SIGMA_EM_TOL={EM_TOL}")
    if float(z["reg_covar"]) != REG_COVAR:
        bad.append(f"reg_covar={float(z['reg_covar'])} vs SIGMA_REG_COVAR={REG_COVAR}")
    if int(z["n_init"]) != BOOT_NINIT:
        bad.append(f"n_init={int(z['n_init'])} vs SIGMA_BOOT_NINIT={BOOT_NINIT}")
    if bad:
        raise SystemExit("estimator mismatch (bootstrap refits must use the "
                         "real fit's estimator): " + "; ".join(bad))


def rebuild_featuremap(b, model, z):
    """R0 and the FeatureMap are deterministic functions of the stored
    seeds/params — recomputed per task instead of storing a multi-GB
    B_R0. Kernel standardization and clip bounds come from model.npz."""
    rng = np.random.default_rng(int(z["seed_r0"]))
    R0 = model.sample(int(z["s_ref"]), z["theta"], rng)
    dic = {k: z[f"dic_{k}"] for k in
           ("centers", "sigmas", "x_mean", "x_std")}
    dic["q50"] = float(z["dic_q50"])
    fm = FeatureMap(model, z["theta"], z["W"], dic, R0,
                    clip_A=float(z["clip_A"]),
                    moments=(z["k_mean"], z["k_std"], z["alpha_box"]))
    return fm, R0


def _tangent_objects(model, theta, n_train, s_ref, J, seed_r0):
    """Everything the test needs from a fitted (model, theta): reference
    set R0, Fisher whitening W (with the LAM_MAX cut), kernel dictionary
    and the FeatureMap with its standardisation/clip moments, all measured
    on R0 (a sample from the fit)."""
    rng = np.random.default_rng(seed_r0)
    R0 = model.sample(s_ref, theta, rng)
    W, wdiag, eig = fisher_whiten(model, theta, R0[:S_FISHER], n_train,
                                  lam_max=LAM_MAX, return_eig=True)
    dic = build_dictionary(R0, J, seed=seed_r0)
    if KERNEL_SIGMA > 0:
        dic = raw_width_dictionary(dic, KERNEL_SIGMA)
    fm = FeatureMap(model, theta, W, dic, R0, clip_A=CLIP_A)
    return R0, W, wdiag, eig, dic, fm


def _build_bank_slice(fm, R0, n_z):
    """B(R0[:n_z]) in the bank dtype. f32 (SIGMA_BANK_DTYPE=float32) halves
    memory and speeds the slice matvecs ~1.4x; the gradient inherits an
    f32 noise floor ~n*1e-7, which teststat's dtype-aware gtol accounts
    for. The chunks are written straight into the f32 matrix."""
    dtype = np.float64 if BANK_DTYPE == "float64" else np.float32
    return fm.B(R0[:n_z], dtype=dtype)


# ── fit stage ────────────────────────────────────────────────────────
def run_fit(b, smoke=False, s_ref=None, j_kernels=None, force=False):
    if os.path.exists(os.path.join(model_dir(b), "model.npz")) and not force:
        print(f"[fit m{b:02d}] model.npz exists, skipping "
              f"(deterministic refit would be identical; --force to redo)")
        return
    if FIXED_K is None:
        raise SystemExit("fit needs SIGMA_K (the number of mixture components)")
    t0 = time.time()
    n_train = 5_000 if smoke else N_TRAIN
    s_ref = s_ref or (100_000 if smoke else S_REF)
    J = j_kernels or (16 if smoke else J_KERNELS)

    X = train_data(b, n_train)
    K = FIXED_K
    print(f"[fit m{b:02d}] train {X.shape}, K = {K}")
    model, theta = fit_em(X, K, seed=SEED_MODEL_BASE + b, n_init=N_INIT)

    psi_max, psi_z = mle_stationarity(model, theta, X)
    print(f"  final fit: P = {model.P}, train ll/ev = "
          f"{model.log_density(X, theta).mean():.5f}")
    print(f"  MLE stationarity: max|mean score| = {psi_max:.3e} "
          f"({psi_z:.2f} SEM)")
    # theta_hat MUST be (close to) a stationary point — the tangent space,
    # the Fisher covariance and the centering of the composite null all
    # assume it.
    if psi_z > 5.0:
        raise RuntimeError(
            f"EM did not reach the MLE: mean training score is {psi_z:.1f} "
            f"SEM from zero (max |mean| = {psi_max:.3e}). Tighten SIGMA_EM_TOL.")

    R0, W, wdiag, (fish_f, fish_V), dic, fm = _tangent_objects(
        model, theta, n_train, s_ref, J, SEED_R0_BASE + b)
    print(f"  whitening (Fisher on {min(S_FISHER, len(R0))} ref events): "
          f"M = {wdiag['M']} / P = {wdiag['P']} ({wdiag['n_dropped']} dropped)")

    os.makedirs(model_dir(b), exist_ok=True)
    np.savez(os.path.join(model_dir(b), "model.npz"),
             K=K, d=X.shape[1], theta=theta, W=W,
             fisher_f=fish_f, fisher_V=fish_V, n_train=n_train,
             seed_r0=SEED_R0_BASE + b, s_ref=s_ref, clip_A=CLIP_A,
             s_fisher=min(S_FISHER, len(R0)), lam_max=LAM_MAX,
             k_mean=fm.k_mean, k_std=fm.k_std, alpha_box=fm.alpha_box,
             dic_centers=dic["centers"], dic_sigmas=dic["sigmas"],
             dic_x_mean=dic["x_mean"], dic_x_std=dic["x_std"],
             dic_q50=dic["q50"], em_tol=EM_TOL, reg_covar=REG_COVAR,
             n_init=N_INIT)
    with open(os.path.join(model_dir(b), "model_info.json"), "w") as f:
        json.dump({"b": b, "K": K, **wdiag,
                   "n_train": n_train, "s_ref": s_ref,
                   "s_fisher": min(S_FISHER, len(R0)), "J": J,
                   "kernel_sigma": KERNEL_SIGMA, "lam_max": LAM_MAX,
                   "q50": dic["q50"], "mle_score_max": psi_max,
                   "mle_score_z": psi_z, "em_tol": EM_TOL,
                   "reg_covar": REG_COVAR, "n_init": N_INIT,
                   "walltime_s": time.time() - t0}, f, indent=2)
    print(f"  saved. ({time.time() - t0:.0f}s)")


# ── null banks ───────────────────────────────────────────────────────
def bank_path(b, nte, chunk=None, nchunks=1):
    tag = nte_tag(nte)
    if chunk is None or nchunks == 1:
        return os.path.join(model_dir(b), f"{tag}_bank.npz")
    return os.path.join(model_dir(b), f"{tag}_bank_c{chunk:02d}of{nchunks:02d}.npz")


def load_bank(b, nte):
    """(bank_comp, bank_point, source): the single bank file, or a complete
    set of chunk files {tag}_bank_cXXofNN.npz (--chunk/--nchunks: contiguous
    toy ranges, concatenated in order)."""
    path = bank_path(b, nte)
    if os.path.exists(path):
        z = np.load(path)
        return z["bank_comp"], z["bank_point"], os.path.basename(path)
    chunks = sorted(glob.glob(os.path.join(model_dir(b),
                                           f"{nte_tag(nte)}_bank_c*of*.npz")))
    chunks = [c for c in chunks if "_partial" not in c]
    if chunks:
        n = int(chunks[0].rsplit("of", 1)[1].split(".")[0])
        if len(chunks) != n:
            raise SystemExit(f"m{b:02d} nte{nte}: {len(chunks)}/{n} bank "
                             f"chunks present — wait for the remaining chunks")
        zs = [np.load(c) for c in chunks]
        return (np.concatenate([z["bank_comp"] for z in zs]),
                np.concatenate([z["bank_point"] for z in zs]),
                f"{n} chunks")
    raise SystemExit(f"no bank for m{b:02d} nte{nte}: run the gaussian or "
                     f"bootstrap stage first")


def _chunk_range(toys, chunk, nchunks):
    if chunk is not None and nchunks > 1:
        return chunk * toys // nchunks, (chunk + 1) * toys // nchunks
    return 0, toys


def _gaussian_toy_pair(i, b, nte, model, theta, W, fm, B_R0, M):
    """One composite toy + one point toy at index i. The composite toy is
    drawn from the exact mixture at theta_hat + W z, z ~ N(0, I_M); the
    point toy from theta_hat. Returns (t_comp, t_point, n_unconverged)."""
    rng = np.random.default_rng([MASTER, b, nte, BANK_SEED_SLOT, i, 1])
    zb = rng.standard_normal(M)
    Xb = model.sample(nte, theta + W @ zb, rng)
    rc = t_statistics(B_R0, fm.B_sum(Xb), nte, M, fm.alpha_box, LAM_ALPHA,
                      which=("comp",))
    rng = np.random.default_rng([MASTER, b, nte, BANK_SEED_SLOT, i, 2])
    Xp = model.sample(nte, theta, rng)
    rp = t_statistics(B_R0, fm.B_sum(Xp), nte, M, fm.alpha_box, LAM_ALPHA,
                      which=("point",))
    n_unconv = (0 if rc["conv_comp"] else 1) + (0 if rp["conv_point"] else 1)
    return rc["t_comp"], rp["t_point"], n_unconv


def run_gaussian(b, nte, toys=N_TOYS, force=False, chunk=None, nchunks=1):
    """Gaussian null bank of (b, nte). chunk/nchunks: compute only toys
    [chunk*toys/nchunks, (chunk+1)*toys/nchunks) into a chunk file (toys
    are independent by seed, so chunks are exact and parallel); load_bank
    concatenates them."""
    out_path = bank_path(b, nte, chunk, nchunks)
    i_lo, i_hi = _chunk_range(toys, chunk, nchunks)
    if os.path.exists(out_path) and not force:
        print(f"[gaussian m{b:02d} nte{nte}] exists, skipping")
        return
    t0 = time.time()
    model, z = load_model(b)
    theta, W = z["theta"], z["W"]
    M = W.shape[1]
    fm, R0 = rebuild_featuremap(b, model, z)
    n_z = slice_size(nte, len(R0))
    print(f"[gaussian m{b:02d} nte{nte}] building B_R0 slice "
          f"({n_z} x {M + fm.J}) ...", flush=True)
    B_R0 = _build_bank_slice(fm, R0, n_z)
    del R0
    print(f"  slice built ({time.time() - t0:.0f}s)", flush=True)

    partial = out_path.replace(".npz", "_partial.npz")
    bank_comp, bank_point, n_unconv = [], [], 0
    if os.path.exists(partial) and not force:
        zp = np.load(partial)
        if int(zp["toys"]) == toys and int(zp["n_z"]) == n_z and \
                int(zp["i_lo"]) == i_lo:
            bank_comp = list(zp["bank_comp"])
            bank_point = list(zp["bank_point"])
            n_unconv = int(zp["n_unconverged"])
            print(f"  resuming from {len(bank_comp)} toys", flush=True)

    def save(path):
        tmp = path + f".tmp{os.getpid()}"
        np.savez(tmp, bank_comp=np.asarray(bank_comp),
                 bank_point=np.asarray(bank_point), toys=toys, n_z=n_z,
                 M=M, J=fm.J, K=int(z["K"]), b=b, nte=nte,
                 toy_mode="gaussian", seed_slot=BANK_SEED_SLOT,
                 i_lo=i_lo, i_hi=i_hi,
                 chunk=-1 if chunk is None else chunk, nchunks=nchunks,
                 n_unconverged=n_unconv, walltime_s=time.time() - t0)
        os.replace(tmp + ".npz", path)

    for i in range(i_lo + len(bank_comp), i_hi):
        tc, tp, nu = _gaussian_toy_pair(i, b, nte, model, theta, W, fm, B_R0, M)
        bank_comp.append(tc)
        bank_point.append(tp)
        n_unconv += nu
        if (i + 1 - i_lo) % BANK_CKPT == 0 or i + 1 == i_hi:
            save(partial)
            print(f"  {i + 1 - i_lo}/{i_hi - i_lo} toys: bank means "
                  f"{np.mean(bank_comp):.1f}/{np.mean(bank_point):.1f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    save(out_path)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"  done ({time.time() - t0:.0f}s, {n_unconv} unconverged)")


def bootstrap_refit(model, theta, n_train, b, i):
    """Bootstrap toy i of fit b: a fit-sized sample from p_theta_hat, refit
    cold with the real fit's estimator (best of BOOT_NINIT k-means++ EM
    starts). Returns (X_i, theta_i)."""
    rng = np.random.default_rng([MASTER, b, 0, BANK_SEED_SLOT, i, 3])
    Xi = model.sample(n_train, theta, rng)
    seed_i = SEED_MODEL_BASE + b + 1_000_000 * (i + 1) + 500_000
    _, theta_i = fit_em(Xi, model.K, seed=seed_i, n_init=BOOT_NINIT)
    return Xi, theta_i


def _arm_dir(suf, b):
    """Run dir m{b} of an extra Lambda arm: this run's name with SIGMA_SUFFIX
    replaced by the arm's suffix."""
    pre = _SUFFIX[:len(_SUFFIX) - len(os.environ.get("SIGMA_SUFFIX", ""))]
    return os.path.join(RUNS, _BASE + pre + suf, f"m{b:02d}")


def run_bootstrap(b, ntes, toys=N_TOYS, force=False, chunk=None, nchunks=1):
    """Parametric-bootstrap null bank for SEVERAL N_test layers from ONE
    refit per toy. The refit + re-whitening dominate a toy and do not depend
    on N_test, and the bank slices are prefix-nested (B(R0[:n_z])), so one
    refit serves every layer at ~the cost of one. Toy i's fit sample is
    shared across layers (the layers' banks are correlated toy by toy,
    harmless for per-layer calibration); the toy test data are seeded per
    layer.

    SIGMA_BOOT_LAMS="lam:suffix,...": every toy's refit also serves extra
    LAM_MAX arms — its whitening is rebuilt from the toy's own Fisher
    eigendecomposition at each lam (tangent.rebuild_whitening) and only the
    tangent columns of its bank slice are recomputed (kernel block, refit
    and toy test data are shared, so the arms are paired toy by toy). Arm
    banks are written to runs/<bench>_K{K}...<suffix>/m{b}/ with the same
    file names; their point bank is the primary's (t_point does not depend
    on Lambda). SIGMA_LAM_MAX must be the Lambda the fit was whitened at."""
    lam_arms = [(float(a.split(":")[0]), a.split(":")[1])
                for a in os.environ.get("SIGMA_BOOT_LAMS", "").split(",") if a.strip()]
    ntes = sorted(int(n) for n in ntes)
    i_lo, i_hi = _chunk_range(toys, chunk, nchunks)
    outs = {nte: bank_path(b, nte, chunk, nchunks) for nte in ntes}
    todo = [nte for nte in ntes if force or not os.path.exists(outs[nte])]
    if not todo:
        print(f"[bootstrap m{b:02d}] all layers exist, skipping")
        return
    t0 = time.time()
    model, z = load_model(b)
    check_estimator(z)
    if "lam_max" in z and abs(float(z["lam_max"]) - LAM_MAX) > 1e-12 * LAM_MAX:
        raise SystemExit(f"model.npz was whitened at Lambda {float(z['lam_max'])} "
                         f"but SIGMA_LAM_MAX = {LAM_MAX}")
    theta = z["theta"]
    n_train = int(z["n_train"])
    s_ref, J = int(z["s_ref"]), int(len(z["dic_sigmas"]))
    n_zs = {nte: slice_size(nte, s_ref) for nte in todo}
    n_z_max = max(n_zs.values())
    # point null: the real fit's features and slice. The plug-in test
    # neglects the fit's scatter by construction, so its null must not
    # contain it; only the composite bank is bootstrap-based.
    fm0, R00 = rebuild_featuremap(b, model, z)
    B0 = _build_bank_slice(fm0, R00, n_z_max)
    del R00
    M0 = z["W"].shape[1]
    ctag = -1 if chunk is None else chunk
    partial = os.path.join(model_dir(b),
                           f"bootstrap_c{ctag:02d}of{nchunks:02d}_partial.npz")
    res = {nte: {"comp": [], "point": []} for nte in todo}
    res_x = {suf: {nte: [] for nte in todo} for _, suf in lam_arms}   # extra arms: comp only
    Ms_x = {suf: [] for _, suf in lam_arms}
    guards, Ms, n_unconv = [], [], 0
    if os.path.exists(partial) and not force:
        zp = np.load(partial)
        if (int(zp["toys"]) == toys and int(zp["i_lo"]) == i_lo
                and list(int(n) for n in zp["ntes"]) == todo
                and [str(s) for s in (zp["lam_sufs"] if "lam_sufs" in zp.files else [])]
                == [s for _, s in lam_arms]):
            for k, nte in enumerate(todo):
                res[nte]["comp"] = list(zp["bank_comp"][k])
                res[nte]["point"] = list(zp["bank_point"][k])
            for a, (_, suf) in enumerate(lam_arms):
                for k, nte in enumerate(todo):
                    res_x[suf][nte] = list(zp["bank_comp_x"][a, k])
                Ms_x[suf] = list(zp["M_toy_x"][a])
            guards, Ms = list(zp["guard_sem"]), list(zp["M_toy"])
            n_unconv = int(zp["n_unconverged"])
            print(f"  resuming from {len(guards)} toys", flush=True)

    def save_partial():
        tmp = partial + f".tmp{os.getpid()}"
        x = dict(lam_sufs=np.array([s for _, s in lam_arms], dtype=str),
                 bank_comp_x=np.array([[res_x[s][n] for n in todo] for _, s in lam_arms]),
                 M_toy_x=np.array([Ms_x[s] for _, s in lam_arms])) if lam_arms else {}
        np.savez(tmp, bank_comp=np.array([res[n]["comp"] for n in todo]),
                 bank_point=np.array([res[n]["point"] for n in todo]),
                 ntes=np.array(todo), toys=toys, i_lo=i_lo,
                 guard_sem=np.asarray(guards), M_toy=np.asarray(Ms),
                 n_unconverged=n_unconv, **x)
        os.replace(tmp + ".npz", partial)

    print(f"[bootstrap m{b:02d}] layers {todo}, toys {i_lo}..{i_hi - 1}, "
          f"n_z_max {n_z_max}", flush=True)
    for i in range(i_lo + len(guards), i_hi):
        ti = time.time()
        Xi, theta_i = bootstrap_refit(model, theta, n_train, b, i)
        print(f"  toy {i}: cold EM (best of {BOOT_NINIT} inits) "
              f"done in {time.time() - ti:.0f}s; train ll "
              f"{model.log_density(Xi, theta_i).mean():.5f} vs theta_hat "
              f"{model.log_density(Xi, theta).mean():.5f}", flush=True)
        psi_max, psi_z = mle_stationarity(model, theta_i, Xi)
        R0_i, W_i, wd_i, eig_i, dic_i, fm_i = _tangent_objects(
            model, theta_i, n_train, s_ref, J,
            int(z["seed_r0"]) + 1_000_000 * (i + 1))
        B_i = _build_bank_slice(fm_i, R0_i, n_z_max)
        M_i = W_i.shape[1]
        # extra Lambda arms: same refit / kernel block, tangent columns from the rebuilt whitening
        arm_feats = []
        for lam, suf in lam_arms:
            W_l, _ = rebuild_whitening(eig_i[0], eig_i[1], n_train, lam)
            fm_l = FeatureMap(model, theta_i, W_l, dic_i, None, clip_A=CLIP_A,
                              moments=(fm_i.k_mean, fm_i.k_std, fm_i.alpha_box))
            B_l = np.empty((n_z_max, W_l.shape[1] + B_i.shape[1] - M_i), dtype=B_i.dtype)
            for lo, hi in _chunks(n_z_max):
                B_l[lo:hi, :W_l.shape[1]] = model.scores(R0_i[lo:hi], theta_i) @ W_l
            B_l[:, W_l.shape[1]:] = B_i[:, M_i:]
            arm_feats.append((suf, fm_l, B_l))
        del R0_i
        for nte in todo:
            n_z = n_zs[nte]
            rng_t = np.random.default_rng([MASTER, b, nte, BANK_SEED_SLOT, i, 4])
            Yi = model.sample(nte, theta, rng_t)
            r = t_statistics(B_i[:n_z], fm_i.B_sum(Yi), nte, M_i,
                             fm_i.alpha_box, LAM_ALPHA, which=("comp",))
            for suf, fm_l, B_l in arm_feats:
                rl = t_statistics(B_l[:n_z], fm_l.B_sum(Yi), nte, fm_l.M,
                                  fm_l.alpha_box, LAM_ALPHA, which=("comp",))
                res_x[suf][nte].append(rl["t_comp"])
                n_unconv += 0 if rl["conv_comp"] else 1
            rng_p = np.random.default_rng([MASTER, b, nte, BANK_SEED_SLOT, i, 2])
            rp = t_statistics(B0[:n_z], fm0.B_sum(model.sample(nte, theta, rng_p)),
                              nte, M0, fm0.alpha_box, LAM_ALPHA, which=("point",))
            res[nte]["comp"].append(r["t_comp"])
            res[nte]["point"].append(rp["t_point"])
            n_unconv += (0 if r["conv_comp"] else 1) + (0 if rp["conv_point"] else 1)
        for suf, fm_l, _ in arm_feats:
            Ms_x[suf].append(fm_l.M)
        del arm_feats
        guards.append(psi_z)
        Ms.append(M_i)
        save_partial()
        print(f"  {i + 1 - i_lo}/{i_hi - i_lo} bootstrap toys x {len(todo)} "
              f"layers: comp means " +
              "/".join(f"{np.mean(res[n]['comp']):.1f}" for n in todo) +
              f"  (M_toy {M_i}, stationarity {psi_z:.2f} SEM, "
              f"{time.time() - ti:.0f} s/toy, {time.time() - t0:.0f}s)",
              flush=True)
    for nte in todo:
        tmp = outs[nte] + f".tmp{os.getpid()}"
        np.savez(tmp, bank_comp=np.asarray(res[nte]["comp"]),
                 bank_point=np.asarray(res[nte]["point"]),
                 toys=toys, n_z=n_zs[nte], M=M0, J=J, K=int(z["K"]), b=b,
                 nte=nte, toy_mode="bootstrap",
                 seed_slot=BANK_SEED_SLOT, boot_n_init=BOOT_NINIT,
                 i_lo=i_lo, i_hi=i_hi, chunk=ctag, nchunks=nchunks,
                 guard_sem=np.asarray(guards), M_toy=np.asarray(Ms),
                 n_unconverged=n_unconv, walltime_s=time.time() - t0)
        os.replace(tmp + ".npz", outs[nte])
        for lam, suf in lam_arms:
            out_x = os.path.join(_arm_dir(suf, b), os.path.basename(outs[nte]))
            os.makedirs(os.path.dirname(out_x), exist_ok=True)
            tmp = out_x + f".tmp{os.getpid()}"
            np.savez(tmp, bank_comp=np.asarray(res_x[suf][nte]),
                     bank_point=np.asarray(res[nte]["point"]),
                     toys=toys, n_z=n_zs[nte], J=J, K=int(z["K"]), b=b, nte=nte,
                     lam_max=lam, toy_mode="bootstrap_lambda_arm",
                     seed_slot=BANK_SEED_SLOT, boot_n_init=BOOT_NINIT,
                     i_lo=i_lo, i_hi=i_hi, chunk=ctag, nchunks=nchunks,
                     guard_sem=np.asarray(guards), M_toy=np.asarray(Ms_x[suf]),
                     walltime_s=time.time() - t0)
            os.replace(tmp + ".npz", out_x)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"  done ({time.time() - t0:.0f}s, {n_unconv} unconverged)")


# ── observed sets ────────────────────────────────────────────────────
def _obs_name(nte):
    fresh = os.environ.get("SIGMA_OBS_FRESH", "0") == "1"
    return f"{nte_tag(nte)}_obs{'_fresh' if fresh else ''}", fresh


def run_observed(b, nte, force=False):
    """t and p for every observed set of (b, nte) against the bank in this
    run dir. Writes {tag}_obs[_fresh].json (per-set t, p and solver
    diagnostics; the bank stored once at top level).

    SIGMA_OBS_TONLY=1: observed t only, no bank needed -> {...}_tonly.json,
    so this stage can run while the banks are still being built; `fill`
    later attaches a bank and writes the normal file. The t values depend
    only on the model, featuremap and observed sets, so one t-only file
    serves every null arm sharing them."""
    name, fresh = _obs_name(nte)
    tonly = os.environ.get("SIGMA_OBS_TONLY", "0") == "1"
    out_path = os.path.join(model_dir(b), name + ("_tonly" if tonly else "") + ".json")
    if os.path.exists(out_path) and not force:
        print(f"[observed m{b:02d} nte{nte}] exists, skipping")
        return
    t0 = time.time()
    bank_comp, bank_point, src = (np.zeros(0), np.zeros(0), None) if tonly else load_bank(b, nte)
    model, z = load_model(b)
    W = z["W"]
    M = W.shape[1]
    fm, R0 = rebuild_featuremap(b, model, z)
    n_z = slice_size(nte, len(R0))
    print(f"[observed m{b:02d} nte{nte}] bank {src} ({len(bank_comp)} toys); "
          f"building B_R0 slice ({n_z} x {M + fm.J}) ...", flush=True)
    B_R0 = _build_bank_slice(fm, R0, n_z)
    del R0
    obs, n_unconv = [], 0
    for s in range(N_SETS):
        Xs = observed_data(s, nte, b, fresh)
        r = t_statistics(B_R0, fm.B_sum(Xs), nte, M, fm.alpha_box, LAM_ALPHA)
        r["set"] = s
        r["p_comp"] = None if tonly else p_addone(r["t_comp"], bank_comp)
        r["p_point"] = None if tonly else p_addone(r["t_point"], bank_point)
        n_unconv += (0 if r["conv_comp"] else 1) + \
                    (0 if r["conv_point"] else 1)
        obs.append(r)
        print(f"  set {s}: t_comp={r['t_comp']:.2f} (p={r['p_comp']})"
              f"  t_point={r['t_point']:.2f} (p={r['p_point']})"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    result = {
        "b": b, "nte": nte, "toys": int(len(bank_comp)), "M": M, "J": fm.J,
        "K": int(z["K"]), "n_z": n_z, "bench": BENCH,
        "obs_mode": ("overlap" if BENCH == "jets4d"
                     else ("fresh" if fresh else "shared")),
        "bank_source": src,
        "bank_comp": [float(t) for t in bank_comp],
        "bank_point": [float(t) for t in bank_point],
        "observed": obs,
        "n_unconverged": n_unconv,
        "walltime_s": time.time() - t0,
    }
    tmp = out_path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out_path)
    if tonly:
        print(f"  t_comp median {np.median([r['t_comp'] for r in obs]):.2f} (t only)  "
              f"done ({time.time() - t0:.0f}s)")
        return
    print(f"  p_comp median {np.median([r['p_comp'] for r in obs]):.3f}, "
          f"p_point median {np.median([r['p_point'] for r in obs]):.3f}  "
          f"done ({time.time() - t0:.0f}s)")


def run_fill(b, nte, src=None, force=False):
    """Attach this run's (complete) bank to a t-only observed file
    (SIGMA_OBS_TONLY=1) and write the normal {tag}_obs[_fresh].json.
    src: the t-only file, or a run dir holding m{b:02d}/{tag}_obs[_fresh]_tonly.json
    (another null arm with the same model and observed sets; its full
    observed file is accepted too, only its t values are used); default
    this run's own t-only file."""
    name, _ = _obs_name(nte)
    out_path = os.path.join(model_dir(b), name + ".json")
    if os.path.exists(out_path) and not force:
        print(f"[fill m{b:02d} nte{nte}] exists, skipping")
        return
    if src is None:
        src = os.path.join(model_dir(b), name + "_tonly.json")
    elif os.path.isdir(src):
        src = next((p for p in (os.path.join(src, f"m{b:02d}", name + e)
                                for e in ("_tonly.json", ".json"))
                    if os.path.exists(p)),
                   os.path.join(src, f"m{b:02d}", name + "_tonly.json"))
    with open(src) as f:
        result = json.load(f)
    bank_comp, bank_point, bsrc = load_bank(b, nte)
    for r in result["observed"]:
        r["p_comp"] = p_addone(r["t_comp"], bank_comp)
        r["p_point"] = p_addone(r["t_point"], bank_point)
    result.update(toys=int(len(bank_comp)), bank_source=bsrc, t_source=src,
                  bank_comp=[float(t) for t in bank_comp],
                  bank_point=[float(t) for t in bank_point])
    tmp = out_path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out_path)
    print(f"[fill m{b:02d} nte{nte}] {len(result['observed'])} sets, bank {bsrc} "
          f"({len(bank_comp)} toys): p_comp median "
          f"{np.median([r['p_comp'] for r in result['observed']]):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["fit", "gaussian", "bootstrap", "observed", "fill"])
    p.add_argument("--b", type=int, default=0, help="fit index")
    p.add_argument("--nte", type=int, default=25_000, help="N_test")
    p.add_argument("--ntes", type=int, nargs="+", default=None,
                   help="bootstrap: N_test layers served by one refit per toy "
                        "(default: --nte)")
    p.add_argument("--toys", type=int, default=N_TOYS)
    p.add_argument("--chunk", type=int, default=None,
                   help="gaussian/bootstrap: toy chunk index")
    p.add_argument("--nchunks", type=int, default=1)
    p.add_argument("--s-ref", type=int, default=None, dest="s_ref",
                   help=f"fit: reference-set size (default {S_REF})")
    p.add_argument("--src", type=str, default=None,
                   help="fill: t-only file or run dir")
    p.add_argument("--smoke", action="store_true",
                   help="fit: tiny sizes (5k train, 100k reference, 16 kernels)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.stage == "fit":
        run_fit(args.b, smoke=args.smoke, s_ref=args.s_ref, force=args.force)
    elif args.stage == "gaussian":
        run_gaussian(args.b, args.nte, toys=args.toys, force=args.force,
                     chunk=args.chunk, nchunks=args.nchunks)
    elif args.stage == "bootstrap":
        run_bootstrap(args.b, args.ntes or [args.nte], toys=args.toys,
                      force=args.force, chunk=args.chunk, nchunks=args.nchunks)
    elif args.stage == "observed":
        run_observed(args.b, args.nte, force=args.force)
    else:
        run_fill(args.b, args.nte, src=args.src, force=args.force)


if __name__ == "__main__":
    main()
