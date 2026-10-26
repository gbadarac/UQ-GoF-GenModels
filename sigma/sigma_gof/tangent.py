"""
Tangent-space machinery: Fisher whitening of the GMM scores, the fixed
Gaussian-kernel dictionary, and the FeatureMap that produces
B(x) = (Phi(x), k_tilde(x)).

Everything model-derived is estimated on the single reference set R0 (a
large sample from the fitted model); the FeatureMap is a frozen function of
x once built.

Environment:
  SIGMA_CHUNK  row chunk for every pass over a sample (default 200000).
"""

import os

import numpy as np
from scipy.linalg import eigh

# Row chunk for every pass over a sample. The score pass holds several
# (chunk, K, d, d)-sized temporaries, ~1.3 KB per row per 100 components
# in 4D: 200k rows is ~40 GB at K=512. SIGMA_CHUNK=50000 keeps that near
# 10 GB; results are identical up to summation order.
CHUNK = int(os.environ.get("SIGMA_CHUNK", "200000"))


def _chunks(n, size=CHUNK):
    for lo in range(0, n, size):
        yield lo, min(lo + size, n)


def fisher_whiten(model, theta, R0, n_fit, lam_max=0.8, rel_floor=1e-12,
                  return_eig=False):
    """F_hat = mean psi psi^T over R0; Cov = F^+ / n_fit shares F's
    eigenvectors with eigenvalues 1/(n_fit f_i). Directions with cov
    eigenvalue > lam_max (or f below the relative floor) are DROPPED, not
    capped. Returns (W (P,M), diag dict); with return_eig, also the full
    (f, V) so the whitening can be rebuilt at another lam_max without
    another pass over R0 (see rebuild_whitening)."""
    P = model.P
    F = np.zeros((P, P))
    for lo, hi in _chunks(len(R0)):
        psi = model.scores(R0[lo:hi], theta)
        F += psi.T @ psi
    F /= len(R0)
    f, V = eigh(F)                          # ascending
    W, diag = rebuild_whitening(f, V, n_fit, lam_max, rel_floor)
    if return_eig:
        return W, diag, (f, V)
    return W, diag


def rebuild_whitening(f, V, n_fit, lam_max, rel_floor=1e-12):
    """The cut/scale step of fisher_whiten from a stored eigendecomposition
    — lets the cut lam_max vary without re-estimating F on R0."""
    P = len(f)
    ok = f > rel_floor * f.max()
    lam_cov = np.full_like(f, np.inf)
    lam_cov[ok] = 1.0 / (n_fit * f[ok])
    keep = ok & (lam_cov <= lam_max)
    W = V[:, keep] * np.sqrt(lam_cov[keep])[None, :]
    diag = {"P": int(P), "M": int(keep.sum()),
            "n_dropped": int(P - keep.sum()),
            "cov_eig_kept_max": float(lam_cov[keep].max()) if keep.any() else None,
            "fisher_eig_min_kept": float(f[keep].min()) if keep.any() else None}
    return W, diag


def build_dictionary(R0, J, seed, scales=(0.25, 0.60), km_sub=20_000,
                     q_sub=2_000):
    """J Gaussian kernels: k-means centers on a subsample of R0, widths at
    two fractions of the median pairwise distance q50 (half the kernels
    each, interleaved). Coordinates are per-dim standardized by R0 moments
    before distances, so all data dimensions weigh in comparably. Kernels
    are then standardized under R0 (in FeatureMap)."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    x_mean, x_std = R0.mean(axis=0), R0.std(axis=0)
    Xs = (R0[rng.choice(len(R0), size=min(km_sub, len(R0)), replace=False)]
          - x_mean) / x_std
    km = KMeans(n_clusters=J, n_init=4, random_state=seed).fit(Xs)
    centers = km.cluster_centers_

    Q = Xs[rng.choice(len(Xs), size=min(q_sub, len(Xs)), replace=False)]
    d2 = ((Q[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
    q50 = float(np.sqrt(np.median(d2[np.triu_indices(len(Q), k=1)])))

    sigmas = np.where(np.arange(J) % 2 == 0, scales[0] * q50,
                      scales[1] * q50)
    return {"centers": centers, "sigmas": sigmas,
            "x_mean": x_mean, "x_std": x_std, "q50": q50}


def raw_width_dictionary(dic, sigma):
    """The dictionary with its centers mapped back to raw data coordinates,
    no coordinate standardization, and every kernel isotropic with width
    sigma in raw data units (the k-means centers are unchanged)."""
    out = dict(dic)
    out["centers"] = dic["centers"] * dic["x_std"] + dic["x_mean"]
    out["x_mean"] = np.zeros_like(dic["x_mean"])
    out["x_std"] = np.ones_like(dic["x_std"])
    out["sigmas"] = np.full(len(dic["sigmas"]), float(sigma))
    return out


def _raw_kernels(X, dic):
    Xs = (X - dic["x_mean"]) / dic["x_std"]
    d2 = ((Xs[:, None, :] - dic["centers"][None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2.0 * dic["sigmas"][None, :] ** 2))


class FeatureMap:
    """B(x) = (Phi(x), k_tilde(x)) in R^{M+J}. Kernel standardization
    moments and the sup-norm clip bounds are estimated on R0 at build time
    and then frozen."""

    def __init__(self, model, theta, W, dic, R0, clip_A=1.0, moments=None):
        self.model, self.theta, self.W, self.dic = model, theta, W, dic
        self.M, self.J = W.shape[1], len(dic["sigmas"])
        if moments is not None:
            # Moments already measured on the full R0 (stored by the fit
            # stage): recomputing them costs two extra passes over R0 for a
            # bit-identical answer.
            self.k_mean, self.k_std, self.alpha_box = moments
            return
        # kernel moments + clip bounds on the full R0, chunked
        s1 = np.zeros(self.J)
        s2 = np.zeros(self.J)
        for lo, hi in _chunks(len(R0)):
            k = _raw_kernels(R0[lo:hi], dic)
            s1 += k.sum(0)
            s2 += (k ** 2).sum(0)
        self.k_mean = s1 / len(R0)
        self.k_std = np.sqrt(np.maximum(s2 / len(R0) - self.k_mean ** 2,
                                        1e-30))
        kmax = np.zeros(self.J)
        for lo, hi in _chunks(len(R0)):
            kt = np.abs((_raw_kernels(R0[lo:hi], dic) - self.k_mean)
                        / self.k_std)
            kmax = np.maximum(kmax, kt.max(0))
        self.alpha_box = clip_A / kmax          # sup-norm clip on the tilt

    def phi(self, X):
        return self.model.scores(X, self.theta) @ self.W

    def kernels(self, X):
        return (_raw_kernels(X, self.dic) - self.k_mean) / self.k_std

    def B(self, X, dtype=np.float64):
        """Per-row feature matrix. `dtype=np.float32` writes each chunk
        straight into a float32 bank instead of building the float64
        matrix and copying (which would need 3x the bank's memory)."""
        out = np.empty((len(X), self.M + self.J), dtype=dtype)
        for lo, hi in _chunks(len(X)):
            out[lo:hi, :self.M] = self.phi(X[lo:hi])
            out[lo:hi, self.M:] = self.kernels(X[lo:hi])
        return out

    def B_sum(self, X):
        """Feature-sum vector — the only way data enters the statistic.
        Phi is linear in the scores, so the scores are summed first
        (UGMM.scores_sum, no per-row score matrix) and W applied once."""
        s = np.zeros(self.M + self.J)
        for lo, hi in _chunks(len(X)):
            s[:self.M] += self.model.scores_sum(X[lo:hi], self.theta) @ self.W
            s[self.M:] += self.kernels(X[lo:hi]).sum(0)
        return s
