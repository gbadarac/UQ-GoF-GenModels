"""
Unconstrained-parametrized full-covariance Gaussian mixture.

theta in R^P, P = (K-1) + K*d + K*d*(d+1)/2, laid out as
    [ logits a_1..a_{K-1} (a_K = 0 fixed) |
      means mu_k row-major |
      per-component precision-Cholesky params ]
The precision of component k is U_k U_k^T with U_k UPPER-triangular
(sklearn's precisions_cholesky_ convention: z = U^T (x - mu), so
log N = -d/2 log 2pi + sum_i log U_ii - ||z||^2 / 2). Diagonal entries are
stored as log U_ii so every theta in R^P is a valid density; off-diagonals
(i < j) are stored raw, row-major.

Everything is float64 and vectorized over (N, K).

Environment:
  SIGMA_EM_TOL     EM stopping tolerance on the change in mean log-likelihood
                   per iteration (default 1e-6).
  SIGMA_REG_COVAR  sklearn's covariance floor added to every component
                   (default 1e-10).
"""

import os

import numpy as np
from scipy.special import logsumexp
from scipy.linalg import solve_triangular

LOG2PI = np.log(2.0 * np.pi)


class UGMM:
    def __init__(self, K, d):
        self.K, self.d = int(K), int(d)
        iu = np.triu_indices(d)
        self.iu_i, self.iu_j = iu                     # upper-tri incl. diag
        self.n_chol = d * (d + 1) // 2
        self.P = (K - 1) + K * d + K * self.n_chol

    # ── pack / unpack ────────────────────────────────────────────────
    def pack(self, weights, means, prec_chols):
        """From constrained quantities (weights sum to 1, U_k upper with
        positive diagonal) to theta."""
        K, d = self.K, self.d
        a = np.log(weights[:-1]) - np.log(weights[-1])
        chol_params = np.empty((K, self.n_chol))
        for k in range(K):
            U = prec_chols[k]
            vals = U[self.iu_i, self.iu_j].copy()
            diag_pos = self.iu_i == self.iu_j
            vals[diag_pos] = np.log(np.diag(U))
            chol_params[k] = vals
        return np.concatenate([a, np.asarray(means).ravel(),
                               chol_params.ravel()]).astype(np.float64)

    def unpack(self, theta):
        K, d = self.K, self.d
        a = np.concatenate([theta[:K - 1], [0.0]])
        logw = a - logsumexp(a)
        means = theta[K - 1:K - 1 + K * d].reshape(K, d)
        cp = theta[K - 1 + K * d:].reshape(K, self.n_chol)
        Us = np.zeros((K, d, d))
        diag_pos = self.iu_i == self.iu_j
        for k in range(K):
            vals = cp[k].copy()
            vals[diag_pos] = np.exp(vals[diag_pos])
            Us[k][self.iu_i, self.iu_j] = vals
        return logw, means, Us

    # ── densities ────────────────────────────────────────────────────
    def _component_terms(self, X, theta):
        """u (N,K,d), z (N,K,d), logN (N,K)."""
        logw, means, Us = self.unpack(theta)
        u = X[:, None, :] - means[None, :, :]
        z = np.einsum('nki,kij->nkj', u, Us)
        logdet = np.log(np.diagonal(Us, axis1=1, axis2=2)).sum(axis=1)
        logN = (-0.5 * self.d * LOG2PI + logdet[None, :]
                - 0.5 * (z ** 2).sum(axis=2))
        return logw, means, Us, u, z, logN

    def log_density(self, X, theta):
        logw, _, _, _, _, logN = self._component_terms(X, theta)
        return logsumexp(logw[None, :] + logN, axis=1)

    # ── analytic scores ──────────────────────────────────────────────
    def scores(self, X, theta):
        """(N, P) matrix of d log p / d theta_a at theta."""
        K, d = self.K, self.d
        logw, means, Us, u, z, logN = self._component_terms(X, theta)
        lj = logw[None, :] + logN
        r = np.exp(lj - logsumexp(lj, axis=1, keepdims=True))   # (N,K)
        w = np.exp(logw)

        g_a = r[:, :K - 1] - w[None, :K - 1]
        g_mu = r[:, :, None] * np.einsum('kij,nkj->nki', Us, z)

        # dlogN/dU_ij = delta_ij / U_ii - u_i z_j ; log-diag chain rule.
        G = -u[:, :, :, None] * z[:, :, None, :]                # (N,K,d,d)
        diag_scale = np.diagonal(Us, axis1=1, axis2=2)          # (K,d)
        idx = np.arange(d)
        G[:, :, idx, idx] = 1.0 + diag_scale[None, :, :] * G[:, :, idx, idx]
        g_chol = r[:, :, None] * G[:, :, self.iu_i, self.iu_j]  # (N,K,nchol)

        return np.concatenate(
            [g_a, g_mu.reshape(len(X), K * d),
             g_chol.reshape(len(X), K * self.n_chol)], axis=1)

    def scores_sum(self, X, theta):
        """sum_n d log p(x_n) / d theta — the same vector as
        scores(X, theta).sum(0) (agrees to ~1e-15 relative), computed as
        K batched (d x N)(N x d) products instead of materialising the
        (N, K, d, d) tensor: ~2.5x faster and O(N K d) memory instead of
        O(N K d^2). Used wherever only the feature SUM is needed (B_sum:
        every toy and every observed set)."""
        K, d = self.K, self.d
        logw, means, Us, u, z, logN = self._component_terms(X, theta)
        lj = logw[None, :] + logN
        r = np.exp(lj - logsumexp(lj, axis=1, keepdims=True))   # (N,K)
        w = np.exp(logw)
        g_a = r[:, :K - 1].sum(0) - len(X) * w[:K - 1]
        g_mu = np.einsum('nk,nki->ki', r, np.einsum('kij,nkj->nki', Us, z))
        # sum_n r_nk (-u_i z_j): per component a weighted (d,N)@(N,d)
        G = -np.einsum('nki,nkj->kij', r[:, :, None] * u, z, optimize=True)
        diag_scale = np.diagonal(Us, axis1=1, axis2=2)          # (K,d)
        idx = np.arange(d)
        G[:, idx, idx] = r.sum(0)[:, None] + diag_scale * G[:, idx, idx]
        g_chol = G[:, self.iu_i, self.iu_j]                     # (K,nchol)
        return np.concatenate([g_a, g_mu.ravel(), g_chol.ravel()])

    # ── sampling ─────────────────────────────────────────────────────
    def sample(self, n, theta, rng):
        logw, means, Us = self.unpack(theta)
        counts = rng.multinomial(n, np.exp(logw))
        out = np.empty((n, self.d))
        pos = 0
        for k in range(self.K):
            nk = counts[k]
            if nk == 0:
                continue
            eps = rng.standard_normal((self.d, nk))
            # cov = U^{-T} U^{-1}  ->  x = mu + U^{-T} eps
            out[pos:pos + nk] = (means[k][:, None]
                                 + solve_triangular(Us[k].T, eps,
                                                    lower=True)).T
            pos += nk
        return out[rng.permutation(n)]


# ── fitting ──────────────────────────────────────────────────────────
# sklearn's default tol=1e-3 is a bar on the CHANGE IN MEAN LOG-LIKELIHOOD
# per EM step, which is too loose here: the tangent construction is erected
# at theta_hat and assumes it is (close to) a stationary point (the score
# identity centres Phi; Cov = F^-1/N_fit is the MLE sampling covariance; the
# composite null is centred at theta_hat). On a well-specified test problem
# tol=1e-3 stopped after 4 iterations at a KL from the converged answer ~180x
# the estimation floor. The default 1e-6 brings theta_hat close enough to
# stationarity for this purpose; mle_stationarity() reports the residual and
# the fit stage refuses fits more than 5 SEM from stationarity.
EM_TOL = float(os.environ.get("SIGMA_EM_TOL", "1e-6"))
EM_MAXITER = 2000
# Covariance floor. The default 1e-10 lets rank-deficient components
# collapse (sd ~1e-5) in large-K fits; 1e-8 (sd floor 1e-4) blocks that while
# changing the thinnest real structure in the 4D data by ~0.2% in variance.
REG_COVAR = float(os.environ.get("SIGMA_REG_COVAR", "1e-10"))


def fit_em(X, K, seed, n_init=3, reg_covar=None, max_iter=EM_MAXITER,
           tol=EM_TOL):
    """sklearn EM fit (best of n_init k-means++ initialisations) ->
    (UGMM, theta). The EM solution is used as theta_hat directly."""
    reg_covar = REG_COVAR if reg_covar is None else reg_covar
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=K, covariance_type='full',
                         reg_covar=reg_covar, n_init=n_init,
                         max_iter=max_iter, random_state=seed, tol=tol,
                         init_params='k-means++')
    gm.fit(X)
    model = UGMM(K, X.shape[1])
    theta = model.pack(gm.weights_, gm.means_, gm.precisions_cholesky_)
    return model, theta


def mle_stationarity(model, theta, X):
    """max |mean_i psi_a(x_i)| over the TRAINING sample — algebraically zero
    at the MLE, so this is a direct read of optimizer convergence. Reported
    relative to the per-coordinate score scale, i.e. in units of the
    standard error of that mean, which is how far theta_hat sits from the
    stationary point in Fisher metric."""
    psi = model.scores(X, theta)
    mean = psi.mean(0)
    sem = psi.std(0, ddof=1) / np.sqrt(len(X))
    z = np.abs(mean) / np.maximum(sem, 1e-300)
    return float(np.abs(mean).max()), float(z.max())
