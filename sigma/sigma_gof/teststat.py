"""
Convex penalized log-likelihood-ratio maximization and the two statistics:

  ell(v, alpha) = sum_B . beta - n_test * (logsumexp(B_R0 beta) - log S)
                  - ||v||^2 / 2 - lam_alpha ||alpha||^2

  t_comp  = 2 [ max_{v, |alpha|<=b} ell  -  max_v ell(v, 0) ]
  t_point = 2   max_{|alpha|<=b} ell(0, alpha)

with beta = (v, alpha), v the M whitened tangent coordinates and alpha the J
kernel coefficients. -ell is convex (log-sum-exp of linear functions +
convex quadratics + box) so the optimum is unique.

Solver. The primary solver is a fixed-preconditioner ascent: exact gradient
every step, with the Hessian FROZEN at beta = 0,

  H0 = n Cov_R0(B) + 2 diag(pen),

built and Cholesky-factored once per feature block and cached. Same
objective, same gradient and same convergence bar as an exact Newton
method — only the step direction is approximate, costing a few cheap extra
iterations instead of an S x F^2 Hessian build per iteration. Whenever the
sup-norm box becomes active or the frozen iteration stalls, the solve falls
back to an exact projected Newton method using

  H(beta) = n * Cov_tilted(B) + 2 diag(pen),   tilt weights u ∝ exp(B beta).

Why second-order at all: the J kernel features are strongly collinear —
Cov_R0(k_tilde) spans ~1e1 down to ~1e-9 because overlapping Gaussians on a
low-dimensional space are highly redundant. The L2 ridge lifts the bottom of
the spectrum to 2*lam_alpha, making the optimum unique, but leaves cond(H) ~
n*lam_max/(2*lam_alpha), growing linearly with N_test; first-order and
quasi-Newton methods (L-BFGS-B) then return under-maximized ell. Newton-type
steps are invariant to this conditioning.

Environment:
  SIGMA_CHUNK  row chunk of the Hessian build (default 250000).
"""

import os

import numpy as np
from scipy.linalg import cho_factor, cho_solve

MAXIT = 60
# Hessian-build row chunk: the f64 copy + tilt-weighted copy of a chunk
# cost 2 x rows x F x 8 bytes (31 GB at 250k rows and F = 7743).
# SIGMA_CHUNK (also tangent's score chunk) caps it.
CHUNK = int(os.environ.get("SIGMA_CHUNK", "250000"))
MAXIT_FROZEN = 120
# Acceptance bound on the remaining improvement in ell for the frozen
# solver (t = 2 ell; null-bank sd ~ 10): used both as the in-loop stop and
# the final convergence criterion.
DELTA_BOUND = 1e-3
# Ridge on the tangent block: half the inverse prior variance of v in
# whitened units, i.e. the N(0, I) budget of the composite null.
V_PEN = 0.5
_H0_CACHE = {}


def _h0_key(B, n, pen):
    ai = B.__array_interface__
    return (ai["data"][0], B.shape, ai["strides"], float(n),
            hash(pen.tobytes()))


def _frozen_factor(B, n, pen):
    key = _h0_key(B, n, pen)
    hit = _H0_CACHE.get(key)
    if hit is None:
        S = len(B)
        p0 = np.full(S, 1.0 / S)
        Ep0 = B.mean(axis=0)
        H0 = _hessian(B, pen, n, p0, Ep0)
        hit = cho_factor(H0, lower=True)
        if len(_H0_CACHE) > 6:          # one solve needs <= 3 blocks
            _H0_CACHE.clear()
        _H0_CACHE[key] = hit
    return hit


def _gtol_eff(B, n, gtol):
    """f32 banks put a noise floor ~ n * eps_f32 on the gradient; the
    stopping bar must sit above it or every solve stalls."""
    if B.dtype == np.float32:
        return max(gtol, 3e-7 * max(n, 1.0))
    return gtol


def _maximize_frozen(B, sB, n, pen, box=None, gtol=1e-5):
    """Fixed-preconditioner ascent: exact gradient each step, H0^-1 as the
    (frozen) step metric. Returns None to request the exact-Newton
    fallback (box active, stall, or numerical trouble)."""
    F = B.shape[1]
    gtol = _gtol_eff(B, n, gtol)
    try:
        c = _frozen_factor(B, n, pen)
    except np.linalg.LinAlgError:
        return None
    lo = -box if box is not None else None
    hi = box if box is not None else None
    bounded = np.isfinite(box) if box is not None else np.zeros(F, bool)
    beta = np.zeros(F)
    it = 0
    for it in range(1, MAXIT_FROZEN + 1):
        ell, grad, _, _ = _obj_grad(B, sB, pen, n, beta)
        if bounded.any():
            eps = np.where(bounded, 1e-6 * np.where(bounded, box, 0.0), 0.0)
            if (bounded & (((beta <= lo + eps) & (grad < 0))
                           | ((beta >= hi - eps) & (grad > 0)))).any():
                return None             # box active: exact solver owns this
        gmax = np.abs(grad).max()
        if gmax < gtol:
            break
        step = cho_solve(c, grad)
        # Stop on the remaining-improvement bound 0.5 g^T H0^-1 g (= half
        # g.step, free) — the SAME acceptance criterion applied after the
        # loop, tested every iteration. Without it the f32 objective noise
        # keeps the line search "improving" by rounding at tiny step
        # fractions (hundreds of objective evaluations per solve where ~20
        # move beta); with the bound the solve is 4-10x faster at
        # |delta t| ~ 1e-3.
        if 0.5 * float(grad @ step) < DELTA_BOUND:
            break
        t, improved = 1.0, False
        for _ in range(12):
            cand = beta + t * step
            if box is not None:
                cand = np.clip(cand, lo, hi)
            if _obj_grad(B, sB, pen, n, cand)[0] > ell:
                beta, improved = cand, True
                break
            t *= 0.5
        if not improved:
            break
    ell, grad, _, _ = _obj_grad(B, sB, pen, n, beta)
    gmax = float(np.abs(grad).max())
    converged = bool(gmax < 1e-6 * max(n, 1e3))
    if not converged:
        # Bound the remaining improvement in ell directly: for a concave
        # objective, ell* - ell <= 0.5 g^T H0^-1 g (H0 ~ H in the small-
        # tilt regime). If that bound is negligible for the STATISTIC
        # (t = 2 ell), the solve is done even though f32 objective noise
        # keeps the gradient from reaching the strict bar.
        delta = 0.5 * float(grad @ cho_solve(c, grad))
        if delta < DELTA_BOUND:
            converged = True
        elif gmax >= gtol:
            return None                 # genuinely stalled: exact Newton
    return ell, {"x": beta, "nit": it, "gmax": gmax,
                 "converged": converged,
                 "n_box_active": 0}


def _obj_grad(B, sB, pen, n, beta):
    """ell, gradient, tilt weights p (normalized), and E_p[B]. Works for a
    float32 feature bank: the matvecs run in the bank's dtype (no 2x-memory
    upcast copy), the scalar reductions in float64."""
    b = beta.astype(B.dtype, copy=False)
    eta = (B @ b).astype(np.float64)
    m = eta.max()
    u = np.exp(eta - m)
    su = u.sum()
    p = u / su
    if B.dtype == np.float32:
        # a single 1M-term f32 dot loses ~5e-5 absolute per feature mean,
        # which the n* multiplier turns into O(1) gradient noise: reduce in
        # ~64k-row f32 chunks, accumulate the chunk results in f64.
        p32 = p.astype(np.float32)
        Ep = np.zeros(B.shape[1])
        for lo in range(0, len(B), 65536):
            hi = min(lo + 65536, len(B))
            Ep += (p32[lo:hi] @ B[lo:hi]).astype(np.float64)
    else:
        Ep = p @ B
    lz = np.log(su) + m - np.log(len(B))
    ell = sB @ beta - n * lz - (pen * beta ** 2).sum()
    grad = sB - n * Ep - 2.0 * pen * beta
    return ell, grad, p, Ep


def _hessian(B, pen, n, p, Ep):
    """n * (E_p[B B^T] - E_p[B] E_p[B]^T) + 2 diag(pen), chunked so the
    tilt-weighted copy of B never materializes in full."""
    F = B.shape[1]
    S = np.zeros((F, F), dtype=np.float64)
    for lo in range(0, len(B), CHUNK):
        hi = min(lo + CHUNK, len(B))
        Bc = B[lo:hi]
        # ALWAYS accumulate the Hessian in f64: the kernel block's
        # covariance is very ill-conditioned, and f32 GEMM noise (~n*1e-4)
        # exceeds the +2*pen spectral floor at large n, making H noise-
        # indefinite. For the frozen solver this runs once per feature
        # block, so the f64 cost is amortized.
        Bc = Bc.astype(np.float64, copy=False)
        S += Bc.T @ (Bc * p[lo:hi, None])
    S -= np.outer(Ep, Ep)
    return n * S + 2.0 * np.diag(pen)


def _maximize(B, sB, n, pen, box=None, gtol=1e-5):
    """Maximize ell over one feature block: the frozen-Hessian solver,
    falling back to the exact projected Newton method. box: per-coordinate
    |beta_i| <= box[i] (np.inf for unbounded coords), or None for a fully
    unbounded block. Returns (ell, info dict)."""
    out = _maximize_frozen(B, sB, n, pen, box=box, gtol=gtol)
    if out is not None:
        return out
    # fall through to the exact projected Newton
    gtol = _gtol_eff(B, n, gtol)
    F = B.shape[1]
    beta = np.zeros(F)
    lo = -box if box is not None else None
    hi = box if box is not None else None
    # the composite block passes inf bounds on the v-coordinates
    bounded = np.isfinite(box) if box is not None else np.zeros(F, bool)
    it = 0
    for it in range(1, MAXIT + 1):
        ell, grad, p, Ep = _obj_grad(B, sB, pen, n, beta)
        # Bertsekas two-metric projection: an eps-band (not an exact-
        # boundary test) defines the active set, and active coordinates
        # move along the scaled gradient rather than the Newton direction.
        # A projected Newton step is not a descent direction near a bound,
        # which makes the naive version zigzag.
        if box is not None:
            eps = np.where(bounded, 1e-6 * np.where(bounded, box, 0.0), 0.0)
            active = bounded & (((beta <= lo + eps) & (grad < 0))
                                | ((beta >= hi - eps) & (grad > 0)))
        else:
            active = np.zeros(F, bool)
        free = ~active
        gmax = np.abs(grad[free]).max() if free.any() else 0.0
        if gmax < gtol:
            break
        step = np.zeros(F)
        H = _hessian(B, pen, n, p, Ep)
        try:
            step[free] = np.linalg.solve(H[np.ix_(free, free)], grad[free])
        except np.linalg.LinAlgError:
            step[free] = np.linalg.lstsq(H[np.ix_(free, free)],
                                         grad[free], rcond=None)[0]
        if active.any():
            step[active] = grad[active] / np.diag(H)[active]
        t = 1.0
        improved = False
        for _ in range(50):                     # backtracking on ell
            cand = beta + t * step
            if box is not None:
                cand = np.clip(cand, lo, hi)
            e2 = _obj_grad(B, sB, pen, n, cand)[0]
            if e2 > ell:
                beta, improved = cand, True
                break
            t *= 0.5
        if not improved:                        # at the optimum to precision
            break
    ell, grad, _, _ = _obj_grad(B, sB, pen, n, beta)
    if box is not None:
        active = bounded & (((beta <= lo + 1e-14) & (grad < 0))
                            | ((beta >= hi - 1e-14) & (grad > 0)))
    else:
        active = np.zeros(F, bool)
    free = ~active
    gmax = float(np.abs(grad[free]).max()) if free.any() else 0.0
    # Tolerance scales with n: the gradient is a sum over n_test events, so
    # a fixed absolute bar would be needlessly strict at large N_test.
    return ell, {"x": beta, "nit": it, "gmax": gmax,
                 "converged": bool(gmax < 1e-6 * max(n, 1e3)),
                 "n_box_active": int(active.sum())}


def t_statistics(B_R0, sum_B, n_test, M, alpha_box, lam_alpha,
                 which=("comp", "point")):
    """Statistics for one dataset (which enters only via its feature sum
    sum_B). B_R0: the reference-slice feature matrix, tangent block first
    (M columns), kernel block after."""
    J = B_R0.shape[1] - M
    out = {}

    if "comp" in which:
        pen_full = np.concatenate([np.full(M, V_PEN), np.full(J, lam_alpha)])
        box_full = np.concatenate([np.full(M, np.inf), alpha_box])
        ell_full, r_full = _maximize(B_R0, sum_B, n_test, pen_full, box_full)
        ell_v, r_v = _maximize(B_R0[:, :M], sum_B[:M], n_test,
                               np.full(M, V_PEN), None)
        out["t_comp"] = 2.0 * (ell_full - ell_v)
        out["ell_full"], out["ell_v"] = ell_full, ell_v
        out["conv_comp"] = bool(r_full["converged"] and r_v["converged"])
        out["nit_comp"] = r_full["nit"] + r_v["nit"]
        out["gmax_comp"] = max(r_full["gmax"], r_v["gmax"])
        out["n_box_active"] = r_full["n_box_active"]

    if "point" in which:
        ell_a, r_a = _maximize(B_R0[:, M:], sum_B[M:], n_test,
                               np.full(J, lam_alpha), alpha_box)
        out["t_point"] = 2.0 * ell_a
        out["conv_point"] = bool(r_a["converged"])
        out["nit_point"] = r_a["nit"]
        out["gmax_point"] = r_a["gmax"]
        out["n_box_active_point"] = r_a["n_box_active"]

    return out


def p_addone(t_obs, bank):
    bank = np.asarray(bank)
    return (int((bank >= t_obs).sum()) + 1) / (len(bank) + 1)
