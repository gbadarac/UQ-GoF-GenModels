"""
2D toy benchmark: a bimodal Gaussian mixture in x0 times a skew-normal in x1.
"""

import numpy as np


def generate_2d_gmm_skew(N, seed=42):
    """
    2D distribution:
      x0: bimodal Gaussian mixture  (50/50, mu=-0.70/sig=0.12 and mu=-0.30/sig=0.12)
      x1: skew-normal               (loc=1.0, scale=0.75, alpha=8.0)
    Uses (and reseeds) numpy's global RandomState.
    """
    np.random.seed(seed)

    # Feature 0: bimodal Gaussian mixture
    wG = 0.50
    mu_a, sig_a = -0.70, 0.12
    mu_b, sig_b = -0.30, 0.12
    n_a = np.random.binomial(N, wG)
    x0 = np.concatenate([
        np.random.normal(mu_a, sig_a, n_a),
        np.random.normal(mu_b, sig_b, N - n_a),
    ])

    # Feature 1: skew-normal
    loc, scale, alpha = 1.0, 0.75, 8.0
    delta = alpha / np.sqrt(1.0 + alpha ** 2)
    z0 = np.random.randn(N)
    z1 = np.random.randn(N)
    x1 = loc + scale * (delta * np.abs(z0) + np.sqrt(1.0 - delta ** 2) * z1)

    data = np.column_stack([x0, x1])
    np.random.shuffle(data)
    return data.astype(np.float64)
