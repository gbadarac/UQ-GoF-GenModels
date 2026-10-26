"""Regenerate the bootstrap null's refits for one fit, exactly as the
`bootstrap` stage of pipeline.py draws them (pipeline.bootstrap_refit), and
save their parameters — the bank files keep only the test statistics. Used
e.g. for marginal plots with bootstrap bands.

Toy i of fit b: fit sample X_i ~ p_theta_hat (n_train events, seed
[MASTER, b, 0, BANK_SEED_SLOT, i, 3]), refit cold with the real fit's
estimator (best of SIGMA_BOOT_NINIT k-means++ EM starts, seed
SEED_MODEL_BASE + b + 1e6 (i+1) + 5e5). The estimator env (SIGMA_EM_TOL,
SIGMA_REG_COVAR, SIGMA_BOOT_NINIT) must match the model — checked against
model.npz.

usage: python refits.py <run dir>/m00 <i_lo> <i_hi>   -> <run dir>/m00/refits/toy_{i:03d}.npz
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmm import UGMM
from pipeline import bootstrap_refit, check_estimator


def main(mdir, i_lo, i_hi):
    z = np.load(os.path.join(mdir, "model.npz"))
    check_estimator(z)
    b = int(os.path.basename(os.path.normpath(mdir))[1:])
    model, theta, n_train = UGMM(int(z["K"]), int(z["d"])), z["theta"], int(z["n_train"])
    out = os.path.join(mdir, "refits")
    os.makedirs(out, exist_ok=True)
    for i in range(i_lo, i_hi):
        f = os.path.join(out, f"toy_{i:03d}.npz")
        if os.path.exists(f):
            continue
        t0 = time.time()
        Xi, th = bootstrap_refit(model, theta, n_train, b, i)
        ll = float(model.log_density(Xi, th).mean())
        np.savez(f, theta=th, train_ll=ll, i=i)
        print(f"toy {i}: train ll {ll:.5f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
