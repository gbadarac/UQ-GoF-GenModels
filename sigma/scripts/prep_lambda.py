"""Write one model per Lambda_max for the ablation: the base run's model.npz with the whitening W rebuilt from its
stored Fisher eigendecomposition at each Lambda_max (fit, dictionary and kernel moments unchanged), into
<base>_lam<tag> and <base>_lam<tag>_boot. Checks that the rebuild at the base's own Lambda_max reproduces its W.

usage (from sigma_gof/): python ../scripts/prep_lambda.py <base run> <n_fits>
"""
import json
import os
import sys

import numpy as np

from pipeline import RUNS
from tangent import rebuild_whitening

ARMS = {"1e-4": 1e-4, "1e-3": 1e-3, "0.03": 0.03, "0.1": 0.1}
base, nf = sys.argv[1], int(sys.argv[2])
for b in range(nf):
    z = dict(np.load(f"{RUNS}/{base}/m{b:02d}/model.npz"))
    W0, _ = rebuild_whitening(z["fisher_f"], z["fisher_V"], int(z["n_train"]), float(z["lam_max"]))
    assert W0.shape == z["W"].shape and np.allclose(W0, z["W"]), f"m{b:02d}: rebuild does not reproduce W"
    Ms = {}
    for tag, lam in ARMS.items():
        W, diag = rebuild_whitening(z["fisher_f"], z["fisher_V"], int(z["n_train"]), lam)
        Ms[tag] = diag["M"]
        out = dict(z, W=W, lam_max=lam)
        for arm in ("", "_boot"):
            d = f"{RUNS}/{base}_lam{tag}{arm}/m{b:02d}"
            os.makedirs(d, exist_ok=True)
            np.savez(f"{d}/model.npz", **out)
            info = f"{RUNS}/{base}/m{b:02d}/model_info.json"
            if os.path.exists(info):
                json.dump(dict(json.load(open(info)), lam_max=lam, M=diag["M"]), open(f"{d}/model_info.json", "w"), indent=2)
    print(f"m{b:02d}: kept directions M per Lambda_max {Ms} (nominal {z['W'].shape[1]})", flush=True)
