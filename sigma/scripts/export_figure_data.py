"""Collect the raw statistics behind Fig. 7 and Fig. 15, and the parameter draws behind the marginal bands
(Figs. 4 and 8), into data/.

usage: SIGMA_RUNS=<runs dir> python export_figure_data.py [fig7|null_dists|bands|all]
  fig7       -> data/fig7/toy2d_K64_statistics.npz: per fit (20) and N_test, observed composite / point statistics
                (100 sets), bootstrap null (100), Gaussian null (250), point null (250)
  null_dists -> data/null_dists/jets4d_K256_{gaussian,bootstrap}_nte{001k,020k,100k}.json: the same, pooled over the
                10 fits of the 4D K = 256 arm
  bands      -> data/marginals/<bench>_K<K>_fit0_bands.npz: fit 0's parameters, 200 Gaussian draws theta + W z and
                the 100 bootstrap refits (saved by `pipeline.py refits`)
"""
import glob
import json
import os
import sys

import numpy as np

from export_pvalues import RUNS, ROOT, bank, observed, tag

DATA = os.path.join(ROOT, "data")


def fig7():
    run, out = "toy2d_K64_J500_s0.4", {"n_test": np.array([1000, 10000, 20000, 50000, 100000, 200000]), "fits": np.arange(20)}
    for n in out["n_test"]:
        L, k = tag(int(n)), f"{int(n) // 1000}k"
        rows = {x: [] for x in ("t_comp_obs", "t_point_obs", "boot_comp", "gauss_comp", "point_null")}
        for b in range(20):
            g, bt = f"{RUNS}/{run}/m{b:02d}", f"{RUNS}/{run}_boot/m{b:02d}"
            tc, tp = observed((g, bt), L)
            rows["t_comp_obs"].append(tc); rows["t_point_obs"].append(tp)
            rows["boot_comp"].append(bank(bt, L, "bank_comp"))
            rows["gauss_comp"].append(bank(g, L, "bank_comp")); rows["point_null"].append(bank(g, L, "bank_point"))
        for x, v in rows.items():
            out[f"{x}_{k}"] = np.array(v)
    os.makedirs(f"{DATA}/fig7", exist_ok=True)
    np.savez_compressed(f"{DATA}/fig7/toy2d_K64_statistics.npz", **out); print("wrote fig7 data")


def null_dists():
    run = "jets4d_K256_J500_s0.53"
    os.makedirs(f"{DATA}/null_dists", exist_ok=True)
    for n in (1000, 20000, 100000):
        L = tag(n)
        g = dict(observed=[], bank_comp=[], bank_point=[]); bo = dict(observed=[], bank_comp=[])
        for b in range(10):
            gd, bd = f"{RUNS}/{run}/m{b:02d}", f"{RUNS}/{run}_boot/m{b:02d}"
            tc, tp = observed((gd, bd), L)
            obs = [dict(t_comp=float(c), t_point=float(p)) for c, p in zip(tc, tp)]
            g["observed"] += obs; bo["observed"] += obs
            g["bank_comp"] += list(map(float, bank(gd, L, "bank_comp"))); g["bank_point"] += list(map(float, bank(gd, L, "bank_point")))
            bo["bank_comp"] += list(map(float, bank(bd, L, "bank_comp")))
        json.dump(g, open(f"{DATA}/null_dists/jets4d_K256_gaussian_{L}.json", "w"))
        json.dump(bo, open(f"{DATA}/null_dists/jets4d_K256_bootstrap_{L}.json", "w"))
    print("wrote null-distribution data")


def bands():
    os.makedirs(f"{DATA}/marginals", exist_ok=True)
    for bench, run in (("toy2d_K64", "toy2d_K64_J500_s0.4"), ("jets4d_K256", "jets4d_K256_J500_s0.53")):
        z = np.load(f"{RUNS}/{run}/m00/model.npz")
        theta, W = z["theta"], z["W"]
        rng = np.random.default_rng(20260925)
        gauss = np.array([theta + W @ rng.standard_normal(W.shape[1]) for _ in range(200)])
        boot = np.array([np.load(f)["theta"] for f in sorted(glob.glob(f"{RUNS}/{run}_boot/m00/refits/toy_*.npz"))])
        np.savez_compressed(f"{DATA}/marginals/{bench}_fit0_bands.npz", K=int(z["K"]), d=int(z["d"]),
                            n_train=int(z["n_train"]), fit=0, theta=theta, gaussian_thetas=gauss, bootstrap_thetas=boot)
        print(f"wrote {bench} bands ({len(boot)} refits)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, f in (("fig7", fig7), ("null_dists", null_dists), ("bands", bands)):
        if which in (k, "all"):
            f()
