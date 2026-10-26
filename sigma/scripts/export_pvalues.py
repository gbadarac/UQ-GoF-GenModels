"""Collect per-set p-values from finished runs into the files under data/pvalues/ that the plots read.

For every arm (a pair of run directories: <run> holds the fits, the Gaussian/point banks and the observed statistics;
<run>_boot holds the bootstrap banks) and every N_test, each observed test set of each fit gets
  p_bootstrap = rank of its composite statistic in that fit's bootstrap null,
  p_gaussian  = rank of its composite statistic in that fit's Gaussian null,
  p_point     = rank of its point statistic in that fit's point null,
with p = (#{null >= observed} + 1) / (n_null + 1). Observed statistics are read from <run> or <run>_boot (full or
t-only observed files; both directories share the fit and the test sets).

usage: SIGMA_RUNS=<runs dir> python export_pvalues.py [ladders|resolution|lambda|all]  -> ../data/pvalues/*.json
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.environ.get("SIGMA_RUNS", os.path.join(ROOT, "runs"))
OUT = os.path.join(ROOT, "data", "pvalues")
NT_2D = [1000, 10000, 20000, 50000, 100000, 200000]
NT_4D = [100, 200, 1000, 2000, 10000, 20000, 50000, 100000]


def tag(n):
    return f"nte{n // 1000:03d}k" if n % 1000 == 0 else f"nte{n}"


def pv(t, bank):
    return (1 + (bank[None, :] >= t[:, None]).sum(1)) / (len(bank) + 1)


def bank(d, L, key):
    fs = [f for f in glob.glob(f"{d}/{L}_bank*.npz") if "partial" not in f]
    return np.concatenate([np.load(f)[key] for f in sorted(fs)]) if fs else None


def observed(dirs, L):
    for d in dirs:
        for suf in ("_obs_fresh.json", "_obs.json", "_obs_fresh_tonly.json", "_obs_tonly.json"):
            f = f"{d}/{L}{suf}"
            if os.path.exists(f):
                obs = json.load(open(f))["observed"]
                return np.array([o["t_comp"] for o in obs]), np.array([o["t_point"] for o in obs])
    return None


def cell(run, nfits, n):
    L = tag(n)
    rec = {k: [] for k in ("p_bootstrap", "fit_bootstrap", "p_gaussian", "p_point", "fit_gaussian")}
    for b in range(nfits):
        g, bt = f"{RUNS}/{run}/m{b:02d}", f"{RUNS}/{run}_boot/m{b:02d}"
        ob = observed((g, bt), L)
        if ob is None:
            continue
        tc, tp = ob
        bb = bank(bt, L, "bank_comp")
        if bb is not None:
            rec["p_bootstrap"] += list(pv(tc, bb)); rec["fit_bootstrap"] += [b] * len(tc)
        gc, gp = bank(g, L, "bank_comp"), bank(g, L, "bank_point")
        if gc is not None:
            rec["p_gaussian"] += list(pv(tc, gc)); rec["p_point"] += list(pv(tp, gp)); rec["fit_gaussian"] += [b] * len(tc)
    rec = {k: [float(x) for x in v] if k.startswith("p_") else [int(x) for x in v] for k, v in rec.items()}
    rec["n_fits"] = nfits
    return rec if rec["p_bootstrap"] or rec["p_gaussian"] else None


def write(name, arms):
    out = {}
    for key, run, nfits, ntes in arms:
        for n in ntes:
            c = cell(run, nfits, n)
            if c is not None:
                out[f"{key}_{n}"] = c
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, name), "w"))
    print(f"{name}: {len(out)} cells")


SETS = {
    "ladders": lambda: (write("ladder_toy2d.json", [(f"toy2d_K{K}", f"toy2d_K{K}_J500_s0.4", 20, NT_2D) for K in (16, 32, 64, 128)]),
                        write("ladder_jets4d.json", [(f"jets4d_K{K}", f"jets4d_K{K}_J500_s0.53", 10, NT_4D) for K in (128, 256, 512)])),
    "resolution": lambda: write("resolution_toy2d.json", [(f"toy2d_K{K}_sigma{s}", f"toy2d_K{K}_J500_s{s}", 20, NT_2D)
                                                          for K in (16, 32) for s in ("0.4", "0.1")]),
    "lambda": lambda: write("lambda_toy2d_K64.json", [(f"toy2d_K64_lam{l}", f"toy2d_K64_J500_s0.4" + ("" if l == "0.008" else f"_lam{l}"), 20, NT_2D)
                                                      for l in ("1e-4", "1e-3", "0.008", "0.03", "0.1")]),
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, f in SETS.items():
        if which in (k, "all"):
            f()
