"""Fig. 7 (2D version): distributions of the test statistic for the 2D toy, K = 64, N_test = N_train = 100k (20 fits
pooled): the observed point statistic (dashed) over its point null (grey), and the observed composite statistic
(solid red) over its bootstrap null (light red).

usage: python plot_fig7.py [N_test tag, default 100k]  -> figures/fig7_toy2d_K64_<tag>
Input: data/fig7/toy2d_K64_statistics.npz  (arrays <name>_<tag> with shape fit x set or fit x toy:
       t_comp_obs, t_point_obs, boot_comp, gauss_comp, point_null)
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from common import DATA, save

k = sys.argv[1] if len(sys.argv) > 1 else "100k"
z = np.load(os.path.join(DATA, "fig7", "toy2d_K64_statistics.npz"))
t_point, point_null = z[f"t_point_obs_{k}"].ravel(), z[f"point_null_{k}"].ravel()
t_comp, boot = z[f"t_comp_obs_{k}"].ravel(), z[f"boot_comp_{k}"].ravel()

allv = np.concatenate([t_point, point_null, t_comp, boot])
edges = np.linspace(np.percentile(allv, 0.2), np.percentile(allv, 99.8), 60)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.hist(point_null, bins=edges, density=True, color="#b9b9b9", alpha=0.7, label="point null")
ax.hist(t_point, bins=edges, density=True, histtype="step", ls="--", lw=1.6, color="#555555", label="observed (point)")
ax.hist(boot, bins=edges, density=True, color="#f0b9b1", alpha=0.8, label="bootstrap null")
ax.hist(t_comp, bins=edges, density=True, histtype="step", lw=1.8, color="#c0392b", label="observed (composite)")
ax.set_xlabel("test statistic $t$"); ax.set_ylabel("density")
ax.set_title(rf"2D, $K=64$, $J=500$, $\sigma=0.4$, $N_{{\rm train}}=100k$, $N_{{\rm test}}={k}$", fontsize=10)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
save(fig, f"fig7_toy2d_K64_{k}", dpi=160)
