#!/bin/bash
# Draw every GMM figure of the paper from the data in data/ (no cluster runs needed) into figures/.
# Fig. 8 needs the 4D jet-embedding files: set SIGMA_JETS_DATA to their directory, otherwise it is skipped.
set -e
cd "$(dirname "$0")/plots"
PY=${PY:-python3}
$PY plot_marginals.py toy2d      # Fig. 4
$PY plot_ladders.py toy2d        # Fig. 5
$PY plot_fig7.py 100k            # Fig. 7 (2D version)
if [ -n "$SIGMA_JETS_DATA" ]; then $PY plot_marginals.py jets4d; else echo "skipping Fig. 8 (set SIGMA_JETS_DATA)"; fi
$PY plot_ladders.py jets4d       # Fig. 9
$PY plot_null_dists.py           # Fig. 15 (App. D)
$PY plot_lambda.py               # Fig. 14, Lambda_max ablation (App. C)
$PY plot_resolution.py           # Fig. 16, kernel-width appendix (App. E)
