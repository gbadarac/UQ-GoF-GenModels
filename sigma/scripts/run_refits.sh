#!/bin/bash
# Save the 100 bootstrap refits of fit 0 for the marginal-band figures (Figs. 4 and 8). They are the same refits the
# bootstrap stage draws (same seeds), stored as parameter vectors in <run>_boot/m00/refits/.
set -e
source "$(dirname "$0")/common.sh"
for spec in "toy2d 64 0.4 SIGMA_OBS_FRESH=1" "jets4d 256 0.53"; do
  set -- $spec; RUN=$1_K$2_J500_s$3
  submit refits_$RUN 0-9 16 48G 1-00:00 "" \
    "mkdir -p $SIGMA_RUNS/${RUN}_boot/m00; ln -sf $SIGMA_RUNS/$RUN/m00/model.npz $SIGMA_RUNS/${RUN}_boot/m00/model.npz; \
     env SIGMA_BENCH=$1 SIGMA_K=$2 SIGMA_KERNEL_SIGMA=$3 $PAPER_ENV $4 $PY -u refits.py $SIGMA_RUNS/${RUN}_boot/m00 \$((SLURM_ARRAY_TASK_ID * 10)) \$((SLURM_ARRAY_TASK_ID * 10 + 10))"
done
