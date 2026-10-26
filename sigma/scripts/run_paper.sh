#!/bin/bash
# Every run behind the paper's GMM figures. Each line submits independent SLURM job chains; the Lambda ablation and
# the refits need the corresponding fits to exist, so run those two after the fit jobs have finished.
set -e
D=$(dirname "$0")
for K in 16 32 64 128; do bash $D/run_arm.sh toy2d $K 0.4; done          # Figs. 4, 5, 7
for K in 16 32;        do bash $D/run_arm.sh toy2d $K 0.1; done          # kernel-width appendix (App. E)
for K in 128 256 512; do bash $D/run_arm.sh jets4d $K 0.53; done     # Figs. 8, 9, 14
echo "after the fits finish:  bash $D/run_lambda.sh ;  bash $D/run_refits.sh"
