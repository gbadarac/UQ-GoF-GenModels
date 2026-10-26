#!/bin/bash
# One arm of the paper: fit NF models, then the Gaussian/point null, the bootstrap null and the observed statistics
# at every N_test. Run directories: $SIGMA_RUNS/<bench>_K<K>_J500_s<sigma> (fits, Gaussian/point banks, observed t)
# and the same name + "_boot" (bootstrap banks; the fit's model.npz is linked in).
# usage: run_arm.sh <toy2d|jets4d> <K> <sigma> [n_fits]
#   toy2d,  sigma 0.4 : Figs. 4, 5, 7      (K = 16, 32, 64, 128; 20 fits)
#   toy2d,  sigma 0.1 : kernel-width appendix (App. E) (K = 16, 32; 20 fits)
#   jets4d, sigma 0.53: Figs. 8, 9, 14     (K = 128, 256, 512, 768; 10 fits; needs SIGMA_JETS_DATA)
set -e
source "$(dirname "$0")/common.sh"
BENCH=$1; K=$2; SIG=$3
if [ "$BENCH" = toy2d ]; then NF=${4:-20}; NT=$NT_TOY2D; EXTRA="SIGMA_OBS_FRESH=1"; BIG=48G; LONG=1-00:00
else NF=${4:-10}; NT=$NT_JETS4D; EXTRA=""; BIG=128G; LONG=2-00:00; fi
RUN=${BENCH}_K${K}_J500_s${SIG}
ENV="SIGMA_BENCH=$BENCH SIGMA_K=$K SIGMA_KERNEL_SIGMA=$SIG $PAPER_ENV $EXTRA"
NNT=$(echo $NT | wc -w); NTA="NT=($NT)"
F=$(submit fit_$RUN 0-$((NF - 1)) 16 $BIG $LONG "" \
     "env $ENV SIGMA_SUFFIX=_J500_s$SIG $PY -u pipeline.py fit --b \$SLURM_ARRAY_TASK_ID")
# Gaussian + point null: 250 toys per (fit, N_test)
G=$(submit gauss_$RUN 0-$((NF * NNT - 1)) 16 $BIG $LONG $F \
     "$NTA; env $ENV SIGMA_SUFFIX=_J500_s$SIG $PY -u pipeline.py gaussian --b \$((SLURM_ARRAY_TASK_ID % $NF)) --nte \${NT[\$((SLURM_ARRAY_TASK_ID / $NF))]} --toys 250")
# bootstrap null: 100 cold refits per fit in 10 chunks, every N_test from the same refits
B=$(submit boot_$RUN 0-$((NF * 10 - 1)) 16 $BIG $LONG $F \
     "b=\$((SLURM_ARRAY_TASK_ID % $NF)); m=\$(printf m%02d \$b); mkdir -p $SIGMA_RUNS/${RUN}_boot/\$m; \
      ln -sf $SIGMA_RUNS/$RUN/\$m/model.npz $SIGMA_RUNS/${RUN}_boot/\$m/model.npz; \
      env $ENV SIGMA_SUFFIX=_J500_s${SIG}_boot $PY -u pipeline.py bootstrap --b \$b --ntes $NT --toys 100 --chunk \$((SLURM_ARRAY_TASK_ID / $NF)) --nchunks 10")
# observed statistics (t only; p-values are computed against the banks by scripts/export_pvalues.py)
O=$(submit obs_$RUN 0-$((NF * NNT - 1)) 16 $BIG $LONG $F \
     "$NTA; env $ENV SIGMA_SUFFIX=_J500_s$SIG SIGMA_OBS_TONLY=1 $PY -u pipeline.py observed --b \$((SLURM_ARRAY_TASK_ID % $NF)) --nte \${NT[\$((SLURM_ARRAY_TASK_ID / $NF))]}")
echo "$RUN: fit $F | gaussian $G | bootstrap $B | observed $O"
