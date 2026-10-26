#!/bin/bash
# Lambda_max ablation (2D toy, K = 64, sigma = 0.4), after run_arm.sh toy2d 64 0.4 has fitted the 20 models.
# 1. prep_lambda.py writes one model per Lambda_max (whitening rebuilt from the stored Fisher eigendecomposition;
#    fit, dictionary and kernel moments unchanged) into toy2d_K64_J500_s0.4_lam<L> (+ "_boot").
# 2. Bootstrap: ONE array in the 1e-4 arm whose refits also produce the banks of the other arms
#    (SIGMA_BOOT_LAMS), so the arms are paired toy by toy. 3. Gaussian/point banks and observed t per arm.
# The nominal Lambda_max = 0.008 arm is the main toy2d_K64_J500_s0.4 run.
set -e
source "$(dirname "$0")/common.sh"
NF=20; NT=$NT_TOY2D; NNT=$(echo $NT | wc -w); NTA="NT=($NT)"
BASE=toy2d_K64_J500_s0.4
ENV="SIGMA_BENCH=toy2d SIGMA_K=64 SIGMA_KERNEL_SIGMA=0.4 $PAPER_ENV SIGMA_OBS_FRESH=1"
(cd "$SNAP/sigma_gof" && $PY -u ../scripts/prep_lambda.py $BASE $NF)
ARMS="0.001:_J500_s0.4_lam1e-3_boot,0.03:_J500_s0.4_lam0.03_boot,0.1:_J500_s0.4_lam0.1_boot"
B=$(submit boot_lambda 0-$((NF * 10 - 1)) 16 96G 1-00:00 "" \
     "env $ENV SIGMA_LAM_MAX=0.0001 SIGMA_SUFFIX=_J500_s0.4_lam1e-4_boot SIGMA_BOOT_LAMS=$ARMS $PY -u pipeline.py bootstrap \
      --b \$((SLURM_ARRAY_TASK_ID % $NF)) --ntes $NT --toys 100 --chunk \$((SLURM_ARRAY_TASK_ID / $NF)) --nchunks 10")
for L in 1e-4:0.0001 1e-3:0.001 0.03:0.03 0.1:0.1; do
  t=${L%%:*}; lam=${L##*:}
  G=$(submit gauss_lam$t 0-$((NF * NNT - 1)) 16 48G 1-00:00 "" \
       "$NTA; env $ENV SIGMA_LAM_MAX=$lam SIGMA_SUFFIX=_J500_s0.4_lam$t $PY -u pipeline.py gaussian --b \$((SLURM_ARRAY_TASK_ID % $NF)) --nte \${NT[\$((SLURM_ARRAY_TASK_ID / $NF))]} --toys 250")
  O=$(submit obs_lam$t 0-$((NF * NNT - 1)) 16 48G 1-00:00 "" \
       "$NTA; env $ENV SIGMA_LAM_MAX=$lam SIGMA_SUFFIX=_J500_s0.4_lam$t SIGMA_OBS_TONLY=1 $PY -u pipeline.py observed --b \$((SLURM_ARRAY_TASK_ID % $NF)) --nte \${NT[\$((SLURM_ARRAY_TASK_ID / $NF))]}")
  echo "Lambda_max $lam: gaussian $G | observed $O"
done
echo "bootstrap (all four arms): $B"
