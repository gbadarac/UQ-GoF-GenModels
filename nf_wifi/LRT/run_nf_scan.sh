#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Unified NF LRT driver.  ONE script, THREE jobs, selected by two knobs:
#
#   DIM  = 2d | 4d           — feature dimension. Selects the SCENARIO, the wifi ensemble
#                              pool size (--n_members), the pinned single-model seed, and
#                              the default sigma.
#   SCAN = campaign | sigma  — WHAT to scan:
#       campaign (default) : the full 10-fit grid
#                              CHOICE(0..9) × N_test × { single + ENS × {point,composite} } × calib
#                            at a FIXED sigma. Per choice: NULL_TOYS Gaussian null + OBS_TOYS obs
#                            (each self-calibrated); plot_pvalue_vs_ntest.py POOLS the per-toy
#                            p-values across the 10 choices for the honest 25/75 band. 512 pool;
#                            member seeds per choice read from the wifi choice<k>/member_seeds.txt.
#       sigma              : single-model σ-SELECTION scan — loop σ × calib at a FIXED
#                            N_test, LEAN toys. Results land in distinct _W{sigma} run-tag
#                            dirs (sigma is in the folder name → no mixing). Pick σ from the
#                            nulls (analyse_LRT_output.py per dir: clean null chi2/DOF_eff +
#                            min observed p), then rerun SCAN=campaign at that FIXED σ.
#                                                              (was run_nf_sigma_scan_4d.sh)
#
# Both modes are thin SLURM fan-outs over submit_LRT_toys.sh → LRT.py; the physics lives
# there. Fixed for both: MODEL=nf, N_ref=N_test (1:1, submit default), M=500, clip=3,
# rng_seed=0, --partition=gpu. ENS=1 (single model) ⇒ LRT.py synthesizes w=[1.0] and forces
# frozen / POINT-null (the ensemblecomponents1 wifi is ignored). MEMBER_SEEDS is comma-
# separated (sbatch --export cannot carry commas, its delimiter) so it is exported into the
# environment and propagated via --export=ALL.
#
# sigma candidates (4D): RAW pairwise-distance percentiles {P1,P25,P50,P75,P90,P99}+2×P99 on
# the training data = {0.05,0.53,1.24,2.11,2.8,3.46,6.92} (features on near-equal scales, LRT
# places kernels on RAW data). Sigma is a data-space length scale (N-independent), so the value
# chosen by the scan transfers to the campaign at larger N_test.
#
# Usage:
#   DRYRUN=1 ./run_nf_scan.sh                          # 2D full campaign, print only (DEFAULT)
#   DRYRUN=0 DIM=2d ./run_nf_scan.sh                   # 2D full campaign, SUBMIT
#   DRYRUN=0 DIM=4d SCAN=sigma ./run_nf_scan.sh        # 4D σ-selection scan, SUBMIT
#   DRYRUN=0 DIM=4d SIGMA=<chosen> ./run_nf_scan.sh    # 4D full campaign at the chosen σ
#
# Subset knobs (space-separated lists; override at call time):
#   campaign : CHOICES  NTEST_LIST  ENS_LIST  MODES  CALIBS  INCLUDE_SINGLE  SIGMA  NULL_TOYS  OBS_TOYS  STD_TIME  LONG_TIME
#   sigma    : SIGMA_LIST  NTEST  CALIBS  TIME  NULL_TOYS  OBS_TOYS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
SELECT="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/select_member_seeds.py"
SELPY="/work/gbadarac/miniforge3/envs/nplm_env/bin/python"   # any python with numpy

DRYRUN=${DRYRUN:-1}
DIM=${DIM:-2d}
SCAN=${SCAN:-campaign}
CALIBS=${CALIBS:-"1 0"}                # 1 = null , 0 = observed/test
M=500; CLIP=3; RNG_SEED=0

# ── DIM config: SCENARIO, pool size (--n_members), wifi choice dirs, default sigma ──
# 10-fit: the pool is 512 for BOTH dims; the 10 ensemble choices live in the wifi results at
# $WIFI_BASE/${WIFI_CONFIG}_ensemblecomponents<M>/choice<k>/ (member_seeds.txt + w_i_fitted + cov_w).
WIFI_ROOT="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF_uniform"
case "$DIM" in
  # ENS default = single/32/128 for BOTH dims (decision 2026-09-28: 2D narrowed to this via job
  # cancellation, 4D planned for it). single always via INCLUDE_SINGLE; ENS_DEF holds M>=2 sizes.
  2d) SCENARIO=ens_2d_uniform; NMEM=512; SIGMA_DEF=0.4; ENS_DEF="32 128"
      WIFI_BASE="$WIFI_ROOT/2d_toymodel"; WIFI_CONFIG="N_100000_dim_2_seeds_512_4_4_64_8_2d_bimodal_gaussian_heavy_tail" ;;
  4d) SCENARIO=ens_4d_uniform; NMEM=512; SIGMA_DEF=0.53; ENS_DEF="32 128"   # σ=0.53 = preliminary min-p σ-scan (override SIGMA= if re-scanned)
      WIFI_BASE="$WIFI_ROOT/4d_embedding"; WIFI_CONFIG="N_100000_dim_4_seeds_512_6_8_128_10_4d_embedding_qcd" ;;
  *)  echo "unknown DIM=$DIM (want 2d|4d)" >&2; exit 1 ;;
esac
SINGLE_SEED=${SINGLE_SEED:-100}   # ONLY used by SCAN=sigma (single-model σ scan); campaign M=1 draws per-choice
COMMON="MODEL=nf,SCENARIO=$SCENARIO,N_KERNELS=$M,CLIP_B=$CLIP"   # KERNEL_SIGMA passed per-iteration

# --mem per N_test, right-sized to measured RSS (Derek 2026-08: don't over-reserve). NF single
# N=10k used ~2 GB; the 128-member 100k/200k cell is heaviest (~2 GB state_dicts + data ∝ N).
# Kept comfortably > observed to avoid OOM.
mem_for() {
  if   [[ "$1" -ge 100000 ]]; then echo "14G"    # 100k / 200k
  elif [[ "$1" -ge 10000  ]]; then echo "8G"     # 10k / 20k
  else echo "6G"                                  # ≤2000
  fi
}

# member seeds for ensemble size $1 and choice $2, IN ORDER (last = norm model):
#   M>=2 : read that choice's OWN member_seeds.txt (authoritative — guarantees the LRT loads
#          exactly the members the choice's wifi fit paired its weights/cov with).
#   M==1 : no wifi fit -> draw one flow from the 512 pool (reproducible per choice via
#          rng_seed=choice) = the 10 single-flow "choices" for the single-model baseline.
resolve_seeds() {  # $1=M  $2=choice
  if [[ "$1" -eq 1 ]]; then
    "$SELPY" "$SELECT" --mode uniform --n_members "$NMEM" --M 1 --rng_seed "$2" 2>/dev/null
  else
    local f="$WIFI_BASE/${WIFI_CONFIG}_ensemblecomponents$1/choice$2/member_seeds.txt"
    [[ -f "$f" ]] && cat "$f" || { echo "MISSING member_seeds.txt: $f" >&2; return 1; }
  fi
}

# batch size (toys per SLURM task): pack many toys into one task so the TOTAL job count stays far
# below the cluster MaxJobCount (one-job-per-toy = ~510k jobs = ~5x the cap, hogs the shared table).
# Smaller batch for heavy (large-Ntest) toys so each task stays well inside its wall (a wall-timeout
# then loses <= batch toys, not the whole cell). Override uniformly with BATCH=<n>.
batch_for() {   # CPU-tuned: heavy toys are slower on CPU + the composite fit can crawl, so keep
  if   [[ "$1" -ge 200000 ]]; then echo 2    # each task comfortably under the 12h standard wall
  elif [[ "$1" -ge 50000  ]]; then echo 4
  elif [[ "$1" -ge 10000  ]]; then echo 15
  else echo 50
  fi
}
# array range = ceil(N_toys / batch) - 1.  $1 = calib (1=null → NULL_TOYS ; 0=obs → OBS_TOYS), $2 = batch.
array_for() { local nt; [[ "$1" -eq 1 ]] && nt=$NULL_TOYS || nt=$OBS_TOYS; echo "0-$(( (nt + $2 - 1) / $2 - 1 ))"; }

# Submission throttle + retry: firing hundreds of array jobs in a tight loop overloads slurmctld
# -> "sbatch: Resource temporarily unavailable" (EAGAIN). Pause between submissions and retry
# with backoff on transient failure so the driver rides through a busy controller instead of
# dying (set -e) mid-chunk. Bump SUBMIT_SLEEP if EAGAIN still appears.
SUBMIT_SLEEP=${SUBMIT_SLEEP:-0.5}     # seconds between successful submissions
SUBMIT_RETRIES=${SUBMIT_RETRIES:-10}  # retries on transient failure (backoff 10s,20s,... capped 60s)
# Compute resource — DEFAULT is CPU (standard/t3, no GPU): the bottleneck is the scipy composite
# numerator fit, which is CPU-bound (the GPU sits idle ~93% of a toy), so t3's ~3400 CPU cores
# beat the 16 GPUs massively. LRT.py auto-runs on CPU when no GPU is present, so no code change.
# For a GPU run instead: SB_PARTITION=gpu SB_ACCOUNT=gpu_gres SB_GRES=gpu:1.
SB_PARTITION=${SB_PARTITION:-standard}
SB_ACCOUNT=${SB_ACCOUNT:-t3}
SB_GRES=${SB_GRES:-none}
submit() {  # $1=array $2=time $3=mem $4=export-extra ; MEMBER_SEEDS taken from the env
  if [[ "$DRYRUN" == "1" ]]; then
    echo "MEMBER_SEEDS=${MEMBER_SEEDS} sbatch --partition=$SB_PARTITION --account=$SB_ACCOUNT --gres=$SB_GRES --time=$2 --mem=$3 --array=$1 --export=ALL,$COMMON,$4 submit_LRT_toys.sh"
    return 0
  fi
  local a back
  for ((a=1; a<=SUBMIT_RETRIES; a++)); do
    if sbatch --partition="$SB_PARTITION" --account="$SB_ACCOUNT" --gres="$SB_GRES" --time="$2" --mem="$3" --array="$1" --export=ALL,"$COMMON,$4" submit_LRT_toys.sh; then
      sleep "$SUBMIT_SLEEP"; return 0
    fi
    back=$(( a*10 )); [ "$back" -gt 60 ] && back=60
    echo "  [submit] slurmctld busy (attempt $a/$SUBMIT_RETRIES) — backing off ${back}s..." >&2
    sleep "$back"
  done
  echo "ERROR: sbatch still failing after $SUBMIT_RETRIES retries — controller/limit saturated; wait for the queue to drain, then re-run the remaining CHOICES." >&2
  return 1
}

n=0
if [[ "$SCAN" == "sigma" ]]; then
  # ── σ-SELECTION scan: single model, fixed N_test, loop σ × calib, LEAN toys ──
  SIGMA_LIST=${SIGMA_LIST:-"0.05 0.53 1.24 2.11 2.8 3.46 6.92"}
  NTEST=${NTEST:-10000}
  SIG_TIME=${TIME:-"02:00:00"}
  NULL_TOYS=${NULL_TOYS:-100}; OBS_TOYS=${OBS_TOYS:-50}
  mem=$(mem_for "$NTEST"); b=${BATCH:-$(batch_for "$NTEST")}
  export MEMBER_SEEDS="$SINGLE_SEED"    # ENS=1 ⇒ LRT.py forces frozen/point-null, w=[1.0]
  for sigma in $SIGMA_LIST; do
    for calib in $CALIBS; do
      [[ "$calib" -eq 1 ]] && nt=$NULL_TOYS || nt=$OBS_TOYS
      submit "$(array_for "$calib" "$b")" "$SIG_TIME" "$mem" "ENS=1,CHOICE=0,CALIBRATION=$calib,NTEST=$NTEST,KERNEL_SIGMA=$sigma,TOYS_PER_JOB=$b,N_TOYS=$nt"
      n=$((n+1))
    done
  done
  echo "# $n array-submissions [DRYRUN=$DRYRUN]  (DIM=$DIM σ-scan: single model seed $SINGLE_SEED, N_test=$NTEST, null=$NULL_TOYS / obs=$OBS_TOYS toys per σ)"

elif [[ "$SCAN" == "campaign" ]]; then
  # ── 10-fit campaign: fixed σ, loop CHOICE × N_test × { single + ENS × modes } × calib ──
  # The 10 CHOICES are the fit-to-fit axis (Sean's recipe): each is a random M-subsample of the
  # 512 pool, stored side-by-side under choice<k>/ so plot_pvalue_vs_ntest.py can POOL the per-toy
  # p-values across them for the honest 25/75 band. Per choice: NULL_TOYS Gaussian null + OBS_TOYS
  # obs toys, EACH self-calibrated by its own null.
  SIGMA=${SIGMA:-$SIGMA_DEF}
  [[ -z "$SIGMA" ]] && { echo "campaign mode (DIM=$DIM) needs SIGMA=<value> — run 'SCAN=sigma' first to pick it" >&2; exit 1; }
  # 10-fit grids (2026-09): 50k (=0.5·N_train, the N_train≈N_test regime) INCLUDED for both
  # (Giada 2026-09-26). 4D DROPS 200k (200k/1M test-pool = 20% > Sean's ≤10% rule); 2D KEEPS 200k.
  if [[ "$DIM" == "4d" ]]; then NTEST_DEF="100 200 1000 2000 10000 20000 50000 100000"
  else                          NTEST_DEF="100 200 1000 2000 10000 20000 50000 100000 200000"; fi
  NTEST_LIST=${NTEST_LIST:-$NTEST_DEF}
  ENS_LIST=${ENS_LIST:-$ENS_DEF}   # DIM-dependent default: 2D = full 7 sizes; 4D = 32 128 (see DIM case above)
  MODES=${MODES:-"point composite"}       # point = frozen weights (plug-in) ; composite = constrained
  INCLUDE_SINGLE=${INCLUDE_SINGLE:-1}
  CHOICES=${CHOICES:-"0 1 2 3 4 5 6 7 8 9"}                 # the 10 ensemble choices (fit-to-fit axis)
  NULL_TOYS=${NULL_TOYS:-100}; OBS_TOYS=${OBS_TOYS:-100}   # PER CHOICE (Sean 2026-09-26: 100 Gaussian null + 100 obs)
  STD_TIME=${STD_TIME:-"11:59:00"}; LONG_TIME=${LONG_TIME:-"11:59:00"}; LONG_ENS=128   # CPU standard wall=12h; small batches fit

  for choice in $CHOICES; do
    # member seeds per (ENS, choice), read from that choice's member_seeds.txt (M=1 = pool draw)
    declare -A SEEDS
    for ens in $ENS_LIST; do
      SEEDS[$ens]=$(resolve_seeds "$ens" "$choice") \
        || { echo "ERROR resolving seeds for M=$ens choice=$choice (DIM=$DIM)" >&2; exit 1; }
    done
    if [[ "$INCLUDE_SINGLE" == "1" ]]; then
      SINGLE_CHOICE_SEED=$(resolve_seeds 1 "$choice") \
        || { echo "ERROR resolving single-flow seed for choice=$choice" >&2; exit 1; }
    fi

    for ntest in $NTEST_LIST; do
      mem=$(mem_for "$ntest"); b=${BATCH:-$(batch_for "$ntest")}

      # ── single model: one flow from the pool for THIS choice; ENS=1 ⇒ auto-frozen/point-null ──
      if [[ "$INCLUDE_SINGLE" == "1" ]]; then
        export MEMBER_SEEDS="$SINGLE_CHOICE_SEED"
        for calib in $CALIBS; do
          [[ "$calib" -eq 1 ]] && nt=$NULL_TOYS || nt=$OBS_TOYS
          submit "$(array_for "$calib" "$b")" "$STD_TIME" "$mem" "ENS=1,CHOICE=$choice,CALIBRATION=$calib,NTEST=$ntest,KERNEL_SIGMA=$SIGMA,TOYS_PER_JOB=$b,N_TOYS=$nt"
          n=$((n+1))
        done
      fi

      # ── ensembles ──
      for ens in $ENS_LIST; do
        export MEMBER_SEEDS="${SEEDS[$ens]}"
        [[ "$ens" -eq "$LONG_ENS" ]] && etime="$LONG_TIME" || etime="$STD_TIME"
        for mode in $MODES; do
          [[ "$mode" == "point" ]] && fix="true" || fix="false"
          for calib in $CALIBS; do
            [[ "$calib" -eq 1 ]] && nt=$NULL_TOYS || nt=$OBS_TOYS
            submit "$(array_for "$calib" "$b")" "$etime" "$mem" \
              "ENS=$ens,CHOICE=$choice,FIX_WIFI_WEIGHTS=$fix,CALIBRATION=$calib,NTEST=$ntest,KERNEL_SIGMA=$SIGMA,TOYS_PER_JOB=$b,N_TOYS=$nt"
            n=$((n+1))
          done
        done
      done
    done
    unset SEEDS
  done
  nchoice=$(echo $CHOICES | wc -w)
  echo "# $n array-submissions [DRYRUN=$DRYRUN]  (DIM=$DIM 10-fit campaign at σ=$SIGMA; $nchoice choices × [single+ENS×modes] × N_test × calib; null=$NULL_TOYS / obs=$OBS_TOYS toys PER CHOICE per cell, BATCHED ${BATCH:+BATCH=$BATCH; }~5–50 toys/task so total tasks « MaxJobCount; ENS=$LONG_ENS → $LONG_TIME, others → $STD_TIME)"

else
  echo "unknown SCAN=$SCAN (want campaign|sigma)" >&2; exit 1
fi
