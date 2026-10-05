#!/bin/bash
#SBATCH --job-name=LRT
#SBATCH --array=0-99
#SBATCH --time=11:59:00
#SBATCH --mem=12G          # peak RSS observed ~8 G on these (<=100k) jobs; 12 G = ~1.5x margin
#SBATCH --ntasks=1
#SBATCH --account=gpu_gres
#SBATCH --partition=qgpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -o /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/results/logs/%x-%A_%a.out
#SBATCH -e /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/results/logs/%x-%A_%a.err
# Resource defaults above are GPU (qgpu,gpu / gpu_gres / --gres=gpu:1) because NF (torch) needs
# the GPU. KERNELS run on CPU (density eval is single-threaded numpy; a GPU would sit idle and
# throttle the campaign to the cluster's ~16 GPUs). For a KERNELS run, OVERRIDE back to CPU:
#   --partition=standard --account=t3 --gres=none
# (run_lrt_scan.sh drives the kernels scan -> add the CPU override there before a kernels campaign.)

set -euo pipefail

# ─── Scenario × Model (EDIT THESE) ───────────────────────────────────
# TWO orthogonal switches select the run:
#   MODEL    = kernels | nf                              — the model type (its own axis).
#   SCENARIO = 1model_2d | ens_2d | 1model_4d | ens_4d   — progression stage × dimension.
# The single-model preset is just member seed000 of the matching ensemble dir (NENSEMBLE=1),
# so 1model and ens are nested — only OUT_BASE differs. Kernel paths use the seed-42-aligned
# data (2d_gmm / 4d_embedding_qcd).
#   1model_2d / 1model_4d : single model (ensemble-of-one, frozen). NENSEMBLE=1 -> LRT.py
#       synthesizes w=[1.0], cov=0x0 (no wifi fit); W_PATH/W_COV_PATH stay empty and are NOT
#       passed; forces FIX_WIFI_WEIGHTS=true (frozen; below).
#   ens_2d / ens_4d       : ensemble. ONE training dir per dim; ENS={2,4,...} picks the size
#       (-e = first-ENS subset) AND the matching wifi fit (ensemblecomponents$ENS).
# Implemented MODEL:SCENARIO combos: kernels:{1model_2d,ens_2d,1model_4d,ens_4d}, nf:ens_2d
# (the NF leg — add more nf combos as they are produced). Anything else errors in the case below.
#   sbatch --export=ALL,SCENARIO=1model_2d,CALIBRATION=1 submit_LRT_toys.sh          # kernels (default)
#   sbatch --export=ALL,MODEL=nf,SCENARIO=ens_2d,CALIBRATION=1 submit_LRT_toys.sh    # nf
SCENARIO=${SCENARIO:-1model_2d}
MODEL=${MODEL:-kernels}

CALIBRATION=${CALIBRATION:-1}              # 1 = null toys (SIR + calib pool)  |  0 = observed (target data)
# FIX_WIFI_WEIGHTS=true -> frozen (w=w_hat); default false -> constrained (N(w_hat,Sigma) prior).
FIX_WIFI_WEIGHTS=${FIX_WIFI_WEIGHTS:-false}   # respects sbatch --export override

NTEST=${NTEST:-100000}
# M = # numerator kernels (--n_kernels). NPLM rule (Grosso-Letizia 2408.12296): M >= sqrt(N_test)
# (=316 for 100k, 447 for 200k). Default 300 = principled for the 100k base (old 500 was an
# overshoot per the M_sigma memory); held FIXED across the Ntest scan (not scaled) so DOF_eff is
# comparable. M is in the run_tag (_M%i_) so different M land in DISTINCT dirs -> no silent mixing.
# (run_lrt_scan.sh passes N_KERNELS=$M explicitly, so this default only affects ad-hoc launches.)
N_KERNELS=${N_KERNELS:-300}
# Z = ∫ f_ens(x)·exp(Σ b_j G_j(x)) dx is a genuine 2-SAMPLE test: the integral is estimated from
# a REFERENCE sample (importance-sampled from q). N_REF = # reference events, the 2-sample partner
# of NTEST. DEFAULT 1:1 (N_REF=NTEST) — Sean's choice (Slack, 2026-08): more N_ref only pulls the
# already-asymptotic null closer to chi2, but we CALIBRATE WITH TOYS anyway, so 1:1 is enough;
# extra N_ref just costs compute (esp. NF flow.sample x128). Bump (>>NTEST) only if a CLOSURE
# needs MC-Z suppression. The N_ref/N_test ratio is recorded in the run_tag (_Nrefx%g).
N_REF=${N_REF:-$NTEST}
FIRSTSEED=12345
# MEMBER_SEEDS: comma-separated member indices to load, IN ORDER (last=norm), passed to
# LRT.py --member_seeds. REQUIRED for the uniform/pinned wifi (the ensemble is a NON-first-k
# subset, so the models must match the seeds the wifi fit used). Resolve with
# select_member_seeds.py --rng_seed <same as the wifi fit>. Empty = first-k (legacy).
MEMBER_SEEDS=${MEMBER_SEEDS:-}

# CHOICE: 10-fit ensemble-choice index (0..9), used ONLY by the *_uniform scenarios. It
# selects the choice<CHOICE>/ subdir of the wifi ensemblecomponents<M> dir (that choice's own
# member_seeds.txt + w_i_fitted + cov_w) AND adds a choice<CHOICE>/ level at the top of OUT_BASE
# so the 10 fits store side-by-side and the pooled 25/75 plot can glob choice*/<run_tag>/. The
# driver (run_nf_scan.sh) reads choice<CHOICE>/member_seeds.txt and passes it as MEMBER_SEEDS.
# Empty = legacy single-draw layout (non-uniform scenarios).
CHOICE=${CHOICE:-}

# ─── Numerator knobs (EDIT / OVERRIDE AT SUBMIT TIME) ────────────────
# The numerator is the NPLM exp-tilt  f_num = f_ens*exp(Σ b_j G_j)/Z  (log-space, clean chi2;
# matches LRT_one_model + classifier). FIX_WIFI_WEIGHTS=true => frozen w (convex); default
# (false) => constrained (profile w under the N(w_hat,Sigma) prior).
#   sbatch --export=ALL,FIX_WIFI_WEIGHTS=true,CALIBRATION=1 submit_LRT_toys.sh
KERNEL_SIGMA=${KERNEL_SIGMA:-0.4}   # embedding-dependent: 0.4 (2D & old-4D) | 0.7 (gaussian-4D)
LAM_PERT=${LAM_PERT:-1.0}           # L2 ridge on tilt coeffs b (= one-model/classifier)
CLIP_B=${CLIP_B:-3}                 # box |b_j|<=CLIP_B on tilt coeffs. NON-EMPTY BY DEFAULT (crash fix):
                                    # clip=None makes fit_nplm_tilt use scipy trust-exact, which Choleskys
                                    # the tilt Hessian; on a ~0.1% tilt runaway (separation) H goes non-finite
                                    # -> cho_solve "infs or NaNs" CRASH (job dies, writes nothing) before the
                                    # runaway guard can flag it. Setting clip switches fit_nplm_tilt to L-BFGS-B
                                    # (no Cholesky -> crash impossible) AND bounds b. Good fits have max|b|<=2.9
                                    # (p99=1.23) so 3 sits above them -> bounds runaways, distorts no good fit.
                                    # ⚠ clip CHANGES the estimator: keep it FIXED across a config's null+test;
                                    # do NOT mix clipped T's into an unclipped null/test dist. empty=off (validation).
Z_MODE=${Z_MODE:-sample}            # Z: sample (importance from q; any d, DEFAULT) | grid (2D cross-check)
GRID_POINTS=${GRID_POINTS:-300}     # [grid] points per dimension
GRID_PAD=${GRID_PAD:-0.2}           # [grid] padding beyond data range
W_COV_SCALE=${W_COV_SCALE:-1.0}     # [constrained] diagnostic scale on Sigma_w; <1 -> frozen limit (separate out dir)
# 4D: set Z_MODE=sample + (usually) FIX_WIFI_WEIGHTS=true; the null is self-built (SIR-from-q).

# ─── Per-scenario paths (EDIT THESE) ─────────────────────────────────
REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
PY="$REPO_ROOT/LRT/LRT_rejnull.py"

# defaults; the MODEL case sets env, each scenario overrides its paths (empties stay empty -> not passed)
MODEL_TYPE="$MODEL"; NENSEMBLE=""; PYBIN="python"   # PYBIN: NF (MODEL case) overrides to nplm_env absolute python
ENSEMBLE_DIR=""; NF_TRAIN_DIR=""; ARCH_CONFIG=""
W_PATH=""; W_COV_PATH=""; TARGET_DATA=""; OUT_BASE=""
# TARGET_TRUTH: observed run (calib=0) draws FRESH from this analytic truth each toy
# instead of bootstrapping TARGET_DATA. Set by the 2D scenarios (same truth for kernels
# and nf). Leave empty for 4D (no analytic truth -> bootstrap the finite holdout).
TARGET_TRUTH=""

# MODEL (kernels | nf) is orthogonal to the 1model/ens × 2d/4d scenario, so it is its OWN
# switch: it sets the conda env (and, for nf, the absolute python + the env-block
# LD_LIBRARY_PATH skip below). The SCENARIO case then fills in the model-specific paths.
case "$MODEL" in
  kernels) CONDA_ENV=kernels_env ;;
  nf)      CONDA_ENV=nplm_env
           PYBIN="/work/gbadarac/miniforge3/envs/nplm_env/bin/python" ;;
  *) echo "Unknown MODEL=$MODEL (expected: kernels | nf)"; exit 1 ;;
esac

# Only the MODEL:SCENARIO combos below are implemented (kernels: all four; nf: ens_2d).
case "$MODEL:$SCENARIO" in
  kernels:1model_2d)
    NENSEMBLE=1
    # -e 1 uses seed000 of the 2d_gmm ensemble dir as the single model; no wifi files (synthesized)
    ENSEMBLE_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_gmm/N_100000_dim_2_kernels_SparKer_models1_L5_K80_M300_Nboot100000_lr0.05_clip_10000000_joint_norm_wf0.15-0.035"
    TARGET_TRUTH="2d_gmm_skew"   # observed: draw fresh from the analytic truth each toy (no data file)
    OUT_BASE="$REPO_ROOT/LRT/results/kernels/2d_single_model"
    ;;

  kernels:ens_2d)
    # 2d_gmm (seed-42 aligned). ONE 128-member training dir serves every size; ENS picks how
    # many members (-e = first-ENS subset) AND the matching wifi fit (ensemblecomponents$ENS).
    NENSEMBLE=${ENS:-64}
    ENSEMBLE_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_gmm/N_100000_dim_2_kernels_SparKer_models128_L5_K80_M300_Nboot100000_lr0.05_clip_10000000_joint_norm_wf0.15-0.035"
    WIFI_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernels/2d_toymodel/N_100000_dim_2_kernels_SparKer_models128_L5_K80_M300_Nboot100000_lr0.05_clip_10000000_joint_norm_wf0.15-0.035_2d_gmm_ensemblecomponents${NENSEMBLE}"
    W_PATH="$WIFI_DIR/w_i_fitted.npy"
    W_COV_PATH="$WIFI_DIR/cov_w.npy"
    TARGET_TRUTH="2d_gmm_skew"   # observed: draw fresh from the analytic truth each toy (no data file)
    OUT_BASE="$REPO_ROOT/LRT/results/kernels/2d_ensemble"
    ;;

  nf:1model_2d)
    NENSEMBLE=1
    # -e 1 uses seed000 of the 2d_gmm ensemble dir as the single model; no wifi files (synthesized)
    ENSEMBLE_DIR="$REPO_ROOT//work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_seeds_1_4_4_64_8"
    TARGET_TRUTH="2d_gmm_skew"   # observed: draw fresh from the analytic truth each toy (no data file)
    OUT_BASE="$REPO_ROOT/LRT/results/kernels/2d_single_model"
    ;;

  nf:ens_2d)
    # NF recipe (validated 08-17): the conda env + absolute python are set in the MODEL case
    # above, and the env block below SKIPS LD_LIBRARY_PATH for nf (prepending env/lib breaks
    # torch _C). IS loader: model_%03d/model.pth under NF_TRAIN_DIR (f_i.pth retired); null via
    # SIR-from-q (flow.sample). SIZE SCAN: set ENS + the matching wifi ensemblecomponents
    # {2,4,...,128} (smaller = first-k subset of the same 128-member training dir).
    NENSEMBLE=${ENS:-128}
    NF_TRAIN_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_seeds_128_4_4_64_8"
    ARCH_CONFIG="$NF_TRAIN_DIR/architecture_config.json"
    NF_WIFI_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/2d_toymodel/N_100000_dim_2_seeds_128_4_4_64_8_2d_bimodal_gaussian_heavy_tail_ensemblecomponents${NENSEMBLE}"
    W_PATH="$NF_WIFI_DIR/w_i_fitted.npy"
    W_COV_PATH="$NF_WIFI_DIR/cov_w.npy"
    # Same 2D truth as kernels (identical DGP) — observed draws fresh from it, no data file.
    TARGET_TRUTH="2d_gmm_skew"   # observed: draw fresh from the analytic truth each toy (no data file)
    OUT_BASE="$REPO_ROOT/LRT/results/nf/2d_ensemble"
    ;;

  nf:ens_2d_uniform)
    # 10-fit NF campaign (2D toy) on the 512-flow pool. CHOICE (0..9) picks one of the 10
    # ensemble choices: that choice's members (member_seeds.txt), self-calibrated weights and
    # covariance ALL live in ensemblecomponents<M>/choice<CHOICE>/, and outputs go under a
    # matching choice<CHOICE>/ so the 10 fits sit side-by-side for the pooled 25/75 plot. The
    # driver (run_nf_scan.sh) reads choice<CHOICE>/member_seeds.txt and passes it as MEMBER_SEEDS
    # so LRT.py loads the SAME members, in the SAME order, that the wifi fit used. Single model
    # (ENS=1) = one flow drawn from the pool per choice (driver passes MEMBER_SEEDS); no wifi
    # files (M=1 synthesizes w=[1.0]).
    NENSEMBLE=${ENS:-128}
    NF_TRAIN_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_seeds_512_4_4_64_8"
    ARCH_CONFIG="$NF_TRAIN_DIR/architecture_config.json"
    TARGET_TRUTH="2d_gmm_skew"   # observed: draw fresh from the analytic truth each toy (no data file)
    OUT_BASE="$REPO_ROOT/LRT/results/nf/2d_uniform_ensemble/choice${CHOICE}"
    if [[ "$NENSEMBLE" -gt 1 ]]; then
      NF_WIFI_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF_uniform/2d_toymodel/N_100000_dim_2_seeds_512_4_4_64_8_2d_bimodal_gaussian_heavy_tail_ensemblecomponents${NENSEMBLE}/choice${CHOICE}"
      W_PATH="$NF_WIFI_DIR/w_i_fitted.npy"
      W_COV_PATH="$NF_WIFI_DIR/cov_w.npy"
    fi
    ;;

  nf:ens_4d_uniform)
    # 10-fit NF campaign (4D JetClass QCD embedding) on the 512-flow pool; CHOICE mechanics as
    # in ens_2d_uniform (choice<CHOICE>/ holds member_seeds.txt + w_i_fitted + cov_w; outputs go
    # under a matching choice<CHOICE>/). Single model (ENS=1) = one flow from the pool per choice
    # (driver passes MEMBER_SEEDS; M=1 synthesizes w=[1.0]). No analytic truth (real embedding)
    # -> calib=0 bootstraps the held-out data_heldout_qcd.npy (disjoint from data_train).
    NENSEMBLE=${ENS:-64}
    NF_TRAIN_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/4_dim/4d_embedding_qcd/N_100000_dim_4_seeds_512_6_8_128_10"
    ARCH_CONFIG="$NF_TRAIN_DIR/architecture_config.json"
    TARGET_DATA="$REPO_ROOT/data/4d_embeddings/4d_embedding_qcd_Ntrain100000_Ntest100000_seed42/data_heldout_qcd.npy"
    OUT_BASE="$REPO_ROOT/LRT/results/nf/4d_uniform_ensemble/choice${CHOICE}"
    if [[ "$NENSEMBLE" -gt 1 ]]; then
      NF_WIFI_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF_uniform/4d_embedding/N_100000_dim_4_seeds_512_6_8_128_10_4d_embedding_qcd_ensemblecomponents${NENSEMBLE}/choice${CHOICE}"
      W_PATH="$NF_WIFI_DIR/w_i_fitted.npy"
      W_COV_PATH="$NF_WIFI_DIR/cov_w.npy"
    fi
    ;;

  *)
    echo "Unsupported MODEL/SCENARIO combo: MODEL=$MODEL SCENARIO=$SCENARIO"
    echo "  implemented -> kernels: 1model_2d | ens_2d ; nf: ens_2d | ens_2d_uniform | ens_4d_uniform"
    exit 1 ;;
esac

# The 10-fit uniform scenarios REQUIRE CHOICE (0..9): it indexes the wifi choice<k>/ subdir and
# the choice<k>/ output level. Guard against an empty CHOICE landing outputs in a ".../choice/"
# dir or reading a nonexistent wifi path. (Non-uniform/legacy scenarios don't set it.)
if [[ "$SCENARIO" == *_uniform && -z "$CHOICE" ]]; then
    echo "ERROR: SCENARIO=$SCENARIO requires CHOICE=<0..9> (the 10-fit ensemble-choice index)"; exit 1
fi

# A single model (NENSEMBLE=1) has no wifi weights to profile, so only the frozen fit is valid.
if [[ "$NENSEMBLE" -eq 1 ]]; then
    [[ "$FIX_WIFI_WEIGHTS" != "true" ]] && echo "[single model] forcing FIX_WIFI_WEIGHTS=true (frozen)"
    FIX_WIFI_WEIGHTS=true
fi

# ─── Environment ─────────────────────────────────────────────────────
. /work/gbadarac/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Pin BLAS/OMP to 1 thread. SLURM gives each job 1 CPU, but numpy's OpenBLAS otherwise
# spawns 64 threads -> when many jobs land on the same CPU node they oversubscribe it
# (dozens x 64 threads on 128 cores), thrash, and the heavy toys blow past the 8h wall
# (the calib=1 TIMEOUTs on constr/200k e128). 1 thread = 1 core = no contention; the
# kernel eval is a single-threaded python loop anyway, so this doesn't slow the fit.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [[ "$MODEL_TYPE" == "nf" ]]; then
  # NF recipe (validated 08-17): use nplm_env ABSOLUTE python ($PYBIN) and DO NOT touch
  # LD_LIBRARY_PATH — prepending env/lib breaks torch _C (ByteStorageBase). torch 1.11+cu115
  # in nplm_env finds CUDA via its bundled libs. (conda activate can't change `which python`.)
  :
else
  export LD_LIBRARY_PATH="/work/gbadarac/miniforge3/envs/nplm_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  CUDA_PATH=$("$PYBIN" - <<'PY'
import torch; print(torch.version.cuda or "")
PY
)
  if [[ -n "$CUDA_PATH" ]]; then
    export CUDA_HOME=/usr/local/cuda-$CUDA_PATH
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  fi
fi
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# ─── Run ─────────────────────────────────────────────────────────────
mkdir -p "$OUT_BASE" "$REPO_ROOT/LRT/results/logs"
cd "$REPO_ROOT"

# Toy batching: ONE SLURM array task runs TOYS_PER_JOB toys sequentially. This keeps the total
# job count far below the cluster MaxJobCount — one-job-per-toy is ~5x the cap and hogs the whole
# shared job table. The array index maps to a contiguous toy_id range [START, END]; N_TOYS (total
# toys for this cell,calib) clamps the final partial batch. toy_id -> SEED is unchanged, so the
# results are IDENTICAL to the unbatched run. Defaults (TOYS_PER_JOB=1) reproduce old behaviour.
TOYS_PER_JOB=${TOYS_PER_JOB:-1}
N_TOYS=${N_TOYS:-}
ARRAY_IDX=${SLURM_ARRAY_TASK_ID:-0}
START=$(( ARRAY_IDX * TOYS_PER_JOB ))
END=$(( START + TOYS_PER_JOB - 1 ))
if [[ -n "$N_TOYS" && "$END" -ge "$N_TOYS" ]]; then END=$(( N_TOYS - 1 )); fi

echo "[$(date)] MODEL_TYPE=${MODEL_TYPE}, CALIBRATION=${CALIBRATION}, array_idx=${ARRAY_IDX}, toys ${START}..${END} (batch=${TOYS_PER_JOB}, N_TOYS=${N_TOYS:-unset}), host=${HOSTNAME}"

# Base command shared by every toy in this task (everything EXCEPT the per-toy --toy_id / -s).
BASE_CMD=("$PYBIN" -u "$PY"
  --model_type    "$MODEL_TYPE"
  --out_base      "$OUT_BASE"
  -n              "$NTEST"
  -e              "$NENSEMBLE"
  -c              "$CALIBRATION"
  --n_kernels     "$N_KERNELS"
  --kernel_sigma  "$KERNEL_SIGMA"
  --lam_pert      "$LAM_PERT"
  --z_mode        "$Z_MODE"
  --grid_points   "$GRID_POINTS"
  --grid_pad      "$GRID_PAD"
  --n_ref         "$N_REF"
)
# wifi weights: only for a real ensemble (NENSEMBLE>1); a single model synthesizes them
[[ -n "$W_PATH" ]] && BASE_CMD+=(--w_path "$W_PATH" --w_cov_path "$W_COV_PATH")
if [[ "$MODEL_TYPE" == "kernels" ]]; then
    BASE_CMD+=(--ensemble_dir "$ENSEMBLE_DIR" --seed_format "seed%03d")
elif [[ "$MODEL_TYPE" == "nf" ]]; then
    BASE_CMD+=(--nf_train_dir "$NF_TRAIN_DIR" --arch_config "$ARCH_CONFIG")
fi
[[ -n "$CLIP_B" ]] && BASE_CMD+=(--clip_b "$CLIP_B")
[[ -n "$MEMBER_SEEDS" ]] && BASE_CMD+=(--member_seeds "$MEMBER_SEEDS")
[[ "$W_COV_SCALE" != "1.0" ]] && BASE_CMD+=(--w_cov_scale "$W_COV_SCALE")
[[ "$FIX_WIFI_WEIGHTS" == "true" ]] && BASE_CMD+=(--fix_wifi_weights)
# calib=1 null is self-contained (SIR-from-q). calib=0 observed: 2D toy draws FRESH from the
# analytic truth each toy (--target_truth: genuine oversampling); 4D bootstraps --target_data.
if [[ "$CALIBRATION" -eq 0 ]]; then
  if [[ -n "$TARGET_TRUTH" ]]; then BASE_CMD+=(--target_truth "$TARGET_TRUTH")
  else BASE_CMD+=(--target_data "$TARGET_DATA"); fi
fi

# Run every toy in the batch sequentially. A failing toy is logged and does NOT abort the rest
# (the `if` exempts it from set -e); the task exit code is the worst toy's, so a fully-clean batch
# exits 0 and any failure is visible in sacct.
RC=0
for TOY_ID in $(seq "$START" "$END"); do
  SEED=$((FIRSTSEED + TOY_ID + 1))
  CMD=("${BASE_CMD[@]}" --toy_id "$TOY_ID" -s "$SEED")
  echo "[$(date)] --- toy ${TOY_ID} (seed ${SEED}) ---"
  echo "Running: ${CMD[*]}"
  if "${CMD[@]}"; then rc=0; else rc=$?; fi
  echo "[$(date)] toy ${TOY_ID} exit ${rc}"
  [[ "$rc" -ne 0 ]] && RC=$rc
done
echo "[$(date)] array_idx ${ARRAY_IDX} (toys ${START}..${END}) finished; worst exit ${RC}"
exit $RC
