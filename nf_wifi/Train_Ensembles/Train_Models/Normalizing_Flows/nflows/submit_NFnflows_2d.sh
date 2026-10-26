#!/bin/bash
#SBATCH --job-name=nf_nflows_2d
# EXTEND the 2D NF pool 128 -> 512 (for the 10-fit fit-to-fit campaign, P=512).
# Trains ONLY the remaining members 128..511 (the existing 0..127 are untouched;
# each member's --seed drives its own init + bootstrap, so 128..511 are genuine
# new independent pool members). MODEL_SEEDS is HARDCODED to 128 below so the
# output dir stays the existing seeds_128 dir (grow-in-place) rather than the
# array-count-derived seeds_384 -- see the gotcha note in submit_NFnflows.sh.
#SBATCH --array=128-511
# Measured MaxRSS on the 2D members is ~1.6G, so 4G is ~2.5x headroom (was 32G, a ~15x
# over-allocation the cluster admins flagged). 6h walltime is loose (members finish in
# ~10-25 min) but only --mem reserves node RAM, so that is the one that matters here.
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/logs/nf_%A_%a.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/logs/nf_%A_%a.err

# =============================
# Activate environment
# =============================
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models:$PYTHONPATH

# =============================
# USER PARAMETERS (2D: matches the git-recorded config the original 128 used)
# =============================
DATA_PATH="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/data/2d_gmm_toymodel/2d_gmm_skew_Ntrain100000_Ntest100000_seed42/data_train.npy"
NUM_FEATURES=2
MODEL_SEEDS=128   # HARDCODED pool label (NOT SLURM_ARRAY_TASK_COUNT) -> writes into the
                  # existing seeds_128 dir so 128..511 grow the pool in place to 512.

# Model hyperparameters (identical to the original 2D run)
N_EPOCHS=1001
LR=1e-5          # match the grid search that selected this architecture
BATCH_SIZE=512
HIDDEN_FEATURES=64
NUM_BLOCKS=4
NUM_BINS=8
NUM_LAYERS=4

NFLOWS_DIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows"
ESTIMATOR="${NFLOWS_DIR}/EstimationNFnflows.py"
BASE_DIR="${NFLOWS_DIR}/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail"

# Trial dir (num_events from the data; deterministic -> identical across all tasks)
NUM_EVENTS=$(python -c "import numpy as np; print(np.load('${DATA_PATH}').shape[0])")
TRIAL_DIR="${BASE_DIR}/N_${NUM_EVENTS}_dim_${NUM_FEATURES}_seeds_${MODEL_SEEDS}_${NUM_LAYERS}_${NUM_BLOCKS}_${HIDDEN_FEATURES}_${NUM_BINS}"
mkdir -p "${TRIAL_DIR}" "${BASE_DIR}/logs"

# =============================
# Train this ensemble member
#   -> writes <TRIAL_DIR>/model_{SLURM_ARRAY_TASK_ID:03d}/model.pth (+ info.json).
# =============================
echo "SLURM job id: $SLURM_JOB_ID, array task: $SLURM_ARRAY_TASK_ID / pool ${MODEL_SEEDS}(->512)"
echo "Trial dir: ${TRIAL_DIR}"

python "${ESTIMATOR}" \
    --data_path "${DATA_PATH}" \
    --outdir "${TRIAL_DIR}" \
    --seed "${SLURM_ARRAY_TASK_ID}" \
    --n_epochs "${N_EPOCHS}" \
    --learning_rate "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --hidden_features "${HIDDEN_FEATURES}" \
    --num_blocks "${NUM_BLOCKS}" \
    --num_bins "${NUM_BINS}" \
    --num_layers "${NUM_LAYERS}" \
    --num_features "${NUM_FEATURES}"
