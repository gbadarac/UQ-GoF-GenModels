# Shared settings for the SLURM drivers. Source this file; override any variable in the environment first.
# Paper settings for every arm (both benchmarks): J = 500 kernels of one raw width, EM with one k-means++ start,
# covariance floor 1e-8, bootstrap refits with the same estimator, 100 observed test sets per (fit, N_test).
SNAP=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python3}
export SIGMA_RUNS=${SIGMA_RUNS:-$SNAP/runs}
PARTITION=${PARTITION:-shared}                     # your cluster's partition list
PAPER_ENV="SIGMA_J=500 SIGMA_REG_COVAR=1e-8 SIGMA_EM_TOL=1e-6 SIGMA_NINIT=1 SIGMA_BOOT_NINIT=1 SIGMA_BANK_DTYPE=float32 SIGMA_CHUNK=50000 SIGMA_NSETS=100"
NT_TOY2D="1000 10000 20000 50000 100000 200000"
NT_JETS4D="100 200 1000 2000 10000 20000 50000 100000"
mkdir -p "$SIGMA_RUNS/logs"

# submit <name> <array spec> <cpus> <mem> <time> <dependency or ""> <command>   -> prints the job id
submit () {
  local name=$1 arr=$2 cpus=$3 mem=$4 time=$5 dep=$6; shift 6
  sbatch --parsable -J "$name" -p "$PARTITION" -n 1 -c "$cpus" --mem="$mem" -t "$time" --array="$arr" \
         ${dep:+--dependency=afterok:$dep} -o "$SIGMA_RUNS/logs/${name}_%A_%a.out" \
         --wrap "export OMP_NUM_THREADS=$cpus PYTHONUNBUFFERED=1; cd $SNAP/sigma_gof; $*"
}
