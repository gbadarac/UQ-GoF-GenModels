# Uncertainty-aware density estimation and goodness-of-fit with normalizing flows

This is the working pipeline for the normalizing-flow side of *An end-to-end framework for uncertainty-aware validation
of generative models*: ensembles of normalizing flows, the w_i f_i weight fit with its sandwich covariance, and the
goodness-of-fit test with point and Gaussian composite nulls. A Sparker-kernel density estimator is kept alongside as an
alternative to the flows; it is not used in the paper.

The paper's App. B code is a frozen copy of this pipeline, on the
[`paper` branch](https://github.com/gbadarac/UQ-GoF-GenModels/tree/paper) (`nf_wifi/`). Changes made here are not
carried over automatically.

---

## Table of contents

1. Overview
2. Repository layout
3. Environments
4. Data
5. Step-by-step pipeline
   1. Step 1 — Train NF ensemble
   2. Step 2 — Fit $w$ and propagate uncertainty
   3. Step 3 — Learned one-sample GoF likelihood–ratio test
6. Outputs and analysis
7. Sparker kernels
8. Running elsewhere

---

## 1. Overview

The pipeline has three components.

1. **Density estimation with Normalizing Flows**  
   Ensemble of normalizing flows trained on bootstrap replicas and independent initializations to capture data and
   optimization variability. Ensembles of $M$ members are drawn from a pool of 512 flows.

2. **Frequentist UQ with $w_i f_i$ ensembles**  
   Weighted mixture $\hat f(x)=\sum_i \hat w_i f_i(x)$. Weights come from maximum likelihood with a normalization
   constraint ($\sum_i w_i = 1$; individual weights may be negative). Covariance of $\hat w$ via a sandwich estimator,
   propagated to a pointwise band $\hat f \pm \sigma_{\hat f}$.

3. **Learned one-sample GoF likelihood–ratio test**  
   Uncertainty on $\hat w$ is propagated inside the test: the composite null profiles $w$ under the prior
   $\mathcal N(\hat w,\mathrm{Cov}(\hat w))$, while the point null fixes $w=\hat w$. The alternative tilts the ensemble
   multiplicatively, $f(x)\propto f_{\rm ens}(x;w)\,\exp\big(\sum_j b_j G_j(x)\big)$, with Gaussian kernels $G_j$.

---

## 2. Repository layout

```text
UQ-GoF-GenModels/
├─ data/                                  # Shared inputs; self-reproducing generators
│  ├─ 2d_gmm_toymodel/                    # 2D GMM+skew "heavy tail" target + generator
│  └─ 4d_embeddings/                      # 4D embedding datasets (JetClass): cache + held-out set
├─ Train_Ensembles/                       # Orchestration for NF ensembles (arrays, logs)
│  └─ Train_Models/                       # Train individual ensemble members
│     ├─ utils_flows.py                   # Flow construction, shared by every step
│     ├─ Normalizing_Flows/               # NF training (nflows backend)
│     │  └─ nflows/                       # nflows-based NF training, grid search, checks
│     └─ Sparker_kernels/                 # Kernel-based density estimation
├─ Uncertainty_Modeling/                  # Weight fitting (w_i), sandwich covariance, propagation
│  └─ wifi/                               # w_i f_i frequentist ensembles and covariance
│     ├─ utils_NF_wifi.py                 # NF ensemble evaluation, propagation, marginal plots
│     ├─ utils_kernel_wifi.py             # Kernel-ensemble counterpart
│     └─ Fit_Weights/                     # MLE for w, sandwich covariance, propagation
│        ├─ select_member_seeds.py        # Draws the members of each ensemble choice
│        ├─ fit_ensemble_weights.py       # Unified weight-fitting script (MODEL_TYPE toggle)
│        └─ submit_fit_weights.sh         # SLURM submission (set MODEL_TYPE=nf|kernels)
├─ LRT/                                   # One-sample learned LRT with uncertainty-aware reference
│  ├─ LRT_rejnull.py                      # Unified LRT script (--model_type kernels|nf)
│  ├─ submit_LRT_toys.sh                  # SLURM submission (set MODEL=kernels|nf)
│  └─ run_nf_scan.sh                      # Submits the full NF campaign
├─ shared/                                # Shared library used across pipeline stages
│  └─ Sparker_utils/                      # Kernel utilities (LRTGOFutils_v2, GENutils, SPARKutils, …)
├─ envs/                                  # Conda/pip environment files
└─ README.md
```

Trained flows, fit results and test results are written inside this tree but are not tracked.

---

## 3. Environments

| Environment | Used for |
|---|---|
| `nf_env` | flow training and weight fits |
| `nplm_env` | the test with flows; analysis and plots |
| `kernels_env` | the Sparker-kernel pipeline; also works for analysis and plots |

Create them with `conda env create -f envs/<name>.yml`. The SLURM scripts activate the right one themselves. Do not use
`nf_env` for analysis: its scipy and numpy versions conflict.

---

## 4. Data

**2D toy.** `python generate_2d_gaussian_heavy_tail_target_data.py` (in `data/2d_gmm_toymodel/`) writes 100k training
events (seed 42) and 100k test events (seed 43) to `2d_gmm_skew_Ntrain100000_Ntest100000_seed42/`, the same split Sean's
`wifi_better_basis` pipeline uses. `plot_2d_gaussian_heavy_tail_target.ipynb` plots it. The toy has an analytic truth,
so the test draws its observed sets fresh from it.

**4D jets.** The raw input is a pool of 20 × 100k QCD jets (`ZJetsToNuNu_*.npz`) in
`data/4d_embeddings/4d_embedding_data_JetClass/`, which is not tracked. In `data/4d_embeddings/`:

```bash
python make_qcd_cache.py --n_train 100000 --n_test 100000 --seed 42 \
    --benchmark 4d_embedding_qcd --data-dir 4d_embedding_data_JetClass   # disjoint 100k/100k train/test split
python carve_heldout_qcd.py                                               # data_heldout_qcd.npy: the ~1.9M events not used for training
```

Both write to `4d_embedding_qcd_Ntrain100000_Ntest100000_seed42/`. The test's observed sets are drawn from
`data_heldout_qcd.npy`. Without `--benchmark` and `--data-dir`, `make_qcd_cache.py` builds the alternative
`4d_gaussian_embedding_qcd` cache instead. `inspect_4d_embedding_qcd.py` makes inspection plots of an embedding.

---

## 5. Step-by-step pipeline

### 5.1 Step 1 — Train NF ensemble

Each array task of `submit_NFnflows_2d.sh` (2D) or `submit_NFnflows.sh` (4D) trains one flow with
`EstimationNFnflows.py`. Its seed sets both the initialization and a bootstrap replica of `data_train.npy`; the replica
is split 80/20 into training and validation, with early stopping after 10 epochs without improvement.

```bash
cd Train_Ensembles/Train_Models/Normalizing_Flows/nflows
sbatch submit_NFnflows_2d.sh     # or submit_NFnflows.sh for 4D
```

| | 2D toy | 4D jets |
|---|---|---|
| layers, blocks, hidden, bins | 4, 4, 64, 8 | 6, 8, 128, 10 |
| learning rate, batch size | 1e-5, 512 | 1e-5, 512 |
| pool | 512 flows | 512 flows |

Each pool lands in `EstimationNFnflows_outputs/<d>_dim/<dataset>/N_<N>_dim_<d>_seeds_<P>_<layers>_<blocks>_<hidden>_<bins>/`,
with `architecture_config.json` and one `model_NNN/` per flow (`model.pth`, `info.json`, `losses.npz`). The `seeds_<P>`
label is set in the submit script (`submit_NFnflows.sh` uses the array size, `submit_NFnflows_2d.sh` hardcodes it), so
check it before training more flows into an existing pool.

`submit_NFgridsearch.sh` and `eval_gridsearch.py` rank candidate architectures by held-out NLL.
`plot_nf_marginals.py --trial_dir <pool>` and `plot_nf_loss.py --trial_dir <pool>` are quick per-flow checks.

### 5.2 Step 2 — Fit $w$ and propagate uncertainty

`submit_fit_weights.sh` (in `Uncertainty_Modeling/wifi/Fit_Weights/`) is a 10-task array. Task k is ensemble choice k:
M flows drawn at random from the pool by `select_member_seeds.py --mode uniform --rng_seed k`, then fitted by
`fit_ensemble_weights.py`. The fit keeps the density positive at every data point; the covariance is the sandwich
estimator.

```bash
cd Uncertainty_Modeling/wifi/Fit_Weights
sbatch --export=ALL,NDIM=2,N_WIFI=32 submit_fit_weights.sh   # MODEL_TYPE=nf|kernels, NDIM=2|4, N_WIFI=M, SELECTION=uniform
```

With `N_WIFI=1` there is no fit: `plot_single_seed_marginal.py` plots the single flow without a band.
`plot_10fit_aggregate.py` combines the 10 choices of each configuration into one marginal plot (median fit, Gaussian
weight-uncertainty band).

### 5.3 Step 3 — Learned one-sample GoF likelihood–ratio test

`LRT_rejnull.py` computes T = 2 (ℓ_NUM − ℓ_DEN) for one toy, with J Gaussian kernels G_j of width σ centred on the first
J events of the toy.
- **Point null:** w fixed at ŵ (`FIX_WIFI_WEIGHTS=true`). A single flow (`ENS=1`) always uses it.
- **Composite null:** w profiled under the prior N(ŵ, Σ_w) (`FIX_WIFI_WEIGHTS=false`, the default).
- **Null toys** (`CALIBRATION=1`): N_test events drawn from the ensemble by capped rejection sampling from the
  equal-weight member mixture. For the composite null, w is first drawn from N(ŵ, Σ_w).
- **Observed toys** (`CALIBRATION=0`): fresh draws from the analytic truth (2D), or draws from
  `data_heldout_qcd.npy` (4D).
- **Normalization:** estimated by importance sampling, with N_ref = N_test reference events.

`LRT/run_nf_scan.sh` submits the whole campaign through `submit_LRT_toys.sh`: 10 choices × N_test × {single flow,
M = 32, M = 128} × {point, composite} × {null, observed}, with 100 null and 100 observed toys per choice. It only prints
the `sbatch` commands unless `DRYRUN=0`.

```bash
cd LRT
DRYRUN=0 DIM=2d ./run_nf_scan.sh                 # 2D campaign (σ = 0.4)
DRYRUN=0 DIM=4d SCAN=sigma ./run_nf_scan.sh      # 4D: σ scan on a single flow
DRYRUN=0 DIM=4d SIGMA=0.53 ./run_nf_scan.sh      # 4D campaign at the chosen σ
```

`CHOICES`, `NTEST_LIST`, `ENS_LIST`, `MODES` and `CALIBS` select a subset (see the script header).

**Settings used in the paper**

| Variable (`submit_LRT_toys.sh`) | Paper value | Meaning |
|---|---|---|
| `MODEL`, `SCENARIO` | `nf`, `ens_2d_uniform` / `ens_4d_uniform` | model type and dataset |
| `ENS` | 1, 32, 128 | ensemble size M |
| `CHOICE`, `MEMBER_SEEDS` | 0–9, from `choice<k>/member_seeds.txt` | ensemble choice and its flows (`run_nf_scan.sh` fills both) |
| `N_KERNELS` | 500 | kernels J |
| `KERNEL_SIGMA` | 0.4 (2D), 0.53 (4D) | kernel width σ in raw data units |
| `LAM_PERT` | 1 | ridge λ on the tilt coefficients |
| `CLIP_B` | 3 | box \|b_j\| ≤ A |
| `N_REF` | N_test | reference events for the normalization |
| `TOYS_PER_JOB`, `N_TOYS` | — | toys per array task (memory and job-count setting) |

---

## 6. Outputs and analysis

- **Weight fits:** `Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF_uniform/<2d_toymodel|4d_embedding>/<pool>_<dataset>_ensemblecomponents<M>/choice<k>/`
  holds `member_seeds.txt` (tracked), `w_i_fitted.npy` (M weights), `cov_w.npy` ((M−1) × (M−1)), `loss_history.npy`
  and plots.
- **Test:** `LRT/results/nf/<2d|4d>_uniform_ensemble/choice<k>/<run_tag>/{calibration,test}/seed<N>/` holds
  `seed<N>_T.npy` (the statistic) and fit diagnostics (`seed<N>_fit_report.json`, tilt coefficients, kernel centres,
  fitted weights). The run tag encodes every setting, e.g. `NF_Nens128_Ntest100000_M500_W0.4_..._composite_null`, and
  is the same across the 10 choices so that they can be pooled.

```bash
python LRT/analyse_LRT_output.py --results_dir <run_tag dir> [--dof <n>] [--w_cov_path <cov_w.npy>]   # one configuration
python LRT/plot_pvalue_vs_ntest.py --base <results dir(s)> --out <fig.pdf>                            # median p-value vs N_test
python LRT/plot_pvalue_ladder_paper.py --dim 2d                                                       # paper figure, pooled over the 10 choices
python LRT/plot_Tdist_vs_ntest.py --base <results dir>                                                # null vs observed T
```

---

## 7. Sparker kernels

The kernel pipeline is kept but not used in the paper; only its 2D training outputs are on disk.
- **Training:** `Train_Ensembles/Train_Models/Sparker_kernels/submit_SparkerKernels.sh` runs `EstimationKernels.py` as a
  128-member array on the 2D toy, into `EstimationKernels_outputs/2_dim/2d_gmm/`.
- **Weights:** `submit_fit_weights.sh` with `MODEL_TYPE=kernels`.
- **Test:** `submit_LRT_toys.sh` with `MODEL=kernels` and `SCENARIO=1model_2d` or `ens_2d`. `LRT/run_lrt_scan.sh` drives
  the 2D scan and `LRT/make_pvalue_plots.sh` plots it.
- **Single-model test:** `LRT/LRT_kernels_1model_profiling.py` (with `submit_LRT_kernels_1model_profiling_toys.sh`)
  tests one kernel model by profiling the weights of its internal components instead of ensemble members.

---

## 8. Running elsewhere

The SLURM scripts are the drivers used on the ETH Zürich T3 cluster and contain its absolute paths. Before running
elsewhere, set:
- `REPO_ROOT` and the other `/work/gbadarac/MonoJet_NPLM/UQ-GoF-GenModels/...` paths;
- the conda paths (`/work/gbadarac/miniforge3`);
- the `#SBATCH` log paths, account and partition.
