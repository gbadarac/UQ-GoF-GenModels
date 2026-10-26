# SIGMA goodness-of-fit: code and data for the GMM figures

This is the code behind the Gaussian-mixture results of *An end-to-end framework for uncertainty-aware validation of
generative models*: the fits, the three null calibrations (bootstrap, Gaussian, point), the observed test statistics,
and the scripts that turn them into the paper's figures. The normalizing-flow appendix (App. B) is a separate
implementation, in `../nf_wifi/`.

It is a cleaned copy of the production code that made the figures, not a separate reimplementation (see
**Verification** below).

## Layout

| Path | Contents |
|---|---|
| `sigma_gof/` | the pipeline (flat Python modules) |
| `scripts/` | SLURM drivers for every run, and the exports that turn finished runs into `data/` |
| `data/` | everything the plots read, exported from the paper's runs |
| `plots/` | one script per figure; reads `data/`, writes `figures/` |
| `figures/` | the figures as produced by `make_figures.sh` |
| `environment.yml` | package versions used for the paper results |

## Making the figures (no cluster needed)

```bash
bash make_figures.sh          # all figures into figures/
```

| Paper figure | Command (in `plots/`) | Data |
|---|---|---|
| Fig. 4, 2D marginals with uncertainty bands | `python plot_marginals.py toy2d` | `data/marginals/toy2d_K64_fit0_bands.npz` |
| Fig. 5, 2D closure test | `python plot_ladders.py toy2d` | `data/pvalues/ladder_toy2d.json` |
| Fig. 7, test-statistic distributions (2D, K = 64, N_test = N_train) | `python plot_fig7.py 100k` | `data/fig7/toy2d_K64_statistics.npz` |
| Fig. 8, 4D marginals with uncertainty bands | `python plot_marginals.py jets4d` | `data/marginals/jets4d_K256_fit0_bands.npz` + the jet data (below) |
| Fig. 9, 4D closure test (K = 128–512) | `python plot_ladders.py jets4d` | `data/pvalues/ladder_jets4d.json` |
| Fig. 14 (App. C), Λ_max ablation | `python plot_lambda.py` | `data/pvalues/lambda_toy2d_K64.json` |
| Fig. 15 (App. D), null distributions, 4D K = 256 | `python plot_null_dists.py` | `data/null_dists/` |
| Fig. 16 (App. E), kernel width σ = 0.1 | `python plot_resolution.py` | `data/pvalues/resolution_toy2d.json` |

Fig. 8 draws held-out events from the 4D jet embedding, which is not included here. Point `SIGMA_JETS_DATA` at a
directory of `.npz` files that each hold an `embeddings` array (N × 4). They are concatenated in sorted file-name order
and split exactly as in the paper: a 1M-event test region and a training bank.

Fig. 2 (the one-dimensional Gaussian example) is not produced by this code.

### Data formats

- **`data/pvalues/*.json`**: one record per arm and N_test, keyed `<bench>_K<K>[_<variant>]_<N_test>`. Each record
  holds per-test-set p-values, `p_bootstrap`, `p_gaussian`, `p_point`, plus `fit_bootstrap` and `fit_gaussian` (the
  fit each test set belongs to) and `n_fits`. Error bars resample fits, because test sets of the same fit are not
  independent.
- **`data/fig7/toy2d_K64_statistics.npz`**: for each N_test, arrays of shape (fit, set) or (fit, toy):
  - `t_comp_obs_<N>`, `t_point_obs_<N>` (20 × 100): observed statistics;
  - `boot_comp_<N>` (20 × 100): bootstrap null;
  - `gauss_comp_<N>`, `point_null_<N>` (20 × 250): Gaussian and point nulls.
- **`data/null_dists/`**: the same quantities, pooled over the 10 fits of the 4D K = 256 arm, at N_test = 1k, 20k and 100k.
- **`data/marginals/*_bands.npz`**: fit 0's parameters (`theta`), 200 Gaussian draws θ̂ + W z (`gaussian_thetas`)
  and the 100 bootstrap refits (`bootstrap_thetas`).

## The pipeline

`sigma_gof/pipeline.py` runs one stage for one fit `b`. All settings are environment variables. Run directories are
`$SIGMA_RUNS/<bench>_K<K><SIGMA_SUFFIX>/m<b>/` (default `SIGMA_RUNS` = `runs/` next to this README).

| Stage | What it does |
|---|---|
| `fit --b B` | draws the training sample, fits the GMM by EM, builds the reference set, the Fisher whitening and the kernel dictionary; writes `model.npz` |
| `gaussian --b B --nte N` | Gaussian composite null (toys from θ̂ + W z) and point null (toys from θ̂) for one N_test |
| `bootstrap --b B --ntes N1 N2 …` | bootstrap null: per toy, a fresh training sample from the fit, a cold EM refit, the whole construction rebuilt at the refit; one refit serves every N_test |
| `observed --b B --nte N` | test statistics on the observed test sets (with `SIGMA_OBS_TONLY=1`, statistics only, no bank needed) |
| `fill --b B --nte N` | attach a finished bank to a t-only observed file (optional; the exports compute p-values themselves) |

`refits.py <run>/m00 i_lo i_hi` saves the bootstrap refits' parameters, for the marginal bands.

The Gaussian and bootstrap stages write banks with the same file names, so each runs in its own directory. The
scripts use `<run>` and `<run>_boot`, with the fit's `model.npz` linked into the latter.

### Settings used in the paper

| Variable | Paper value | Meaning |
|---|---|---|
| `SIGMA_BENCH` | `toy2d` / `jets4d` | benchmark |
| `SIGMA_K` | 16–128 (2D), 128–512 (4D) | mixture components |
| `SIGMA_J` | 500 | kernels in the dictionary |
| `SIGMA_KERNEL_SIGMA` | 0.4 (2D), 0.53 (4D); 0.1 in App. E | kernel width in raw data units |
| `SIGMA_LAM_MAX` | 0.008 | whitening cut Λ_max (App. C varies it) |
| `SIGMA_NINIT`, `SIGMA_BOOT_NINIT` | 1 | k-means++ EM starts for the fit and for every refit |
| `SIGMA_REG_COVAR` | 1e-8 | covariance floor |
| `SIGMA_EM_TOL` | 1e-6 | EM tolerance |
| `SIGMA_NSETS` | 100 | observed test sets per (fit, N_test) |
| `SIGMA_OBS_FRESH` | 1 (2D) | fresh observed sets per fit, drawn from the true density |
| `SIGMA_BANK_DTYPE`, `SIGMA_CHUNK` | float32, 50000 | memory settings |
| `SIGMA_NTRAIN` | 100000 | training-set size |
| `SIGMA_BOOT_LAMS` | (App. C only) | extra Λ_max arms served by the same bootstrap refits |
| `SIGMA_JETS_DATA` | — | directory of the 4D jet-embedding files |

The defaults in the code are not all the paper values. `scripts/common.sh` sets the paper values for every job.

Other fixed settings:
- ridge λ_α = 1 on the kernel amplitudes;
- box constant A = 10 (`CLIP_A`);
- reference set S = 2 × 10⁶ events, Fisher information on the first 10⁶;
- the slice used at each N_test is 10 N_test events.

## Reproducing the runs

```bash
export PARTITION=<your slurm partitions>  SIGMA_JETS_DATA=<jet data dir>
bash scripts/run_paper.sh          # all arms: fits, Gaussian/point and bootstrap nulls, observed statistics
# once the K = 64 (2D) and K = 256 (4D) fits exist:
bash scripts/run_lambda.sh         # Λ_max ablation (paired bootstrap arms)
bash scripts/run_refits.sh         # refits for the marginal bands
# when everything has finished:
cd scripts
python export_pvalues.py all       # -> data/pvalues/
python export_figure_data.py all   # -> data/fig7, data/null_dists, data/marginals
cd .. && bash make_figures.sh
```

Each arm is `scripts/run_arm.sh <bench> <K> <sigma>`. The bootstrap is the expensive part: 100 cold EM refits per
fit, and each 4D refit at K ≥ 512 takes hours. The 4D arms used 10 fits; the 2D arms used 20.

## Verification

- **Pipeline.** Run single-threaded on a small end-to-end case (2D, K = 8, N_train = 10⁴, J = 500, σ = 0.4), the
  cleaned code reproduces the original exactly. That covers the model file, every Gaussian, point and bootstrap bank
  (including an extra Λ_max arm), every observed statistic and p-value, and the saved refits. Multithreaded runs
  differ at the 10⁻⁶ relative level, from thread nondeterminism in scikit-learn's k-means.
- **Fitting stage.** The paper's models were fitted with a 64-kernel dictionary, which was then rebuilt with 500
  kernels and converted to one raw width. Here `fit` builds the final dictionary directly. On fit 0 of the 2D K = 64
  arm, it reproduces the production model file to ~10⁻¹¹ relative. The one difference is the sign of 10 of the 187
  columns of W, which no statistic depends on.
- **Figures.** Every figure in the table above, drawn by `make_figures.sh` from `data/`, is pixel-identical to the
  production figure.
