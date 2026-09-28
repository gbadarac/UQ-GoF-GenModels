# HANDOFF — 2D + 4D NF LRT campaign (10-fit fit-to-fit bands)

**Author:** Giada Badaracco (gbadaracco@phys.ethz.ch) · ETH Zürich
**Date:** 2026-09-28
**For:** whoever runs the NF LRT campaign (BOTH 2D and 4D) on a larger cluster. t3 can't: only ~16 GPUs (weeks), and its shared CPU queue is fairshare-throttled to ~30 concurrent jobs (also weeks) — so both legs are handed off.

---

## 0. TL;DR

You are running the **Normalizing-Flow LRT campaign (2D and 4D)** for the WiFi-NPLM paper (NF appendix).
Everything is a thin SLURM fan-out over `LRT/submit_LRT_toys.sh` → `LRT/LRT.py`; the physics is
already validated. Your job is to **run it at scale** on a cluster with many GPUs.

Steps, in order:
1. Upload this folder to your cluster and `tar -xf` it — it is a COMPLETE copy of the repo: all
   the code PLUS the 4D models/data already in place (no git clone, no separate download).
2. Build the `nplm_env` conda env from `envs/nplm_env.yml`.
3. Edit ~8 hardcoded t3 paths + the SLURM directives (list in §4).
4. Dry-run, sanity-check ONE cell, then launch (§5).

Total workload ≈ **~170k toys** — 4D ≈ 80k (10 fits × 8 N_test × 5 configs × 200) + 2D ≈ 90k (2D
keeps the 200k point → 9 N_test). Both dims run **M = single / 32 / 128** (decision 2026-09-28).
Weeks on 16 GPUs; fast on a big cluster.

---

## 1. What this campaign is

The paper studies the NPLM likelihood-ratio-test **p-value vs. N_test** for a WiFi ensemble
density estimate. The **methodological deliverable** is the **10-fit fit-to-fit band**: instead of
ONE ensemble draw per cell, we draw **10 random M-member subsamples** ("ensemble choices") from a
pool of **512 trained flows**, run the full LRT for each, and **pool the per-toy p-values across the
10 fits** for an honest 25/75 band. This captures the flow-to-flow (model-choice) spread that the
WiFi analytic covariance under-estimates when the flow is misspecified — which in 4D it is
(empirically the fit-to-fit spread is ~4–5× the WiFi `f_err` in 4D vs ~1× in 2D).

- **NF pipeline = appendix-only** (the tangent method is the main body). The appendix need not
  match the main body point-for-point.
- **BOTH the 2D and 4D legs are handed off.** t3's 16 GPUs make it weeks; its shared CPU nodes are
  fairshare-throttled to ~30 concurrent jobs (also weeks). Profiling showed the per-toy cost is
  **~93% the composite numerator fit** (a `scipy.optimize.minimize` loop — CPU-bound; the flows are
  evaluated once and **cached**, NOT re-run per fit-iteration; flow eval is only ~7%). So the
  campaign just needs **many parallel jobs**, which a big cluster provides — GPU or CPU (see §4).
- This is the **"updated runs"**: the 10-fit rework, committed to `main` 2026-09-27. It supersedes
  the earlier **single-draw** campaign (the preliminary, already plotted).

## 2. The exact run — grid, sizes, budget

Driver: `LRT/run_nf_scan.sh` (mode `DIM=4d SCAN=campaign`). It resolves seeds and fans out
`sbatch` array jobs over `LRT/submit_LRT_toys.sh` (scenario `nf:ens_4d_uniform`).

| Knob | Value (4D) | Where |
|---|---|---|
| Ensemble choices (fit-to-fit axis) | `0 1 2 3 4 5 6 7 8 9` (10) | `CHOICES` |
| N_test grid | `100 200 1000 2000 10000 20000 50000 100000` (8) | `NTEST_DEF` (4D **drops 200k**: 200k/1M test-pool = 20% > the ≤10% rule) |
| Ensemble sizes M | **`32 128`** + single (M=1) — decision 2026-09-28 for BOTH dims (was `2 4 8 16 32 64 128`) | `ENS_DEF` (the default for both 2D & 4D) |
| Null model | `point` (frozen w) **and** `composite` (constrained w~N(ŵ,Σ)) | `MODES` |
| Toys **per choice** | 100 Gaussian null + 100 obs (self-calibrated) | `NULL_TOYS` / `OBS_TOYS` |
| σ (kernel bandwidth) | **0.53** (preliminary min-p σ-scan) | `SIGMA` / `SIGMA_DEF` |
| M numerator kernels | 500 | `M` (`--n_kernels`) |
| clip on tilt coeffs | 3 (L-BFGS-B; crash-safe) | `CLIP` (`--clip_b`) |
| N_ref | = N_test (1:1) | submit default |
| Pool size P | 512 flows (arch L6/blk8/h128/bins10) | on disk |

Single model (M=1) is auto-forced to **frozen / point-null** and draws one flow from the 512 pool
per choice (reproducible: `select_member_seeds --rng_seed <choice>`). Ensembles (M≥2) read that
choice's OWN `member_seeds.txt` so LRT loads exactly the flows the wifi fit paired its weights with.

> **The default is already `32 128`** (plus single) for both dims: a bare `DIM=4d` (or `DIM=2d`) run
> does the right thing — no override needed. Pass `ENS_LIST=...` only for a different set.
>
> **To run 2D:** just `DIM=2d`. σ=0.4 and the 200k-inclusive grid (`100…200000`, 9 points) are its
> built-in defaults, and obs draws fresh from the analytic truth (`--target_truth 2d_gmm_skew`), so
> **2D needs no obs data file** (unlike 4D's held-out pool). Everything else is identical to 4D.
>
> **This 4D run mirrors the 2D run exactly, except two intended dimension-specific differences:**
> (1) σ = 0.53 (4D) vs 0.4 (2D) — data-space length scale, re-derived per embedding; and (2) the
> N_test grid **drops 200k** in 4D (4D obs data is a finite held-out pool → the ≤10% rule; 2D draws
> fresh from analytic truth so 200k is fine). Choices (10), ensemble sizes (single/32/128), null
> modes (point+composite), toys (100 null + 100 obs/choice), M=500, clip=3, N_ref=1:1 are identical.

**Observed data (calib=0):** 4D has no analytic truth, so obs toys bootstrap the **held-out** file
`data_heldout_qcd.npy` (1.9M QCD jets, disjoint from the 100k training set — the "N_test must not
come from training" rule). Null toys (calib=1) are self-contained (SIR-from-q, `flow.sample`).

## 3. What's in this bundle (the gitignored artifacts)

This folder is a COMPLETE, ready-to-run copy of the repo: all the code PLUS the three big artifacts
git normally excludes (`.gitignore` drops `*.pth`, `*.npy/.npz`, and the whole
`EstimationNFnflows_outputs/` tree). They are already at the paths below inside this folder —
nothing to download or rsync separately.

| Artifact | Repo-relative path | Size | What it is |
|---|---|---|---|
| 512-flow pool (4D) | `Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/4_dim/4d_embedding_qcd/N_100000_dim_4_seeds_512_6_8_128_10/` | 4.1 G | 512× `model_NNN/model.pth` + `architecture_config.json` (arch L6/blk8/h128/bins10). LRT loads the actual flow densities from `model_%03d/model.pth` — the wifi folder only stores seed *indices*, not models, so the pool IS required. Full pool shipped: the {single,32,128}×10-choice union already references **490/512** flows (10× M=128 alone hits 475), so a subset saves only ~4%. |
| 10-fit wifi choices (4D) | `Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF_uniform/4d_embedding/` | ~2 M | 7 `…_ensemblecomponents{2,4,8,16,32,64,128}` × 10 `choice{0..9}`, each with `member_seeds.txt`, `w_i_fitted.npy`, `cov_w.npy` (png plots stripped). With the {32,128} decision only those two configs are used, but all are shipped (negligible size). |
| Held-out obs data (4D) | `data/4d_embeddings/4d_embedding_qcd_Ntrain100000_Ntest100000_seed42/data_heldout_qcd.npy` | 44 M | bootstrap source for 4D calib=0 |
| 512-flow pool (2D) | `Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_seeds_512_4_4_64_8/` | 303 M | 512× `model_NNN/model.pth` + `architecture_config.json` (arch L4/blk4/h64/bins8). Same role as the 4D pool; per-model pngs stripped. |
| 10-fit wifi choices (2D) | `Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF_uniform/2d_toymodel/` | 2.1 M | 2D analog: 7 `…_ensemblecomponents{2..128}` × 10 `choice{0..9}`, each with `member_seeds.txt`, `w_i_fitted.npy`, `cov_w.npy` (pngs stripped). |
| Obs data (2D) | — (none) | — | **No file needed** — 2D obs draws fresh from analytic truth (`--target_truth 2d_gmm_skew`). |

**The code is also included** (it's tracked in git, exported into this folder): `LRT/LRT.py`,
`LRT/submit_LRT_toys.sh`, `LRT/run_nf_scan.sh`, the analysis/plot scripts, the shared utils
`shared/Sparker_utils/{LRTGOFutils_v2,GENutils}.py` (LRT.py adds them to `sys.path` relative to its
own location), `Uncertainty_Modeling/wifi/Fit_Weights/select_member_seeds.py`, and the env spec
`envs/nplm_env.yml`. So this one folder is everything.

## 4. Setup on the new cluster

**Step 1 — upload & unpack.** Copy the tarball to your cluster and extract:
```
tar -xf MonoJet_NPLM_analysis.tar
cd MonoJet_NPLM_analysis
```
This is the whole repo with the 512-flow pool, wifi choices, and held-out data already in place —
no git clone, no artifact rsync.

**Step 2 — build the env.**
```
conda env create -f envs/nplm_env.yml       # creates env "nplm_env"
```
Note the absolute python: `conda run -n nplm_env which python` → you'll paste this into the scripts.

**Step 3 — edit the hardcoded t3 paths.** Replace `/work/gbadarac/...` with YOUR paths:

- `LRT/run_nf_scan.sh`
  - L47 `REPO_ROOT="..."` → your checkout
  - L49 `SELPY="/work/gbadarac/miniforge3/envs/nplm_env/bin/python"` → your nplm_env python
  - L117 / L122 `sbatch --partition=gpu ...` → your GPU partition
- `LRT/submit_LRT_toys.sh`
  - L96 `REPO_ROOT="..."` → your checkout
  - L114 `PYBIN="/work/gbadarac/miniforge3/envs/nplm_env/bin/python"` → your nplm_env python
  - L224 `. /work/gbadarac/miniforge3/etc/profile.d/conda.sh` → your conda profile
  - L7  `#SBATCH --account=gpu_gres` → your account
  - L8  `#SBATCH --partition=qgpu,gpu` → your GPU partition(s)
  - L9  `#SBATCH --gres=gpu:1` → your GRES string
  - L11–12 `#SBATCH -o/-e /work/gbadarac/.../LRT/results/logs/...` → your log dir (or leave as `$REPO_ROOT/...` — the run creates `LRT/results/logs`)

> The kernels-only `LD_LIBRARY_PATH` block (submit_LRT_toys.sh ~L240) references an nplm_env `lib`
> path but is **not** used for `MODEL=nf` (the `if MODEL_TYPE==nf` branch skips it). Ignore for 4D.

**Step 4 — GPU vs CPU (this decides your resource).** The per-toy cost is **~93% the composite fit**,
which is CPU-bound (`scipy.optimize.minimize`); the GPU only accelerates the ~7% flow evaluation. So
**either works — pick whichever gives you the most PARALLEL jobs**:
- **GPU:** the SBATCH defaults are already GPU (`--gres=gpu:1`); just point them at your
  partition/account/GRES (§3 above).
- **CPU (usually the better choice if you have hundreds of cores):** `LRT.py` auto-runs on CPU when
  no GPU is visible — no code change, just submit to a CPU partition. `run_nf_scan.sh` exposes
  `SB_PARTITION` / `SB_ACCOUNT` / `SB_GRES` for exactly this (set `SB_GRES=none`), e.g.
  `SB_PARTITION=<cpu_part> SB_ACCOUNT=<acct> SB_GRES=none DIM=4d SCAN=campaign ./run_nf_scan.sh`.
  Flow eval is a little slower on CPU, but with 100s of concurrent CPU jobs the wall-clock wins
  easily. (This is exactly why t3 failed: not CPU itself, but its ~30-concurrent fairshare throttle.)

## 5. How to run

**Always dry-run first** (prints every `sbatch` line, submits nothing):
```
cd LRT
DRYRUN=1 DIM=4d SCAN=campaign ./run_nf_scan.sh | less
```

**Sanity-check ONE cheap cell end-to-end before the full fan-out** (one choice, small N_test, few
toys — should finish in minutes and write results under `results/nf/4d_uniform_ensemble/choice0/`):
```
DRYRUN=0 DIM=4d SCAN=campaign \
  CHOICES=0 NTEST_LIST=100 ENS_LIST=32 MODES=point \
  NULL_TOYS=4 OBS_TOYS=4 ./run_nf_scan.sh
```
Confirm it exits 0, the `.out` logs show `toy N exit 0`, and a run-tag dir with `null/`+ obs seeds
appears. Then scale up.

**Full campaign** (all 10 choices, M=single/32/128, both modes, full N_test grid):
```
DRYRUN=0 DIM=4d SCAN=campaign ./run_nf_scan.sh      # 4D
DRYRUN=0 DIM=2d SCAN=campaign ./run_nf_scan.sh      # 2D — same knobs; σ=0.4, 200k grid, analytic obs are its defaults
```
Subset knobs (space-separated, override at call time): `CHOICES NTEST_LIST ENS_LIST MODES CALIBS
INCLUDE_SINGLE NULL_TOYS OBS_TOYS STD_TIME LONG_TIME BATCH`. Recommended: **submit a few choices
at a time** (e.g. `CHOICES="0 1 2"`) so you can watch throughput and the pooled band becomes
plottable as soon as ≥2 choices finish.

## 6. Output layout & plotting

Results land at `LRT/results/nf/4d_uniform_ensemble/choice<k>/<run_tag>/{null,<obs seeds>}`
(2D → `LRT/results/nf/2d_uniform_ensemble/…`, same layout).
The `choice<k>/` level is what lets the plot POOL the 10 fits (the run_tag is identical across
choices by construction). Plot (needs `nplm_env` python — scipy is broken in `nf_env`; pass the 2D
base to `--base` for the 2D figure):
```
python LRT/plot_pvalue_vs_ntest.py \
  --base LRT/results/nf/4d_uniform_ensemble \
  --out  LRT/results/nf/4d_uniform_ensemble/figs/pvalue_vs_ntest.pdf
```
It self-calibrates each choice against its own null, pools the per-toy p-values across choices, and
draws the pooled median + 25/75 band. `--mode point|composite`, `--nens`, `--min_ntest` subset.

## 7. Ops gotchas (learned the hard way on t3)

- **Toy BATCHING is mandatory.** One SLURM job per toy = ~510k jobs = ~5× a typical `MaxJobCount`,
  which triggers `sbatch: Resource temporarily unavailable` (EAGAIN) and hammers the shared job
  table / your fairshare. The driver already batches (`batch_for`: 50/20/10/5 toys/task for
  ≤2k/≤20k/≤100k/≥200k N_test) → tens of thousands of tasks, not hundreds of thousands. Override
  uniformly with `BATCH=<n>` if your cluster's limits differ. **toy_id→seed is unchanged by
  batching, so results are identical to unbatched.**
- **Submit throttle/retry** is built in (`SUBMIT_SLEEP=0.5`s between submits, `SUBMIT_RETRIES=10`
  with backoff). Bump `SUBMIT_SLEEP` if your controller still throws EAGAIN.
- **Walls:** ENS=128 gets `LONG_TIME=1-00:00:00`, everything else `STD_TIME=11:59:00`. A
  wall-timeout loses only that task's ≤`BATCH` toys, not the cell. Adjust to your partition limits.
- **Memory:** `mem_for` = 14G (≥100k) / 8G (≥10k) / 6G (else), sized to measured RSS. The
  128-member 100k cell is heaviest (~2 GB state_dicts + data ∝ N).
- **Resumability:** re-running the driver re-submits everything; there's no built-in "skip
  completed." Easiest is to drive with `CHOICES=` (and/or `NTEST_LIST=`/`ENS_LIST=`) to submit only
  the missing slices. Same `toy_id` → same seed → identical results, so a partial rerun is safe.
- **Threads pinned to 1** (`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`) to stop numpy OpenBLAS from
  oversubscribing shared nodes. Keep it.

## 8. Provenance / validation notes

- σ=0.53 came from a min-p σ-scan on the single model (`DIM=4d SCAN=sigma ./run_nf_scan.sh`); it is
  a **power** knob, not a validity knob (every cell is toy-calibrated), so it's safe but possibly
  sub-optimal (the null was mildly over-dispersed vs χ², DOF_eff≈100). Re-scan with `SCAN=sigma` +
  `analyse_LRT_output.py` if you want to optimize power; otherwise 0.53 is fine.
- The earlier **single-draw** 4D campaign already showed a clean monotonic "ensemble helps closure"
  ordering and **point ≈ composite**. The 10-fit rework adds the honest fit-to-fit band on top.
- **No bootstrap null** for the flows (Gaussian toys only) — justified by #params (≤128 ensemble
  weights) ≪ N_train=100k (far from the overfitting limit ⇒ Gaussian ≈ bootstrap) + the empirical
  point≈composite. This is a settled decision, not a TODO.

## 9. Questions

Ping Giada (gbadaracco@phys.ethz.ch). The full paper-campaign rationale lives in the project notes;
the driver/submit scripts are heavily commented at the exact knobs above.
