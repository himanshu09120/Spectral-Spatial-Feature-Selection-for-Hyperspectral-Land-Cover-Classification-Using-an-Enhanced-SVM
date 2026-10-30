# Spectral–Spatial Feature Selection for Hyperspectral Land-Cover Classification Using an Enhanced SVM

FRSS-SVM combines training-only Fisher ranking, correlation pruning, local measurement statistics and an RBF SVM for agricultural land-cover classification. This is a classical, CPU-only Pattern Recognition project; no GPU, PyTorch or TensorFlow is required. The combination is a project proposal, not a claim of new individual algorithms.

**Current status:** official LongKou files downloaded and validated. E0–E6 are implemented. No classifier has been trained, at the owner's request. All scientific results remain pending. Default execution prints a plan, and training is disabled in configuration.

Repository: [himanshu09120 / Spectral-Spatial-Feature-Selection-for-Hyperspectral-Land-Cover-Classification-Using-an-Enhanced-SVM](https://github.com/himanshu09120/Spectral-Spatial-Feature-Selection-for-Hyperspectral-Land-Cover-Classification-Using-an-Enhanced-SVM). This is the configured Git remote; publication/accessibility has not been verified. Branch: `main`.

## Setup

Use Python 3.9–3.12. Python 3.11 or 3.12 is suitable for a new installation. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows activate with `.venv\Scripts\activate`. The project-local `.venv` created during implementation contains compatible packages; no global packages were changed. `requirements-lock.txt` records the exact tested Python 3.9 environment. The broader ranges in `requirements.txt` allow compatible wheels on the documented Python versions.

## Dataset

Source: [official WHU-Hi resource page](https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm). See [data/README.md](data/README.md) for official download provenance and manual alternatives. To obtain only LongKou from the official ZIP:

```sh
python scripts/download_data.py
```

The downloader verifies and skips matching existing files, and refuses to overwrite mismatches. The files already downloaded in this workspace are:

```text
data/raw/WHU_Hi_LongKou.mat       variable WHU_Hi_LongKou
data/raw/WHU_Hi_LongKou_gt.mat    variable WHU_Hi_LongKou_gt
```

The actual cube has shape 550 × 400 × 270, nine classes, 204,542 labelled pixels and 15,458 background pixels. Label 0 is excluded from fitting and scoring. Measured counts and hashes are recorded in `data/DATASET_VALIDATION.json` and `data/DOWNLOAD_PROVENANCE.json`. Raw measurements remain ignored by Git; another machine must download them.

Standard MAT and v7.3/HDF5 are supported. The loader rejects ambiguous variables, spatial misalignment, unexpected labels and nonfinite values. Set `cube_key`, `labels_key` or `cube_axes` explicitly if your official files differ. MATLAB HDF5 dimensions are reversed when the MATLAB metadata indicates that storage convention; generic HDF5 is not silently transposed.

## Safe commands: no classifier training

```sh
.venv/bin/python -m src.main --dry-run
.venv/bin/python -m src.main --validate-data
.venv/bin/python -m src.main --smoke-test
.venv/bin/python -m unittest discover -s tests -v
```

The default `.venv/bin/python -m src.main` also prints a plan. The smoke test checks synthetic array shapes, finite spatial statistics and hand-constructed metric arithmetic. Numerical unit tests use tiny arrays to check preprocessing algebra and do not fit classifiers. Synthetic values are never saved as LongKou findings.

For restricted macOS environments, prefix these commands with `PYTHONPYCACHEPREFIX=/tmp/frss-pycache MPLCONFIGDIR=/tmp/frss-matplotlib` if the normal user cache directories are unwritable.

## Running experiments later (starts CPU training)

Only when you intend to train:

```sh
.venv/bin/python -m src.main --run --allow-training
```

Alternatively set `execution.allow_training` to `true` in `config/config.json` and use `--run`. Neither action was performed during implementation. An optional miniature integration experiment is available as `--smoke-train --allow-training`; it writes only to `outputs/synthetic_smoke_NOT_RESEARCH/` and was not run.

For MATLAB, use `Main('dry-run')`, `Main('validate-data')` or `Main('smoke-test')` without training. Plain `Main` launches the complete Python experiment pipeline and **fails clearly while training remains disabled**. Enable `execution.allow_training` in configuration before a future full evaluation. Interpreter selection is `FRSS_PYTHON`, then local `.venv`, then an available `python3`/`python` command. To configure an interpreter before evaluation:

```matlab
setenv('FRSS_PYTHON', '/absolute/path/to/project/.venv/bin/python');
Main('validate-data');
% After enabling training in config/config.json:
Main;
```

`Main.m` resolves its own directory, quotes paths, restores the prior MATLAB directory with `onCleanup` and raises a MATLAB error on Python failure. `matlab -batch "Main"` propagates that failure as a nonzero process exit. MATLAB was unavailable, so only static launcher checks and direct invocation of its Python entry point have been performed. This is a Python implementation with a MATLAB launcher, not native MATLAB ML code.

## Protocol and experiments

The primary protocol is stratified random pixels with a 70/15/15 train/validation/test split and seeds 7, 19, 42, 73 and 101. A single capped subset of at most 1,000 training pixels per class is shared by E1–E6 and every candidate within a seed. Unused training-pool pixels do not move into validation or test. E0 uses the original training-pool priors so that class balancing does not turn the majority reference into an arbitrary tie.

| ID | Representation | Classifier / candidates |
|---|---|---|
| E0 | Single ignored input; original training majority | Dummy majority |
| E1 | All 270 bands | RBF SVM; optional linear SVM |
| E2 | Training-standardized PCA, 20 or 50 components | RBF SVM |
| E3 | Training-only neighbour noise covariance and MNF, 20 or 50 components | RBF SVM |
| E4 | Fisher ranking and absolute-correlation pruning at 0.95 | RBF SVM; 15/30/45/60 bands |
| E5 | Selected bands + 3×3 local means + local standard deviations | RBF SVM; 3k features |
| E6 | Same family of spatial representations | Random Forest; 100/200 trees |

SVM candidates are `C ∈ {1,10}` and `gamma ∈ {scale,0.01}`. RF depth is capped at 24, minimum leaf size at 2 and workers at 1. PCA/MNF are decomposed once at the largest eligible count per seed; component prefixes are reused. Post-reduction component scaling is fitted only on training rows. Every classifier candidate is fitted once; the winning fitted model is used directly, with no train+validation refit.

Validation macro-F1 selects each method's feature count and hyperparameters. Ties favor the first smaller-count candidate. Infeasible PCA/MNF counts are skipped. Correlation pruning is never relaxed: infeasible band counts are skipped, and if all requested counts exceed the retained nonempty set, that entire retained set is the predeclared fallback. Selected zero-based and MATLAB one-based indices are saved. E4/E5 may choose different counts; their tuned-family ablation is not a perfectly matched fixed-band experiment.

MNF estimates noise with `(x_p-x_q)/sqrt(2)` from adjacent pixels **both in the capped training set**, divides by training standard deviations and regularizes the covariance. It solves `C_x v = λ C_n v`, orders descending total/noise variance and saves diagnostics and eligible pairs. A shortage of pairs raises a clear error. No PCA substitute or external/test pixels are used. Signal edges can inflate estimated noise, a limitation of this estimator.

Random pixel neighbourhoods share scene measurements; this is transductive same-scene evaluation and can be optimistic. To request a separate geographical sensitivity protocol, add `"spatial_block"` to `protocols` before evaluation. Whole 40-pixel blocks are assigned independently, then eroded by a configured 2-pixel buffer. Missing-class partitions fail rather than silently changing seeds. This protocol approximates split fractions, is not strictly stratified, and may be infeasible for this single scene. Its results are kept separate; it has not been executed on LongKou.

## Configuration and resource control

`config/config.json` controls file paths, seeds, protocols, caps, counts, hyperparameters, neighbourhood size, output path, model persistence, threads and map seeds. Relative paths resolve against the repository, not the caller's directory. `--config /path/custom.json` selects another configuration.

The default upper bound is 61 candidate fits per seed, 305 over five seeds, before infeasible-count skips. A CPU SVM search can still take substantial time. The dry run reports the fit budget. No wall-time limit or automatic GPU use is assumed. Reduce the predeclared grid/cap or use one seed for a development check before final evaluation; final course statistics require restoring five seeds. Do not change choices based on test scores.

Only selected bands receive spatial statistics. Statistics use reflected borders and double-precision accumulation before storing float32. Prediction is batched at 4,096 pixels; SVM kernel cache is 256 MB. At most one spatial representation is cached. Classifier parallelism and BLAS threads are bounded. Map prediction runs only for E1/E5 seed 7 by default.

## Outputs and reproducibility

```text
Main.m                      MATLAB launcher
config/config.json          central protocol and resource settings
data/                       download instructions, validation and provenance
scripts/download_data.py    official range-based downloader (standard library)
src/                        loader, preprocessing, methods, runner, metrics, plots, report renderer
tests/                      deterministic unittest suite, no classifier training
report/                     scientific framework, contribution template, presentation outline
outputs/data_validation.json          actual data inspection; generated without training
outputs/manifest.json                 run fingerprint, full config, versions and SHA256 hashes
outputs/logs/run.log                  append-only progress and errors
outputs/metrics/<protocol>/seed_*/    split indices/counts, candidates, bands, MNF diagnostics, test metrics
outputs/metrics/per_seed.csv          raw seed metrics
outputs/metrics/summary.csv           mean ± sample SD and completion flags
outputs/metrics/per_class*.csv        per-seed and aggregate class metrics
outputs/metrics/ablation_deltas.csv    paired E4−E1, E5−E4 and E5−E1 differences
outputs/figures/                      dataset and experiment plots
outputs/predictions/                 test predictions and descriptive full-scene maps
outputs/models/                      joblib classifier + fitted representation
outputs/results.md                   generated table of measured results
```

OA, macro precision/recall/F1, nine-class confusion counts, per-class precision/recall/F1 and runtimes are saved. Undefined per-class ratios use zero. SD is sample SD (`ddof=1`); single-run SD is undefined, never reported as evidence of stability. `summary.csv` marks whether all configured seeds are present. Paired ablations verify common split digests.

Timing separates selected classifier fit, accumulated preprocessing, search wall time and batched test prediction including per-batch transformation. It excludes I/O, plots and optional maps. Shared transformation/cache costs make warm preprocessing times order-dependent; these are not isolated algorithm benchmarks. Reconstructing the winning spatial representation after search is currently outside prediction timing.

Figures cover false colour, ground truth, descriptive spectra, predictions, held-out test error, raw/normalized confusion, Fisher ranking, validation feature curves, accuracy/F1 mean ± SD, and fit-time/dimensionality. Full-scene maps include training pixels and must not be used to estimate test accuracy. Empty background is grey. No performance figure or map has been produced yet.

Resume accepts only matching source/configuration/data/library fingerprints and matching splits. Completed method markers are atomic and written after artifacts. A failed method restarts its search; its candidate audit survives, but individual candidate models are not checkpointed. A root lock prevents duplicate runs. If a process is forcibly killed, verify the PID in `outputs/.run.lock` is inactive before removing that stale lock. Changed code/configuration requires a fresh output directory. Never load untrusted joblib files.

See [report/final_report.md](report/final_report.md) for methodology and explicitly pending results, and [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md) for exact verification status. Fill in the contributions appendix and verify the public repository is accessible before the 5 November 2026 submission.
