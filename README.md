# Spectral–Spatial Feature Selection for Hyperspectral Land-Cover Classification Using an Enhanced SVM

FRSS-SVM combines training-only Fisher ranking, correlation pruning, local measurement statistics and an RBF SVM for agricultural land-cover classification. This is a classical, CPU-only Pattern Recognition project; no GPU, PyTorch or TensorFlow is required. The combination is a project proposal, not a claim of new individual algorithms.

**Current status (5 November 2026):** all five seeds × seven methods (35 evaluations, 155 candidate fits) completed on real WHU-Hi-LongKou data on a separate Windows device. The recorded completion run passed 22 tests and the independent prediction-based audit. The committed [evidence bundle](report/evidence/), [measured report](report/final_report.md), [PDF](output/pdf/final_report.pdf), and [presentation](report/presentation.pptx) preserve that work. This documentation update uses those saved results; it does not represent a new experiment on this Mac.

Repository: [himanshu09120 / Spectral-Spatial-Feature-Selection-for-Hyperspectral-Land-Cover-Classification-Using-an-Enhanced-SVM](https://github.com/himanshu09120/Spectral-Spatial-Feature-Selection-for-Hyperspectral-Land-Cover-Classification-Using-an-Enhanced-SVM). Commit `8e01f2a` (5 November 2026) includes the completed results and reporting tools. Remote publication status is not inferred from the local checkout.

## Recorded results

Values are mean ± sample SD over seeds 7, 19, 42, 73 and 101, from [the saved results](report/evidence/summary.csv).

| Method | Accuracy (%) | Macro-F1 (%) | Mean input features |
|---|---|---|---|
| E0 Majority | 32.785 ± 0.000 | 5.487 ± 0.000 | 1 |
| E1 Full-spectrum SVM | 96.967 ± 0.189 | 93.243 ± 0.362 | 270 |
| E2 PCA-SVM | 97.200 ± 0.263 | 94.026 ± 0.444 | 50 |
| E3 MNF-SVM | 98.384 ± 0.270 | 96.665 ± 0.365 | 20 |
| E4 Selected-band SVM | 89.825 ± 0.290 | 77.947 ± 0.414 | 6 |
| E5 FRSS-SVM | 97.131 ± 0.201 | 92.936 ± 0.476 | 18 |
| E6 Spatial Random Forest | 95.353 ± 0.148 | 89.254 ± 0.188 | 18 |

Strict correlation pruning retained six bands in every seed, triggering the predeclared fallback below the nominal 15/30/45/60-band grid. Local statistics expanded those six bands to 18 E5/E6 features. E5 gained 14.990 macro-F1 percentage points over E4; against E1 it gained 0.164 accuracy points and lost 0.307 macro-F1 points. MNF-SVM had the highest observed mean macro-F1. These are same-scene random-pixel results; no independent geographic validation is claimed.

## Setup

Use Python 3.9–3.12. Python 3.11 or 3.12 is suitable for a new installation. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows activate with `.venv\Scripts\activate`. The completed research run used Python 3.12.5 on Windows, an Intel Core i5-12450H and approximately 16 GB RAM. `requirements-windows-lock.txt` records that environment, including optional PDF dependencies. `requirements-lock.txt` records the earlier Python 3.9 macOS implementation environment. Neither lock file describes every checkout's active interpreter. The broader ranges in `requirements.txt` allow compatible wheels on the documented Python versions.

## Dataset

Source: [official WHU-Hi resource page](https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm). See [data/README.md](data/README.md) for official download provenance and manual alternatives. To obtain only LongKou from the official ZIP:

```sh
python scripts/download_data.py
```

The downloader verifies and skips matching existing files, and refuses to overwrite mismatches. Expected local files are:

```text
data/raw/WHU_Hi_LongKou.mat       variable WHU_Hi_LongKou
data/raw/WHU_Hi_LongKou_gt.mat    variable WHU_Hi_LongKou_gt
```

The actual cube has shape 550 × 400 × 270, nine classes, 204,542 labelled pixels and 15,458 background pixels. Label 0 is excluded from fitting and scoring. Measured counts and hashes are recorded in `data/DATASET_VALIDATION.json` and `data/DOWNLOAD_PROVENANCE.json`. Raw measurements remain ignored by Git; another machine must download them.

Standard MAT and v7.3/HDF5 are supported. The loader rejects ambiguous variables, spatial misalignment, unexpected labels and nonfinite values. Set `cube_key`, `labels_key` or `cube_axes` explicitly if your official files differ. MATLAB HDF5 dimensions are reversed when the MATLAB metadata indicates that storage convention; generic HDF5 is not silently transposed.

## Validation and tests

```sh
.venv/bin/python -m src.main --dry-run
.venv/bin/python -m src.main --validate-data
.venv/bin/python -m src.main --smoke-test
.venv/bin/python -m unittest discover -s tests -v
```

The default `.venv/bin/python -m src.main` also prints a plan. The smoke test checks synthetic array shapes, finite spatial statistics and hand-constructed metric arithmetic. Most tests use tiny arrays or a non-learning stub; the prediction regression fits tiny real classifiers to check exact serial/parallel agreement. Synthetic values are never saved as LongKou findings.

For restricted macOS environments, prefix these commands with `PYTHONPYCACHEPREFIX=/tmp/frss-pycache MPLCONFIGDIR=/tmp/frss-matplotlib` if the normal user cache directories are unwritable.

## Running experiments (starts CPU training)

Only when you intend to train:

```sh
.venv/bin/python -m src.main --run --allow-training
```

Alternatively set `execution.allow_training` to `true` in `config/config.json` and use `--run`. The completed run was explicitly authorized on 2 November 2026. That session also recorded a miniature integration experiment using `--smoke-train --allow-training`; its isolated `outputs/synthetic_smoke_NOT_RESEARCH/` files are software checks only. Viewing the committed report and evidence requires no training. The commands here are reproduction instructions, not steps performed for the 5 November documentation update.

For MATLAB, use `Main('dry-run')`, `Main('validate-data')` or `Main('smoke-test')` without training. Plain `Main` launches the complete experiment with explicit training enabled, then displays aggregate metrics, per-class metrics, comparison charts, confusion matrices and the seed-7 map. Matching completed results resume without fitting again. Interpreter selection is `FRSS_PYTHON`, then local `.venv`, then `python3`/`python`.

`Main.m` resolves its own directory, quotes paths, restores the caller's directory and propagates Python failures. MATLAB R2024b successfully executed the dry-run launcher on the Windows device during the recorded completion work; a full MATLAB-launched training run is not documented. This is a Python implementation with a MATLAB launcher. Default Python execution prints a plan, whereas plain MATLAB `Main` starts training when compatible completed outputs are absent.

On Windows, the equivalent commands are:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/download_data.py
.venv/Scripts/python.exe -m src.main --run --allow-training
.venv/Scripts/python.exe scripts/audit_results.py
```

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

Random pixel neighbourhoods share scene measurements; this is transductive same-scene evaluation and can be optimistic. To request a separate geographical sensitivity protocol, add `"spatial_block"` to `protocols` before evaluation. Whole 40-pixel blocks are assigned independently, then eroded by a configured 2-pixel buffer. Missing-class partitions fail rather than silently changing seeds. This protocol approximates split fractions, is not strictly stratified, and may be infeasible for this single scene. The recorded real-data preflight found missing validation classes for seeds 7, 19 and 73; seeds 42 and 101 were feasible. No block models were trained and no replacement seeds were chosen. See the committed [protocol feasibility record](report/evidence/protocol_feasibility.json). `scripts/check_protocols.py` can reproduce the preflight without fitting classifiers.

## Configuration and resource control

`config/config.json` controls file paths, seeds, protocols, caps, counts, hyperparameters, neighbourhood size, output path, model persistence, threads and map seeds. Relative paths resolve against the repository, not the caller's directory. `--config /path/custom.json` selects another configuration.

The default upper bound is 61 candidate fits per seed, 305 over five seeds, before infeasible-count skips. The completed run retained 31 candidates per seed, 155 in total, after strict band pruning. A CPU SVM search can still take substantial time. The dry run reports the upper fit budget. No wall-time limit or automatic GPU use is assumed. Reduce the predeclared grid/cap or use one seed for a development check before final evaluation; final course statistics require restoring five seeds. Do not change choices based on test scores.

Only selected bands receive spatial statistics. Statistics use reflected borders and double-precision accumulation before storing float32. Prediction is batched at 4,096 pixels; SVM kernel cache is 256 MB. At most one spatial representation is cached. Classifier fitting and BLAS use one thread. Independent prediction batches use four bounded workers; `execution.prediction_workers` can be set to 1 on smaller machines. Serial/parallel prediction equality is tested. Map prediction runs only for E1/E5 seed 7 by default.

## Outputs and reproducibility

```text
Main.m                      MATLAB launcher
config/config.json          central protocol and resource settings
data/                       download instructions, validation and provenance
scripts/download_data.py    official range-based downloader (standard library)
scripts/check_protocols.py  class-coverage and MNF-pair preflight; no classifiers fitted
scripts/audit_results.py    independent audit of original outputs and test predictions
scripts/finalize_report.py  audit, measured report, evidence and outline generation
scripts/render_report.py    PDF export from the checked-in Markdown report
src/                        loader, preprocessing, methods, runner, metrics, plots, report renderer
tests/                      numerical, orchestration and prediction regressions
report/final_report.md       measured scientific report and reproducibility appendix
report/individual_contributions.md    named team members and 50/50 responsibility allocation
report/presentation_outline.md        eight-slide outline with measured talking points
report/presentation.pptx               saved presentation export
report/evidence/                      committed metrics, selections, manifest and audit
report/figures/                       committed report figures
output/pdf/final_report.pdf           saved report PDF
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

Figures cover false colour, ground truth, descriptive spectra, predictions, held-out test error, raw/normalized confusion, Fisher ranking, validation feature curves, accuracy/F1 mean ± SD, and fit-time/dimensionality. Full-scene maps include training pixels and must not be used to estimate test accuracy. Empty background is grey. Research figures and maps are written as each method completes.

Resume accepts only matching source/configuration/data/library fingerprints and matching splits. Completed method markers are atomic and written after artifacts. A failed method restarts its search; its candidate audit survives, but individual candidate models are not checkpointed. A root lock prevents duplicate runs. If a process is forcibly killed, verify the PID in `outputs/.run.lock` is inactive before removing that stale lock. Changed code/configuration requires a fresh output directory. Never load untrusted joblib files.

See [report/final_report.md](report/final_report.md) for the scientific report and [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md) for the current status and dated development history. The [contributions appendix](report/individual_contributions.md) records Himanshu Sinha (S20240020290) and Ghule Yash Mahesh (S20240020288), their requested 50/50 allocation, and AI assistance. The remaining human step is review of those statements and the submission materials.

## Final report and evidence

The completed run's evidence is committed and can be read without running code. `report/evidence/audit.json` records PASS for all 35 method/seed results and 155 candidate fits, with fingerprint `58f8e5ab00fe15cd0179328ae11b7572fce15e3f9eee8fd23e1ddad0142f824b`. It is the audit saved on the experiment device, not a newly executed audit of this checkout.

After a complete run, `python scripts/finalize_report.py` independently recomputes every test metric, validates selection and split provenance, and writes the measured report plus `report/evidence/`, selected `report/figures/`, and the presentation outline. It refuses incomplete or incompatible results. This requires the full original `outputs/` tree, including split indices and test prediction arrays, plus a matching source/configuration/data/library fingerprint. Those ignored files are not included in the committed evidence bundle and are absent from this Mac checkout. The bundle alone cannot resume training or repeat the prediction-based audit. The finalizer targets the default `outputs/` directory and the completed five-seed pixel experiment.

`report/evidence/SHA256SUMS.json` records the original exported bytes. Its 99 text-file digests reflect Windows CRLF line endings; Git's LF checkout conversion changes those byte hashes. The eight figure digests are unaffected. This line-ending difference does not change the saved metric values. Preserve the experiment records when updating documentation.

For a future PDF export, install `requirements-report.txt`, then run `python scripts/render_report.py`; this needs the Markdown report and committed figures, not the original training outputs. The target is `output/pdf/final_report.pdf`. The checked-in PDF and PowerPoint are saved exports; the 5 November documentation-only update does not regenerate them. The finalizer updates the slide outline, not the PowerPoint. All reported scientific values remain the saved five-seed measurements.
