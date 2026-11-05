# Completion work - 3 November 2026

Continued the 2 November completion work in a new session (Claude Code, Sonnet 5) to finish the remaining real-data training and finalize the submission.

- Found the run interrupted after only seeds 7 (complete) and 19 (partial: E0-E1) had finished; resumed `python -m src.main --run --allow-training` to complete the remaining seeds.
- A background-process mistake caused two instances of the runner to execute concurrently for a short window (an orphaned process from a mis-backgrounded first attempt survived after its wrapper script exited, and a stale-looking `.run.lock` was removed while that process was still live, letting a second instance start). Confirmed no live process held the lock before each restart. The race corrupted exactly one artifact, `pixel/seed_42/E1` (identical winning hyperparameters and validation macro-F1 in both the result and selection-audit files, but a mismatched `fit_seconds` field from the two interleaved writers). Deleted that method's result, selection audit, model, prediction and figure files and recomputed it cleanly; `scripts/audit_results.py` now independently re-verifies all 35 seed/method results against saved held-out predictions with status PASS (155 total candidate fits).
- All 22 tests pass after completion. `config/config.json`'s `execution.allow_training` remains `false` by default; training only ran because `--allow-training` was passed explicitly, consistent with the project's safety default.
- Bootstrapped `pip` into the project `.venv` (missing from the prior environment) and installed `requirements-report.txt` to render the PDF.
- Ran `scripts/finalize_report.py` (re-audits, rewrites `report/final_report.md` with measured five-seed statistics, regenerates `report/evidence/` and `report/figures/`) and `scripts/render_report.py` (produces `output/pdf/final_report.pdf`, 10 pages).
- Measured headline result: FRSS-SVM (E5) achieves 97.131% ± 0.201% accuracy and 92.936% ± 0.476% macro-F1 across five seeds, using 18 features on average versus 270 for the full-spectrum baseline. MNF-SVM (E3) has the highest observed mean macro-F1 (96.665%). Full tables are in `report/final_report.md` Sections 13-16.
- Updated `report/individual_contributions.md`'s AI-assistance disclosure to credit this session's work (resuming training, diagnosing/repairing the concurrency artifact, auditing, and report/PDF generation) alongside the prior Codex-assisted work.
- No git commit or push was performed in this session; working tree changes remain local pending the owner's review before submission.

---

# Completion work - 2 November 2026

The current owner explicitly authorized full CPU experiments in this session, superseding the historical no-training restriction documented below. The original implementation log is retained as history, not the current execution status.

- Read all source modules, tests, configuration, documentation, and both pages of `PR Project Instructions M2026.pdf`; no applicable AGENTS.md was found.
- Created a project-local Python 3.12.5 environment on Windows and downloaded official LongKou files; both file hashes match the earlier provenance.
- Passed the original 21 tests and an actual E0-E6 synthetic training run. Synthetic outputs remain isolated and are never research results.
- Updated MATLAB `Main` to explicitly enable training for its run mode and display summary/per-class tables, metric charts, confusion matrices and the map. MATLAB R2024b executed `Main('dry-run')` successfully.
- Serial prediction proved expensive. Preserved the interrupted initial research attempt under `outputs/serial_preflight_20261102/`, verified its PID had stopped, and restarted in a fresh result directory after adding four bounded prediction workers. All statistical choices remain unchanged. Fitting and BLAS remain single-threaded. A new real-classifier regression verifies exact serial/parallel prediction equality; all 22 tests pass.
- Checked real-data block feasibility without training: seeds 7, 19 and 73 fail validation class coverage; seeds 42 and 101 pass. No seeds or block settings were changed. All five pixel splits contain 143,179 pool / 30,681 validation / 30,682 test pixels; the shared fit subset has 9,000 pixels and 2,163-2,276 eligible MNF noise pairs.
- Added independent result auditing, reproducible report generation, PDF export, and a portable report evidence bundle. Completion measurements will be recorded after the full run finishes.
- Public repository accessibility verified. No commit or push has been performed. The owner supplied Himanshu Sinha (S20240020290) and Ghule Yash Mahesh (S20240020288), requesting equal contributions. The appendix records a 50/50 responsibility allocation and explicitly distinguishes allocation from commit-level authorship evidence.

---

# Implementation log and handoff

## Scope and authorization

The supplied master prompt requested a complete Pattern Recognition project. The owner's direct instruction overrides its experiment-execution phases: **do not train anything; write the code for a MacBook without a GPU**. A subsequent instruction explicitly requested downloading the linked dataset. Both were followed. No actual DummyClassifier, SVC or RandomForestClassifier has been fitted, on either real or synthetic data. Numerical preprocessing algebra was checked on tiny arrays. The full runner was exercised using a fixed-output, non-learning test stub, with real classifier fitting patched to raise an error.

Git identity requested for any future commit/push: username `himanshu09120`, email `hsinha0912@gmail.com`. No commit, push, branch change or Git identity configuration was performed.

## Initial inspection, 29 October 2026

- Workspace: the repository containing this log, under the owner's Desktop.
- Directory contained only `.git`; no source, configuration, dataset, course PDF, experiments or local instruction files were present.
- Searched ancestor locations for `AGENTS.md`; none found.
- Git branch `main`, clean initial working tree, no commits. Existing remote preserved: `https://github.com/himanshu09120/Spectral-Spatial-Feature-Selection-for-Hyperspectral-Land-Cover-Classification-Using-an-Enhanced-SVM.git`.
- Read the owner's full attached `Pasted text.txt` project specification before creating files.
- OS Darwin/macOS ARM64; `/usr/bin/python3` is Python 3.9.6. No MATLAB or Octave executable found. System Python lacked NumPy, SciPy, scikit-learn, Matplotlib, pandas, h5py and pytest.
- No useful existing files were overwritten or removed. No sub-agents were used.

## Implementation plan and completed code

1. Inspect repository/environment and official dataset access.
2. Implement robust scene loading, class validation and shared leakage-aware splitting.
3. Implement feature algorithms and bounded classical classifier construction.
4. Implement validation selection, final testing, artifact generation, aggregation and resume rules.
5. Add safe CLI and MATLAB launcher, tests, report framework and reproduction instructions.
6. Download and inspect official data; run only non-training verification.

All six implementation steps are complete within the no-training restriction. Scientific experimentation and MATLAB runtime validation remain unperformed, as detailed below.

## Files created and responsibilities

| File | Implemented functionality |
|---|---|
| `Main.m` | Location-relative MATLAB launcher, interpreter discovery, quoted paths, caller-directory restoration and error propagation; selectable non-training modes |
| `config/config.json` | Central dataset paths, split/cap/seeds, algorithms, CPU limits and explicit training-off default |
| `src/main.py` | Safe default plan, config validation, no-training scene validation and smoke mode, explicit training opt-in, isolated optional synthetic training mode, source/data/library fingerprint and exclusive run lock |
| `src/data_loader.py` | Standard MAT and v7.3/HDF5 loading, explicit/unique variable selection, orientation checks, finite numeric checks, fixed class IDs/names and actual counts |
| `src/preprocessing.py` | Shared stratified pixel split, common capped training subset, optional eroded block split, overlap assertions, train-only scaling and eligible noise-pair indices |
| `src/feature_selection.py` | Multiclass Fisher scores, stable ranking and strict absolute-correlation pruning; constant-band exclusion |
| `src/dimensionality_reduction.py` | Genuine regularized-noise MNF generalized eigenproblem with descending total/noise variance ordering |
| `src/spatial_features.py` | Selected-band raw/mean/std concatenation, reflected borders and float64 variance accumulation |
| `src/classifiers.py` | Construction of majority dummy, RBF/optional linear SVM and bounded Random Forest; no fitting at import or construction |
| `src/experiment.py` | E0–E6 orchestration, train-only transformations, cached PCA/MNF decompositions and one spatial matrix, validation tuning, test-once scoring, model/prediction persistence and per-method resume |
| `src/evaluation.py` | Fixed-order nine-class metrics, atomic JSON, per-seed/class CSV, sample SD, completion flags and paired ablations |
| `src/visualization.py` | Dataset figures, spectra, Fisher scores, maps, held-out errors, raw/normalized confusion, feature curves and accuracy/F1/runtime comparisons |
| `src/reporting.py` | Markdown table renderer derived only from saved actual results; synthetic warning banner |
| `src/__init__.py` | Package declaration |
| `scripts/download_data.py` | Standard-library official ZIP range downloader, streamed extraction, CRC/size/SHA256 checks, verified existing-file skipping and provenance |
| `tests/test_core.py` | Small deterministic numerical, loader, partition, aggregation, guard and static MATLAB tests |
| `tests/test_orchestration.py` | All E0–E6 branches with a non-learning stub, temporary plots/artifacts, resume/mismatch checks and held-out perturbation invariance |
| `requirements.txt` | Compatible scientific dependencies, CPU-only and Python 3.9–3.12 compatible |
| `requirements-lock.txt` | Exact installed package versions after compatibility verification |
| `.gitignore` | Excludes raw data, models/results, local virtualenv, caches, archives and partial downloads |
| `data/raw/.gitkeep` | Retains the empty destination in a source checkout |
| `data/README.md` | Exact publisher download and extraction instructions and provenance explanation |
| `data/DATASET_VALIDATION.json` | Actual observed dimensions, variables, class counts and missing-class checks |
| `data/DOWNLOAD_PROVENANCE.json` | Official URLs, observed archive metadata, extracted file sizes, CRC32 and SHA256 hashes |
| `README.md` | Setup, safe/future-training commands, launcher behavior, protocol, safeguards, artifact layout and limitations |
| `report/final_report.md` | Complete methodology and scientific report framework; every unmeasured result explicitly pending; verified references |
| `report/individual_contributions.md` | Unfilled team contribution/evidence template |
| `report/presentation_outline.md` | Eight-slide outline with pending-results instructions |
| `IMPLEMENTATION_LOG.md` | This handoff |

The `.venv` directory and two local MAT files were also created but are ignored by Git. There are no prior tracked source files to list as modified; all source additions are new.

## Official data acquisition and observations

The supplied [official resource page](https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm) links its MATLAB download to `https://rsidea-whu-hi.mikecrm.com/MqhjRPN`. The form HTML explicitly publishes the redirect to `http://rsidea.whu.edu.cn/code_data/WHU-Hi/Matlab_data_format.zip`. Its HTTPS equivalent returned HTTP 200, size 900,809,518 bytes and range support. No guessed URL or third-party mirror was used, and no personal information/registration was submitted.

Only the two LongKou MAT entries were downloaded (approximately 186 MB transferred, rather than the complete three-scene archive), extracted and checked against ZIP CRC32 and uncompressed entry sizes. Stored MAT size is not the same as the decoded cube's memory footprint. SHA256 hashes are local integrity records, not signed publisher digests.

| Local file | Bytes | SHA256 |
|---|---:|---|
| `data/raw/WHU_Hi_LongKou.mat` | 185,938,925 | `4ef90f2222128245b8fc5c9c6003dfc37f43569991190eb32fcc8a3753430634` |
| `data/raw/WHU_Hi_LongKou_gt.mat` | 5,305 | `b5a3634dc31a907ea6e60192ad7cd2b7946453ec57526ea7724941d8c1078178` |

The actual variables are `WHU_Hi_LongKou` and `WHU_Hi_LongKou_gt`. Observed cube shape is 550×400×270 and label shape 550×400. Label counts 1–9 are 34,511; 8,374; 3,031; 63,212; 4,151; 11,854; 67,056; 7,124; 5,229. There are 204,542 labelled and 15,458 background pixels. Counts match the official table, no classes are missing and validation found no NaN or infinite inputs. In-memory return types are float32 measurements and int16 labels.

## Algorithms and decisions

- Pixel splits: stratified 70/15/15, seeds 7,19,42,73,101. A single deterministic capped training subset (≤1,000/class) is reused by all E1–E6 candidates. Unused training-pool pixels are never reassigned to evaluation.
- E0 majority uses the original training-pool priors, explicitly exempted from balanced subsampling to preserve its meaning. It is computationally trivial but was still not trained here.
- Train-only standardization, Fisher means/variances, correlation and PCA/MNF. PCA uses full SVD at the largest eligible requested component count and reuses prefixes. Reduced components are training-standardized before classification.
- Fisher numerator is between-class sum of squares; denominator is within-class population-variance sum plus 1e-12. Greedy correlation pruning uses absolute Pearson correlation ≤0.95. Requested counts 15/30/45/60 are skipped if unavailable; if all are infeasible, use the entire nonempty pruned set without relaxing the threshold.
- Spatial features are selected raw bands + local means + population standard deviations, with an odd window (3 by default) and reflected borders. No labels enter neighbourhood statistics.
- MNF noise is `(x_p-x_q)/sqrt(2)` from horizontal/vertical neighbours both in the capped training set, scaled by training standard deviations. Add a positive trace-scaled ridge to noise covariance, solve the symmetric generalized eigenproblem, and keep descending eigenvalues. Default minimum 300 and maximum 20,000 pairs; fail transparently on shortage. Save pairs, covariance, eigenvalues and components.
- SVM compact candidates C={1,10}, gamma={scale,.01}; RF={100,200} trees, depth24, leaf size2. Validation macro-F1 chooses the winner; no train+validation refit and no test-guided choices. Upper fit budget 61/seed, 305 over five seeds before dimension feasibility skips.
- One worker, bounded BLAS threads, 256-MB SVM cache, 4,096-row prediction batches, at most one spatial matrix cache. Principal maps only for seed7 by default.
- Optional block evaluation assigns entire 40-pixel blocks, then erodes each split region by buffer2; no overlapping neighbourhood footprints. It is approximate in pixel proportions and not guaranteed class-stratified. Fail on missing classes; do not adaptively retry seeds.
- Classification metrics use a fixed nine-class order and zero division=0. Aggregate SD uses ddof1, with undefined one-run SD preserved. E1/E4/E5 ablations are paired by seed and split digest. Independently selected feature counts mean these are tuned-family, rather than perfectly fixed-band, ablations.
- Completion markers are atomic and written after per-method artifacts. Candidate audits survive a failure, but incomplete candidate models are not checkpointed; the failed method's search restarts. Source, configuration, data SHA256 and library versions prevent mixing incompatible runs. Duplicate run lock is exclusive.
- Timings separate selected fit, accumulated preprocessing, search wall time and batched test prediction. Shared caches are order-dependent and selected spatial-matrix reconstruction is outside prediction timing; these are not isolated cold-start benchmarks.

## Verification executed and actual outcomes

All commands were run in this workspace. Cache prefixes were directed to `/tmp` where needed by the filesystem sandbox.

| Check | Actual outcome |
|---|---|
| Official file download, ZIP CRC/size verification | Passed for both LongKou files |
| Downloader repeat invocation | Passed; verified existing CRC/size and skipped both file transfers without overwriting |
| Direct `load_scene` on actual files | Passed, expected shapes/variables/counts, finite inputs |
| `.venv/bin/python src/main.py --validate-data` | Passed, exit0; wrote `outputs/data_validation.json`; no model fitting |
| `.venv/bin/python -m src.main --dry-run` | Passed, exit0; reported disabled training and 305 maximum candidate fits |
| `.venv/bin/python -m src.main --smoke-test` | Passed, exit0; 36×36×12 synthetic arrays, spatial shape and metric arithmetic, zero classifiers fitted |
| Initial unittest suite | 18/18 passed; later expanded for orchestration and a library regression |
| Final `.venv/bin/python -m unittest discover -s tests -v` | **21/21 passed**, 2.192 seconds, after dependency fix; no real classifier fitting |
| E0–E6 runner branches | Passed with `NonLearningStub`, including all artifact generators, save/resume and incompatibility rejection; temporary outputs deleted automatically; not scientific experiments |
| Held-out perturbation test | Passed: PCA, MNF covariance/projection and Fisher/correlation order invariant when validation/test measurements change |
| Finite small BLAS arithmetic regression | Passed with floating-point exceptions set to raise and independent einsum reference |
| `.venv/bin/python -m pip check` | Passed: no broken requirements |
| Python compilation of src/scripts/tests | Passed with writable `/tmp` bytecode cache |
| Final source/document inspection | Passed for 28 source/document files: Python AST, JSON parsing, terminal newlines and trailing whitespace; training flag false and no research metric/model artifacts |
| MATLAB launcher static contract | Passed; runtime MATLAB execution unavailable |
| Actual SVM/RF/majority fitting | Not executed, intentionally prohibited |
| Optional `--smoke-train` | Not executed, intentionally prohibited |
| Five-seed real experiments / spatial-block experiments | Not executed, intentionally prohibited |

The final unittest run's console capture is `/tmp/frss-verification.log`. It contains an expected parser-error message from the test asserting that `--run` refuses training; the test itself passes. Negative-path log messages from deliberately incompatible resume tests are captured as assertions. No metrics from a non-learning stub are retained in the repository as research output.

## Environment issues encountered and fixes

1. Sandboxed terminal DNS initially failed. Approved network access was used only to retrieve the official data and install packages into project-local `.venv`.
2. The official server rejected urllib's default Python user-agent with HTTP403. Retrying with a browser user-agent succeeded; that header is included in the downloader.
3. Python's configured bytecode cache pointed outside writable paths. Repeating compilation/tests with `PYTHONPYCACHEPREFIX=/tmp/frss-pycache` resolved this; `MPLCONFIGDIR=/tmp/frss-matplotlib` handled figure caches.
4. Initially resolved NumPy2.0.2 produced divide/overflow/invalid warnings even for small finite matrix multiplication. This was independently reproduced outside project code, consistent with reported [macOS BLAS floating-point issues](https://github.com/numpy/numpy/issues/28687). Project-local NumPy was changed to1.26.4, `requirements.txt` now requires `<2`, and the final arithmetic regression passes without those warnings. Warnings were not suppressed.
5. Initial Matplotlib imports emitted upstream Pyparsing deprecation warnings; these are not hidden by project code. Pip reports its default user cache is unwritable under the sandbox; it simply disables caching, and package checks succeed.
6. Strictly absent-class summaries initially held NumPy scalar IDs; serialization was hardened to Python integers. Existing valid downloads can now be reverified and skipped rather than blocking retries. Preprocessing time includes first-time selector construction.

Exact final package versions are in `requirements-lock.txt`; principal packages: NumPy1.26.4, SciPy1.13.1, scikit-learn1.6.1, Matplotlib3.9.4, pandas2.2.3, h5py3.12.1, joblib1.5.3, threadpoolctl3.7.0. No global package upgrades/uninstalls were made.

## Scientific and submission status checklist

- [x] Actual dataset loader verified on official LongKou files.
- [x] Labels/background and observed class mapping/counts verified.
- [x] E0–E6 source implemented and orchestration checked with a non-learning stub.
- [x] Leakage assertions and train-only transformation tests pass.
- [x] PCA and genuine-noise MNF numerical behavior checked on small arrays.
- [x] E1/E4/E5 ablation, all required metric and visualization generators implemented.
- [x] Five seeds and parameter choices recorded in central configuration.
- [x] Unit tests and non-training structural smoke test executed successfully.
- [x] Main.m created and statically checked; direct Python entry validated.
- [x] README, scientific framework, contribution template and eight-slide outline created.
- [x] No fabricated research scores, citations or team identities added.
- [ ] Real five-seed OA/F1/per-class tables and mean ± SD — pending explicit training authorization.
- [ ] Actual model confusion matrices, prediction maps, runtime/feature plots — pending real experiments.
- [ ] Spatial-block feasibility on LongKou — not yet evaluated.
- [ ] Actual MATLAB invocation — requires MATLAB installation and later testing.
- [ ] Completed empirical report and contributions — pending measurement and team input.
- [ ] Public repository visibility and final commit hash — pending publication/verification.

## Remaining steps and exact handoff

For a non-training review, use `--dry-run`, `--validate-data`, `--smoke-test` and unittest commands in README. Data is already present on this machine. Another checkout needs the downloader and environment installation.

When the owner explicitly decides to train later, use `.venv/bin/python -m src.main --run --allow-training`, or enable `execution.allow_training` in configuration before MATLAB `Main`. This is CPU-compatible but can take substantial time; inspect the dry-run fit budget first. Keep the predefined five-seed protocol for final results. If code/configuration changes after any run, choose a new output directory to preserve provenance.

After real experiments, inspect `outputs/results.md`, all `metrics/*.csv`, per-seed selection audits and `figures/`; then replace the report's pending entries with measured values and discuss observed gains or losses. Fill real contribution details, verify the course permits a Python implementation launched through MATLAB, test Main.m in MATLAB, and publish/verify the repository URL for final submission.

**Deviations from the supplied master prompt:** real training and synthetic classifier smoke training were deliberately omitted because the owner's direct instruction prohibited them. MATLAB runtime testing is unavailable. No request was made to publish changes, so work remains local. No course PDF was present. Empirical sections remain marked pending rather than inventing results. The report's Green et al. citation title/authors/year were verified in the search-indexed original scan, but full publisher/scan access was unavailable and is disclosed in the reference.
