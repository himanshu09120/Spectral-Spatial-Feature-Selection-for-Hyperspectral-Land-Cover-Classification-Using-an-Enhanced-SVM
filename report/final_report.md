# Spectral–Spatial Feature Selection for Hyperspectral Land-Cover Classification Using an Enhanced SVM

**Pattern Recognition semester project — deadline: 5 November 2026.**

**Status as of 29 October 2026:** methodology and software implemented; official dataset downloaded and inspected; no classifiers trained at the owner's request. All classification findings, prediction maps and runtime comparisons are **pending**. This document is a scientific report framework, not a completed empirical study.

## 1. Abstract

This project investigates land-cover classification from hyperspectral measurements using classical pattern recognition. The proposed Fisher-guided Redundancy-Reduced Spectral–Spatial SVM (FRSS-SVM) combines supervised band ranking, correlation pruning and local measurement statistics. The experiment design compares majority, full-spectrum SVM, PCA-SVM, MNF-SVM, selected-band SVM and a spatial Random Forest on WHU-Hi-LongKou. A common split and training cap support fair comparisons, while validation macro-F1 selects configurations. Five seeds and an optional buffered block protocol address repeatability and spatial dependence. The official files satisfy the course's size and class requirements. Classification performance has not been measured; superiority remains a research hypothesis.

## 2. Introduction and motivation

Crop mapping can support agricultural inventories and land-use assessment. A hyperspectral pixel contains many closely spaced spectral measurements, which may help distinguish crops but can also be redundant. Spatial context may reduce local ambiguity at the cost of smoothing boundaries. This project examines that trade-off with interpretable feature selection and classical classifiers, avoiding reliance on a GPU.

## 3. Problem statement

Given a cube X ∈ R^(H×W×B), predict one of nine classes for each labelled pixel. The scene's label 0 denotes unlabelled background and is excluded from learning and evaluation. The task is supervised multiclass classification of an agricultural scene, rather than inference on an independently acquired geographic region.

## 4. Objectives and research questions

1. Does supervised, redundancy-aware band selection preserve classification quality with fewer inputs?
2. Do local means and standard deviations improve per-class recognition relative to spectral-only features?
3. How do selected bands compare with PCA and noise-adjusted components under a shared training budget?
4. What accuracy, feature-size and runtime trade-offs arise between SVM and Random Forest?

These are hypotheses to test. No improvement is guaranteed by the design.

## 5. Related work

Hu et al. introduce WHU-Hi as a benchmark for detailed crop recognition using UAV imagery [1]. The official resource page supplies scene and class metadata [2]. Their published scores are not directly comparable with this project's split and are not reused as our findings.

Chang and Lin describe LIBSVM's support-vector classifiers and multiclass implementation [3]. We use the scikit-learn SVC interface to that classical solver. Green et al. develop a noise-aware image transformation [4]; our implementation follows the noise-whitening/generalized-eigenproblem interpretation rather than renaming ordinary PCA. Breiman's Random Forest combines randomized tree predictors [5], providing an alternative nonlinear classifier. The proposed contribution is an evaluated combination of established tools, not invention of Fisher ranking, local statistics or SVM.

## 6. Dataset and measured distribution

The official LongKou MAT files were downloaded from the publisher's ZIP and checked for size and CRC integrity. Actual variable inspection found a 550×400×270 cube, a 550×400 label map and the counts below, matching the official class table [2]. There are 15,458 background and 204,542 labelled pixels. Thus the dataset exceeds 10 attributes, 1,000 instances and 3 classes.

| ID | Class | Observed labelled pixels |
|---:|---|---:|
| 1 | Corn | 34,511 |
| 2 | Cotton | 8,374 |
| 3 | Sesame | 3,031 |
| 4 | Broad-leaf soybean | 63,212 |
| 5 | Narrow-leaf soybean | 4,151 |
| 6 | Rice | 11,854 |
| 7 | Water | 67,056 |
| 8 | Roads and houses | 7,124 |
| 9 | Mixed weed | 5,229 |

The minimum class is much smaller than the maximum, so macro-F1 and class-wise recall complement overall accuracy. No wavelength calibration vector is assumed by the code; spectral plots label band indices. See `data/DATASET_VALIDATION.json` and `data/DOWNLOAD_PROVENANCE.json` for inspection and hashes.

## 7. Preprocessing and leakage prevention

Pixel splits are stratified 70% training, 15% validation and 15% test, with seeds 7, 19, 42, 73 and 101. The common training pool is capped at 1,000 pixels per class for E1–E6, once per seed. Every fitted mean, standard deviation, Fisher score, correlation, PCA covariance and MNF covariance uses this capped set only. E0 uses original training-pool class proportions to provide the correct majority reference. All actual indices and class counts are persisted at execution time.

Validation labels select configurations; the test set is scored after selection. The winning model is retained without a train+validation refit. Scaling does not use full-scene statistics. Full-scene class spectra and colour stretches are descriptive figures only and do not feed learning. Local statistics use measurements, never labels; random-pixel neighbourhoods can nevertheless share measurements across splits, so this protocol is explicitly transductive and geographically optimistic.

## 8. Mathematical formulation

For band j, with class size n_c, class mean μ_cj, population variance σ²_cj and global training mean μ_j:

F_j = [Σ_c n_c(μ_cj − μ_j)²] / [Σ_c n_c σ²_cj + ε].

The implementation uses ε=10⁻¹² and population within-class variances. Bands are ranked descending, then greedily retained if their absolute training Pearson correlation with every previously retained band is ≤0.95. Constant bands are excluded. Strictly pruned sets are never padded with redundant bands.

PCA diagonalizes the covariance of training-standardized inputs and retains the largest-variance directions. The maximum eligible component count is decomposed once; smaller prefixes are reused. Component outputs are rescaled using training statistics before SVM fitting.

For MNF, training-neighbour differences d_pq=(x_p−x_q)/√2 estimate noise under independent equal-covariance noise and locally smooth signal. Both endpoints belong to the capped training set. Differences are expressed in training-standardized units. With S_n=cov(d), use C_n=S_n+α max(tr(S_n)/B, machine epsilon)I, α=0.001. Solve C_x v=λ C_n v, normalize vᵀC_n v=1 and sort λ descending. This is noise whitening followed by PCA in the whitened space; large λ identifies high total-to-noise variance. At least 300 eligible pairs are required and at most 20,000 are sampled with the recorded seed. Too few pairs stop E3 clearly. This estimator can confound signal edges with sensor noise and does not constitute instrument-specific noise calibration [4].

For binary SVM subproblems, the soft-margin objective is ½||w||²+CΣ_i ξ_i subject to y_i(wᵀφ(x_i)+b)≥1−ξ_i and ξ_i≥0. The RBF kernel is K(x,x′)=exp(−γ||x−x′||²); LIBSVM combines binary subproblems for multiclass prediction [3]. Random Forest predicts through an ensemble of randomized classification trees [5].

## 9. Proposed FRSS-SVM

For each selected band, compute a 3×3 reflected-border local mean m and population standard deviation s. Concatenate z=[x,m,s], giving 3k features for k bands; 30 bands produce 90 features. Statistics are computed only for selected bands. Variance uses double-precision accumulation with roundoff negatives clamped to zero before square roots. Train-only scaling precedes the same RBF-SVM family used in the principal baseline.

## 10. Experimental setup and hyperparameters

| Method | Feature candidates | Model candidates |
|---|---|---|
| E0 | Majority prior, no informative features | Dummy most frequent |
| E1 | All bands | RBF SVM |
| E2 | PCA: 20, 50 components | RBF SVM |
| E3 | MNF: 20, 50 components | RBF SVM |
| E4 | 15, 30, 45, 60 selected bands | RBF SVM |
| E5 | Selected bands with mean/std: 45, 90, 135, 180 features | RBF SVM |
| E6 | Same spatial feature candidate family | RF: 100, 200 trees |

SVM uses C={1,10}, γ={scale,0.01}; RF uses depth≤24 and leaf size≥2. All searches maximize validation macro-F1, with deterministic ties favoring earlier smaller counts. Infeasible dimensions are skipped. If no requested band count survives strict pruning, the entire nonempty pruned set is the predeclared fallback. An optional linear SVM is disabled by default.

The default upper budget is 305 candidate fits over five seeds; no such fits have been performed. One worker and bounded BLAS threads, 256-MB SVM cache and batched predictions constrain resource use. Exact execution libraries and configuration are recorded in a fingerprint manifest when a run starts. The tested environment is macOS ARM64, Python 3.9.6, with compatible scientific wheels recorded in `requirements-lock.txt`. No MATLAB executable was found.

## 11. Spatial sensitivity protocol

An additional, opt-in block protocol assigns whole 40×40 pixel blocks to partitions with the configured proportions, then erodes each region by a 2-pixel buffer. Neighbourhood footprints are separated and unit checked. Block splits are not guaranteed class stratification or exact pixel proportions. Missing-class partitions raise an error rather than retrying seeds based on performance. This protocol is coded but has not been evaluated on the real scene; random-pixel findings must not be described as independent geographic generalization.

## 12. Evaluation metrics

Overall accuracy is correct predictions divided by test pixels. Class precision is TP/(TP+FP), recall TP/(TP+FN), and F1 their harmonic mean. Macro metrics give equal weight to the fixed nine-class ordering. Undefined ratios become zero consistently. Raw and row-normalized confusion matrices use true rows and predicted columns. Five-run means and sample SD (denominator n−1) are computed from saved measurements; a single run has undefined SD. Completion flags expose missing seeds.

## 13. Results and comparison tables — pending

**No classifier training, validation search or test evaluation has been run.** The following entries must remain pending until measured output exists.

| Method | OA mean ± sample SD | Macro-F1 mean ± sample SD | Test runs |
|---|---|---|---|
| E0 | Pending | Pending | 0/5 |
| E1 | Pending | Pending | 0/5 |
| E2 | Pending | Pending | 0/5 |
| E3 | Pending | Pending | 0/5 |
| E4 | Pending | Pending | 0/5 |
| E5 | Pending | Pending | 0/5 |
| E6 | Pending | Pending | 0/5 |

Future runs generate `outputs/results.md`, `metrics/per_seed.csv`, `metrics/summary.csv`, `metrics/per_class.csv` and `metrics/per_class_summary.csv`. Copy measured tables here only after verifying all five seeds and common test splits. No published benchmark values are substituted into this table.

## 14. Confusion matrices and prediction maps — pending

The renderer will save count and row-normalized matrices for every method and seed, with one shared class palette. E1/E5 maps are configured for seed 7; complete maps include training pixels and are descriptive. Error maps display test pixels only. Once measured outputs exist, identify confused class pairs, inspect crop boundaries and distinguish systematic errors from small-support effects. No confusion pattern has been observed yet.

## 15. Ablation study — pending

Compare E1 full-spectrum SVM, E4 selected spectral SVM and E5 spatial SVM. Saved `ablation_deltas.csv` uses paired seeds and the same test split, including OA, macro metrics, feature counts and runtimes. Report class-wise precision/recall beside aggregate changes. Since E4/E5 tune counts independently, this isolates tuned method families rather than an exact fixed-band spatial intervention; matched-count validation curves provide additional context. Gains, losses and uncertainty must all be reported honestly.

## 16. Runtime and dimensionality — pending

Report chosen classifier fit time, accumulated preprocessing, search wall time and batched test prediction time separately. Optional map rendering/prediction, I/O and plot generation are outside those timing fields. Training covariance decompositions and spatial caches are reused, so preprocessing timing depends on execution order; it is not a cold-run algorithm benchmark. Winning spatial-cache reconstruction is outside prediction timing. No measured runtime or compression ratio is claimed yet; feature counts will come from selected configurations rather than assumed nominal counts.

## 17. Limitations and threats to validity

One scene cannot establish robustness to a new sensor, season or region. Spatially correlated pixel splits can produce optimistic values; contextual measurements shared across partitions strengthen that concern. Capped, nearly balanced training alters original priors. Small validation grids can miss useful configurations. Noise differences assume local signal smoothness and approximate independent noise, which may fail at crop boundaries. Spatial statistics may blur small objects. Repeated split test sets overlap, so seed variation measures split sensitivity rather than independent-scene uncertainty. A fully trained end-to-end run and MATLAB execution remain unverified. The course's public repository requirement must be checked after publishing; a configured Git remote alone does not prove public access.

## 18. Conclusion and future work

The code implements a reproducible comparison of classical hyperspectral feature methods and validates the actual data structure. This establishes software readiness for experimentation, not evidence of classification improvement. The next scientific step is explicitly authorized CPU training, followed by five-seed analysis and careful interpretation of block-protocol feasibility. Future extensions can examine fixed-band ablations, alternate neighbourhood sizes selected on validation data, calibrated sensor-noise models and an independent scene.

## 19. Verified references

1. X. Hu, Y. Zhong, C. Luo and X. Wang. *WHU-Hi: UAV-borne hyperspectral with high spatial resolution (H2) benchmark datasets for hyperspectral image classification*. arXiv:2012.13920, version 2, 2021. [Author preprint](https://arxiv.org/abs/2012.13920).
2. Wuhan University RSIDEA. *WHU-Hi resource sharing*. [Official dataset and class table](https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm), checked 29 October 2026.
3. C.-C. Chang and C.-J. Lin. *LIBSVM: A Library for Support Vector Machines*. ACM Transactions on Intelligent Systems and Technology, 2011. [Author-maintained paper, updated 2022](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf).
4. A. A. Green, M. Berman, P. Switzer and M. D. Craig. *A Transformation for Ordering Multispectral Data in Terms of Image Quality with Implications for Noise Removal*. IEEE Transactions on Geoscience and Remote Sensing, 1988. [Publisher record, DOI 10.1109/36.3001](https://doi.org/10.1109/36.3001); [indexed original paper scan](https://citeseerx.ist.psu.edu/document?doi=6ae00ebd3a91c0667c79c39035b5163025bcfcad&repid=rep1&type=pdf). Title/authors/year were verified in the indexed original-paper excerpt; the publisher full text and scan fetch were inaccessible in this environment.
5. L. Breiman. *Random Forests*. Machine Learning, 2001. [Author-hosted original paper](https://www.stat.berkeley.edu/users/breiman/randomforest2001.pdf).

These references establish method origins; none establishes the unmeasured performance of this proposed combination. No unverified DOI or accuracy claim is added.

## 20. Appendix: contributions and reproducibility

Complete [individual_contributions.md](individual_contributions.md) using actual team members and evidence. Do not assign authorship based on this template.

Repository configured in Git: [project repository](https://github.com/himanshu09120/Spectral-Spatial-Feature-Selection-for-Hyperspectral-Land-Cover-Classification-Using-an-Enhanced-SVM). Public accessibility and final commit hash: **pending publication/verification**.

Reproduction starts with the official downloader, environment setup and non-training checks in `README.md`. A future experiment is explicitly started with `.venv/bin/python -m src.main --run --allow-training`. MATLAB users enable training in configuration before `Main`; `Main('validate-data')` performs inspection only. Retain the run manifest, split indices, selection audits and raw metrics with the final submission. The present handoff records zero trained classifiers and no measured classification outcomes.
