"""Build the empirical report and portable evidence bundle after a complete audit."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_results import audit

METHODS = {"E0": "Majority", "E1": "Full-spectrum SVM", "E2": "PCA-SVM",
           "E3": "MNF-SVM", "E4": "Selected-band SVM", "E5": "FRSS-SVM", "E6": "Spatial Random Forest"}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def pm(mean, sd, percent=True):
    factor = 100 if percent else 1
    return f"{factor * mean:.3f} ± {factor * sd:.3f}"


def main():
    output, report = ROOT / "outputs", ROOT / "report"
    verification = audit(output)
    manifest = json.loads((output / "manifest.json").read_text())
    summary = pd.read_csv(output / "metrics/summary.csv")
    assert summary.complete.all() and len(summary) == 7
    summary = summary.set_index("method").sort_index()
    raw = pd.read_csv(output / "metrics/per_seed.csv")
    classes = pd.read_csv(output / "metrics/per_class_summary.csv")
    deltas = pd.read_csv(output / "metrics/ablation_deltas.csv")
    records = [json.loads(p.read_text()) for p in sorted((output / "metrics/pixel").glob("seed_*/E*.json"))]
    e1, e5 = summary.loc["E1"], summary.loc["E5"]
    best = summary.macro_f1_mean.idxmax()
    delta_f1 = (e5.macro_f1_mean - e1.macro_f1_mean) * 100
    delta_oa = (e5.accuracy_mean - e1.accuracy_mean) * 100
    # Preserve the reviewed methodology, updating only the measured execution status.
    original = (report / "final_report.md").read_text(encoding="utf-8")
    intro = original.split("## 13.")[0]
    if "S20240020290" not in intro:
        intro = intro.replace("**Pattern Recognition semester project — deadline: 5 November 2026.**",
            "**Pattern Recognition semester project — deadline: 5 November 2026.**\n\n"
            "**Himanshu Sinha — S20240020290**\n\n**Ghule Yash Mahesh — S20240020288**")
    start = intro.index("**Status as of")
    end = intro.index("## 1.")
    intro = intro[:start] + "**Status as of 3 November 2026:** all seven methods completed on all five predeclared seeds. Real held-out predictions were independently audited. Results are for a transductive, same-scene random-pixel protocol.\n\n" + intro[end:]
    abstract_start, abstract_end = intro.index("## 1."), intro.index("## 2.")
    abstract = ("## 1. Abstract\n\n"
                "This study evaluates classical spectral and spatial features for nine-class land-cover recognition on "
                "WHU-Hi-LongKou (550 × 400 pixels, 270 bands). Fisher ranking and strict correlation pruning select "
                "bands; local means and standard deviations augment them for an RBF SVM (FRSS-SVM). "
                "Seven methods share stratified 70/15/15 partitions, with 9,000 capped training pixels and "
                "30,682 held-out test pixels per seed. Validation macro-F1 selects hyperparameters. "
                f"Across five seeds, FRSS-SVM achieves {pm(e5.accuracy_mean, e5.accuracy_std)}% accuracy and "
                f"{pm(e5.macro_f1_mean, e5.macro_f1_std)}% macro-F1. "
                f"Its macro-F1 difference from the full-spectrum SVM is {delta_f1:+.3f} percentage points. "
                f"{METHODS[best]} has the highest mean macro-F1 among the evaluated methods. "
                "Reported variability is sample SD across splits, not independent-scene uncertainty.\n\n")
    intro = intro[:abstract_start] + abstract + intro[abstract_end:]
    intro = intro.replace("These are hypotheses to test. No improvement is guaranteed by the design.", "These questions are assessed using the held-out results in Sections 13-16; improvements are not assumed.")
    budget_marker = "The default upper budget is" if "The default upper budget is" in intro else "The predeclared upper budget is"
    old_start = intro.index(budget_marker)
    old_end = intro.index("## 11.")
    intro = intro[:old_start] + (f"The predeclared upper budget is 305 candidate fits. Strict feature feasibility reduced the actual search to "
                f"{verification['candidate_fits']} fits, all completed. One worker, bounded BLAS threads, a 256-MB "
                "SVM cache and 4,096-row prediction batches constrain resources. Four bounded workers predict independent batches; "
                "classifier fitting and BLAS remain at one thread. Serial/parallel prediction equality was regression tested. The measured environment is "
                "Windows, Python 3.12.5, Intel Core i5-12450H, approximately 16 GB RAM; GPU acceleration was not used. "
                "Exact dependencies, source hashes, dataset hashes and configuration are preserved in "
                "report/evidence/manifest.json and requirements-windows-lock.txt.\n\n") + intro[old_end:]
    intro = intro.replace("This protocol is coded but has not been evaluated on the real scene; random-pixel findings must not be described as independent geographic generalization.",
                "A preflight check on the real scene found missing validation classes for seeds 7, 19 and 73; seeds 42 and 101 were feasible. No block classifiers were trained, and seeds were not replaced or selectively reported. Thus no complete five-seed spatial sensitivity claim is possible with this predefined block design. Random-pixel findings must not be described as independent geographic generalization. Details are in report/evidence/protocol_feasibility.json.")
    rows = [[m + " " + METHODS[m], pm(r.accuracy_mean, r.accuracy_std),
             pm(r.macro_f1_mean, r.macro_f1_std), f"{r.feature_count_mean:.1f}", "5/5"]
            for m, r in summary.iterrows()]
    sections = [intro, "## 13. Measured test results\n",
                "Accuracy and macro-F1 are percentages; ± denotes sample SD over seeds 7, 19, 42, 73 and 101. Every method used the same test split within a seed.\n",
                table(["Method", "Accuracy (%)", "Macro-F1 (%)", "Features (mean)", "Runs"], rows),
                f"\nThe highest mean macro-F1 is {METHODS[best]} ({100 * summary.loc[best].macro_f1_mean:.3f}%). "
                f"FRSS-SVM differs from the full-spectrum SVM by {delta_oa:+.3f} accuracy points and {delta_f1:+.3f} macro-F1 points. "
                "These are descriptive comparisons; no significance test or independent geographic validation is claimed.\n",
                "![Figure 1. Five-seed test macro-F1 with sample SD.](figures/macro_f1_comparison.png)\n",
                "## 14. Per-class performance and confusion\n",
                "The following values are five-seed means in percent. Complete precision, recall, F1 and SD for all seven methods are retained in evidence/per_class_summary.csv.\n"]
    per_class_rows = []
    for _, row in classes[classes.method == "E5"].sort_values("class_id").iterrows():
        base = classes[(classes.method == "E1") & (classes.class_id == row.class_id)].iloc[0]
        per_class_rows.append([row.class_name, f"{100 * base.recall_mean:.2f}",
                               f"{100 * row.precision_mean:.2f}", f"{100 * row.recall_mean:.2f}",
                               pm(row.f1_mean, row.f1_std)])
    sections.append(table(["Class", "E1 recall", "E5 precision", "E5 recall", "E5 F1 ± SD"], per_class_rows))
    weakest = classes[classes.method == "E5"].sort_values("precision_mean").head(2)
    sections.append("\nThe lowest mean E5 precisions are " + "; ".join(
        f"{r.class_name}: {100 * r.precision_mean:.2f}% precision versus {100 * r.recall_mean:.2f}% recall"
        for r in weakest.itertuples()) +
        ". High recall does not imply few false positives. Errors from a large class can noticeably lower a smaller class's precision, "
        "which helps explain the gap between overall accuracy and macro-F1.\n")
    pooled = sum(np.asarray(r["confusion_matrix"]) for r in records if r["method"] == "E5")
    errors = pooled.copy()
    np.fill_diagonal(errors, 0)
    pairs = sorted(((int(errors[i, j]), i, j) for i in range(9) for j in range(9) if i != j), reverse=True)[:3]
    from src.data_loader import CLASS_NAMES
    sections.append("\nThe largest pooled directed E5 errors are " + "; ".join(
        f"{CLASS_NAMES[i]} → {CLASS_NAMES[j]} ({count} predictions)" for count, i, j in pairs) +
        ". These counts pool repeated split evaluations, so a scene pixel can contribute more than once. They are not counts of independent pixels across scenes.\n")
    sections += ["![Figure 2. FRSS-SVM held-out confusion matrix for the predeclared display seed 7; annotations are row percentages.](figures/E5_confusion_normalized.png)\n",
                 "![Figure 3. Ground truth and FRSS-SVM descriptive map, seed 7. The full map includes training pixels; only held-out predictions determine reported metrics.](figures/E5_comparison.png)\n",
                 "![Figure 4. FRSS-SVM errors on held-out test pixels only, seed 7.](figures/E5_test_errors.png)\n",
                 "## 15. Paired ablation and selected representations\n",
                 "Differences are proposed minus baseline, in percentage points. Each pair shares the seed and exact split digest.\n"]
    paired_rows = []
    for comparison, frame in deltas.groupby("comparison", sort=True):
        paired_rows.append([comparison, pm(frame.delta_accuracy.mean(), frame.delta_accuracy.std()),
                            pm(frame.delta_macro_f1.mean(), frame.delta_macro_f1.std())])
    sections.append(table(["Comparison", "Accuracy difference ± SD", "Macro-F1 difference ± SD"], paired_rows))
    spatial_delta = deltas[deltas.comparison == "E5-E4"].delta_macro_f1
    sections.append(f"\nAdding local mean/std features to the selected-band family changes macro-F1 by "
        f"{100 * spatial_delta.mean():+.3f} ± {100 * spatial_delta.std():.3f} percentage points. "
        f"Against the full-spectrum SVM the change is {delta_f1:+.3f} points, with "
        f"{100 * (1 - e5.feature_count_mean / 270):.2f}% fewer classifier input features on average. "
        "Thus recovery of information lost by aggressive band pruning should be distinguished from improvement over all-band classification.\n")
    selections = []
    for seed in manifest["config"]["seeds"]:
        seed_records = {r["method"]: r for r in records if r["seed"] == seed}
        selections.append([seed] + [seed_records[m]["feature_count"] for m in ["E2", "E3", "E4", "E5", "E6"]])
    sections += ["\n" + table(["Seed", "E2 PCA", "E3 MNF", "E4 bands", "E5 features", "E6 features"], selections),
                 "\nE5/E6 feature counts include raw bands, means and standard deviations, so their band counts are one third of the displayed count. "
                 "Strict correlation pruning may leave fewer bands than the smallest nominal request; in that case the predeclared fallback uses all retained bands. "
                 "This is not a post-test hyperparameter change. Every chosen parameter and zero/one-based band index is recorded in evidence/pixel/. "
                 "E4/E5 searches tune counts independently, so the ablation compares tuned families rather than an enforced fixed-band intervention.\n",
                 "![Figure 5. Training-only Fisher scores and selected bands for FRSS-SVM, seed 7.](figures/E5_fisher.png)\n",
                 "## 16. Runtime and dimensionality\n",
                 "Times are seconds, mean ± sample SD. Selected fit is one winning classifier fit; search includes all candidate fits and validation predictions.\n"]
    sections.append(table(["Method", "Selected fit", "Preprocessing", "Search", "Test prediction"],
        [[m, pm(r.fit_seconds_mean, r.fit_seconds_std, False), pm(r.preprocessing_seconds_mean, r.preprocessing_seconds_std, False),
          pm(r.search_seconds_mean, r.search_seconds_std, False), pm(r.prediction_seconds_mean, r.prediction_seconds_std, False)] for m, r in summary.iterrows()]))
    sections += ["\nCaches reuse decompositions and spatial representations. Preprocessing therefore depends on execution order and is not a cold-start benchmark. "
                 "Test prediction includes batched transformations but excludes reconstruction of the winning spatial cache. Plotting, disk I/O and full-scene maps are excluded. "
                 "The computer was also used for documentation and verification, so wall times should be treated as practical observations rather than controlled hardware benchmarks.\n",
                 "## 17. Limitations and threats to validity\n",
                 "This is a transductive same-scene experiment. Random-pixel neighbours can share measurements across partitions, and geographic autocorrelation can inflate accuracy. "
                 "Training-only fitting prevents statistical-transform leakage, but does not make spatially neighbouring test pixels geographically independent. "
                 "The block protocol failed class coverage on three predeclared seeds; no geographic robustness result is claimed. "
                 "Repeated test sets overlap. Five-run SD captures split sensitivity, not independent-scene uncertainty. "
                 "The nearly balanced training cap alters class priors. Local noise differences may include signal edges; spatial statistics can blur boundaries. "
                 "The small validation grid may miss useful settings. The proposed method combines established algorithms and does not introduce a new SVM optimization objective.\n",
                 "## 18. Conclusion\n",
                 f"All 35 method/seed evaluations completed and passed independent prediction-based checks. FRSS-SVM achieved "
                 f"{pm(e5.accuracy_mean, e5.accuracy_std)}% accuracy and {pm(e5.macro_f1_mean, e5.macro_f1_std)}% macro-F1. "
                 f"Relative to the full-spectrum SVM, its macro-F1 changed by {delta_f1:+.3f} percentage points while using "
                 f"{e5.feature_count_mean:.1f} features on average versus 270. "
                 f"The best observed mean macro-F1 belongs to {METHODS[best]}. "
                 "Future work should predeclare a class-covering geographic split, use an independent scene, and compare matched-band spatial ablations.\n"]
    references = original.split("## 19.", 1)[1].split("## 20.", 1)[0]
    references = references.replace("These references establish method origins; none establishes the unmeasured performance of this proposed combination.",
                                    "These references establish method origins; the performance reported above comes solely from this project's saved experiments.")
    sections.append("## 19." + references)
    contributions = (report / "individual_contributions.md").read_text(encoding="utf-8")
    sections += ["## 20. Contributions and reproducibility appendix\n", contributions.split("\n", 1)[1],
                 "\nThe project repository was verified publicly accessible on 2 November 2026: "
                 "[source repository](https://github.com/himanshu09120/Spectral-Spatial-Feature-Selection-for-Hyperspectral-Land-Cover-Classification-Using-an-Enhanced-SVM). "
                 "This local completion has not been pushed; the remote may still contain the earlier version.\n",
                 "From a prepared Python 3.12 environment, run `python scripts/download_data.py`, then "
                 "`python -m src.main --run --allow-training`. In MATLAB, `Main` starts the complete pipeline and displays summary tables and figures. "
                 "Exact matching completed runs resume without retraining. A source/configuration mismatch requires a fresh output directory.\n",
                 f"Run fingerprint: `{manifest['fingerprint']}`. "
                 "The portable report/evidence bundle retains the manifest, independently verified audit, all per-seed metrics, configuration selections, "
                 "band indices, class summaries and paired ablations. Large model files and pixel arrays remain under ignored outputs/. "
                 "The report can be regenerated with `python scripts/finalize_report.py`, then `python scripts/render_report.py`.\n"]
    evidence = report / "evidence"
    evidence.mkdir(exist_ok=True)
    for name in ["manifest.json", "audit.json", "data_validation.json", "protocol_feasibility.json"]:
        shutil.copy2(output / name, evidence / name)
    for path in (output / "metrics").rglob("*"):
        if path.suffix in (".csv", ".json"):
            dest = evidence / path.relative_to(output / "metrics")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    figures = report / "figures"
    figures.mkdir(exist_ok=True)
    for name in ["macro_f1_comparison.png", "accuracy_comparison.png", "runtime_vs_features.png"]:
        shutil.copy2(output / "figures/pixel" / name, figures / name)
    for name in ["E5_confusion_normalized.png", "E5_confusion_counts.png", "E5_comparison.png", "E5_test_errors.png", "E5_fisher.png"]:
        shutil.copy2(output / "figures/pixel/seed_7" / name, figures / name)
    (report / "final_report.md").write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
    presentation = f"""# Eight-slide presentation with measured talking points

1. **Problem and objective.** Nine-class agricultural land-cover mapping from hyperspectral imagery. Explain spectral redundancy and the motivation for local spatial context. Presenters: Himanshu Sinha (S20240020290) and Ghule Yash Mahesh (S20240020288).
2. **Dataset.** WHU-Hi-LongKou: 550 × 400 × 270, 204,542 labelled pixels, nine classes. Label 0 is excluded. Show `outputs/figures/dataset/ground_truth.png` and the observed class counts. Source: https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm.
3. **Evaluation design.** Five fixed seeds; stratified 70/15/15 pool/validation/test; 9,000 common fitting pixels per seed; 30,682 held-out test pixels. Fit transformations on training only; select by validation macro-F1; score test after selection.
4. **Methods.** E0 majority; E1 full-spectrum SVM; E2 PCA-SVM; E3 MNF-SVM; E4 Fisher/correlation-selected SVM; E5 selected raw/mean/std FRSS-SVM; E6 spatial Random Forest. Explain the strict-pruning fallback using the actual counts in report Section 15.
5. **Overall results.** FRSS-SVM accuracy: {pm(e5.accuracy_mean, e5.accuracy_std)}%; macro-F1: {pm(e5.macro_f1_mean, e5.macro_f1_std)}%. Show `report/figures/macro_f1_comparison.png`. Highest observed mean macro-F1: {METHODS[best]}. Do not claim every proposed method beats every baseline.
6. **Ablation and cost.** E5 minus E1: {delta_f1:+.3f} macro-F1 percentage points and {delta_oa:+.3f} accuracy points. E5 uses {e5.feature_count_mean:.1f} features on average. Use the paired ablation table and `report/figures/runtime_vs_features.png`; distinguish search time from winning-fit time.
7. **Class errors and maps.** Show `report/figures/E5_confusion_normalized.png` and `report/figures/E5_comparison.png`. Discuss the largest directed errors listed in report Section 14. The complete map includes training pixels and is descriptive; test errors are separately masked.
8. **Conclusion, limitations and contributions.** Same-scene random pixels share contextual measurements; three planned block splits lacked validation classes. No independent-region claim. Equal contribution allocation is recorded in the appendix, with AI assistance disclosed. Provide the repository link and explain reproduction through `Main` or the Python CLI.

All numerical talking points derive from the audited run. See `report/final_report.md` for full references, exact metrics, definitions and limitations.
"""
    (report / "presentation_outline.md").write_text(presentation, encoding="utf-8")
    files = {str(p.relative_to(report)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in [evidence, figures] for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"}
    (evidence / "SHA256SUMS.json").write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")
    print(f"Finalized {len(records)} results; report and evidence saved in {report}")


if __name__ == "__main__":
    main()
