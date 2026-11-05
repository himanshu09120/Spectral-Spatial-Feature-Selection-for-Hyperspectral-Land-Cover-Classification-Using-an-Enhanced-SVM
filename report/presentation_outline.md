# Eight-slide presentation with measured talking points

1. **Problem and objective.** Nine-class agricultural land-cover mapping from hyperspectral imagery. Explain spectral redundancy and the motivation for local spatial context. Presenters: Himanshu Sinha (S20240020290) and Ghule Yash Mahesh (S20240020288).
2. **Dataset.** WHU-Hi-LongKou: 550 × 400 × 270, 204,542 labelled pixels, nine classes. Label 0 is excluded. Show `outputs/figures/dataset/ground_truth.png` and the observed class counts. Source: https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm.
3. **Evaluation design.** Five fixed seeds; stratified 70/15/15 pool/validation/test; 9,000 common fitting pixels per seed; 30,682 held-out test pixels. Fit transformations on training only; select by validation macro-F1; score test after selection.
4. **Methods.** E0 majority; E1 full-spectrum SVM; E2 PCA-SVM; E3 MNF-SVM; E4 Fisher/correlation-selected SVM; E5 selected raw/mean/std FRSS-SVM; E6 spatial Random Forest. Explain the strict-pruning fallback using the actual counts in report Section 15.
5. **Overall results.** FRSS-SVM accuracy: 97.131 ± 0.201%; macro-F1: 92.936 ± 0.476%. Show `report/figures/macro_f1_comparison.png`. Highest observed mean macro-F1: MNF-SVM. Do not claim every proposed method beats every baseline.
6. **Ablation and cost.** E5 minus E1: -0.307 macro-F1 percentage points and +0.164 accuracy points. E5 uses 18.0 features on average. Use the paired ablation table and `report/figures/runtime_vs_features.png`; distinguish search time from winning-fit time.
7. **Class errors and maps.** Show `report/figures/E5_confusion_normalized.png` and `report/figures/E5_comparison.png`. Discuss the largest directed errors listed in report Section 14. The complete map includes training pixels and is descriptive; test errors are separately masked.
8. **Conclusion, limitations and contributions.** Same-scene random pixels share contextual measurements; three planned block splits lacked validation classes. No independent-region claim. Equal contribution allocation is recorded in the appendix, with AI assistance disclosed. Provide the repository link and explain reproduction through `Main` or the Python CLI.

All numerical talking points derive from the audited run. See `report/final_report.md` for full references, exact metrics, definitions and limitations.
