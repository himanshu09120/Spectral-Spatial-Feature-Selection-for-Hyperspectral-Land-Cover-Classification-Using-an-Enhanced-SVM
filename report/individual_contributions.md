# Individual contributions appendix

The team requests an equal (50% / 50%) contribution allocation. The responsibilities below record that allocation; they do not claim separate commit authorship or independently logged execution by each member. Both members share responsibility for reviewing and explaining the complete submission.

| Team member | Roll number | Share | Allocated responsibilities |
|---|---|---|---|
| Himanshu Sinha | S20240020290 | 50% | Problem formulation and related work; Fisher ranking and correlation pruning; spectral/spatial SVM methodology; baseline and ablation interpretation; methodology and discussion sections of the report; joint presentation and final review. |
| Ghule Yash Mahesh | S20240020288 | 50% | Dataset provenance and validation; train/validation/test protocol and leakage checks; PCA/MNF and Random Forest comparisons; metric auditing, plots and reproducibility; experimental-results and implementation documentation; joint presentation and final review. |

Shared evidence: the implementation modules in `src/`, the regression suite in `tests/`, the experiment manifest, per-seed configuration audits, independent `audit.json`, and the implementation log. These artifacts establish what the project executed; they do not establish individual human authorship.

AI assistance disclosure: Codex assisted with repository review, implementation, automated verification, experiment execution, and drafting the report. Claude Code (Sonnet 5) assisted with resuming and completing the five-seed training run, diagnosing and repairing a concurrent-process artifact from an interrupted run, independent auditing, and generating the final measured report and PDF. Team members should review the code, measured results, interpretation and assigned contribution statements before submission and follow the course's disclosure rules.
