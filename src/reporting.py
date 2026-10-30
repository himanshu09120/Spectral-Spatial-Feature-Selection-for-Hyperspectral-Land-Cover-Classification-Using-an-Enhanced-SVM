"""Render measured aggregates without generating conclusions or invented values."""


def write_results(output, summary, synthetic=False):
    lines = ["# " + ("SYNTHETIC SOFTWARE TEST — NOT RESEARCH RESULTS" if synthetic else "Measured experiment results"),
             "", "Generated exclusively from saved per-seed measurements. SD uses ddof=1; one-run SD is undefined.",
             "", "| Protocol | Method | Seeds | Complete | OA mean ± SD | Macro-F1 mean ± SD | Features mean | Fit s mean |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.protocol} | {row.method} | {row.n_seeds} | {row.complete} | "
                     f"{row.accuracy_mean:.4f} ± {row.accuracy_std:.4f} | "
                     f"{row.macro_f1_mean:.4f} ± {row.macro_f1_std:.4f} | "
                     f"{row.feature_count_mean:.1f} | {row.fit_seconds_mean:.3f} |")
    lines += ["", "Use metrics/per_class_summary.csv for per-class comparisons and metrics/ablation_deltas.csv for paired ablations.",
              "Descriptive full-scene maps include training pixels. Only held-out test predictions determine the reported metrics.",
              "Inspect the saved confusion matrices before writing the analysis. No superiority claim is made automatically.", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
