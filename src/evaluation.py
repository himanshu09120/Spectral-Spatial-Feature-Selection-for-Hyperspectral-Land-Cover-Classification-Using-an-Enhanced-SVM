"""Fixed nine-class metrics and seed-aligned aggregation; zero division = 0."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .data_loader import CLASS_IDS, CLASS_NAMES


METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "feature_count",
           "preprocessing_seconds", "search_seconds", "fit_seconds", "prediction_seconds"]


def classification_metrics(truth, prediction):
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=CLASS_IDS, zero_division=0)
    matrix = confusion_matrix(truth, prediction, labels=CLASS_IDS)
    return {"accuracy": float(accuracy_score(truth, prediction)),
            "macro_precision": float(np.mean(precision)), "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)), "confusion_matrix": matrix.tolist(),
            "per_class": [{"class_id": int(c), "class_name": name, "precision": float(p),
                           "recall": float(r), "f1": float(f), "support": int(n)}
                          for c, name, p, r, f, n in zip(CLASS_IDS, CLASS_NAMES, precision, recall, f1, support)]}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def aggregate_results(output, expected_seeds):
    records = [json.loads(p.read_text()) for p in sorted((output / "metrics").glob("*/seed_*/E*.json"))]
    if not records:
        return pd.DataFrame(), pd.DataFrame()
    rows, classes = [], []
    for record in records:
        row = {k: record[k] for k in ["protocol", "seed", "method", "split_digest"] + METRICS}
        rows.append(row)
        classes.extend([dict(protocol=row["protocol"], seed=row["seed"], method=row["method"], **c)
                        for c in record["per_class"]])
    raw = pd.DataFrame(rows)
    if raw.duplicated(["protocol", "seed", "method"]).any():
        raise ValueError("Duplicate per-seed results")
    for _, group in raw.groupby(["protocol", "seed"]):
        if group.split_digest.nunique() != 1:
            raise ValueError("Methods evaluated on incompatible test splits")
    raw.to_csv(output / "metrics" / "per_seed.csv", index=False)
    per_class = pd.DataFrame(classes)
    per_class.to_csv(output / "metrics" / "per_class.csv", index=False)
    class_summary = per_class.groupby(["protocol", "method", "class_id", "class_name"])[["precision", "recall", "f1"]].agg(["mean", "std"])
    class_summary.columns = ["_".join(c) for c in class_summary.columns]
    class_summary.to_csv(output / "metrics" / "per_class_summary.csv")
    grouped = raw.groupby(["protocol", "method"])
    summary = grouped[METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary["n_seeds"] = grouped.seed.nunique()
    summary["complete"] = grouped.seed.apply(lambda x: set(x) == set(expected_seeds))
    summary.reset_index().to_csv(output / "metrics" / "summary.csv", index=False)
    # Paired ablation deltas use the same seed and test indices, rather than unrelated means.
    deltas = []
    for (protocol, seed), group in raw.groupby(["protocol", "seed"]):
        indexed = group.set_index("method")
        for baseline, proposed in [("E1", "E4"), ("E4", "E5"), ("E1", "E5")]:
            if baseline in indexed.index and proposed in indexed.index:
                deltas.append(dict(protocol=protocol, seed=int(seed), comparison=f"{proposed}-{baseline}",
                                   **{f"delta_{m}": float(indexed.loc[proposed, m] - indexed.loc[baseline, m])
                                      for m in METRICS}))
    pd.DataFrame(deltas).to_csv(output / "metrics" / "ablation_deltas.csv", index=False)
    return raw, summary.reset_index()
