"""Independently verify every research result against saved held-out predictions."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_scene
from src.evaluation import classification_metrics, write_json
from src.experiment import split_digest
from src.main import experiment_fingerprint, file_hash, versions
from src.preprocessing import assert_disjoint, make_split


def audit(output=ROOT / "outputs"):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert not manifest["synthetic_only"], "Synthetic results cannot become research results"
    config = manifest["config"]
    cube, labels, _ = load_scene(config["dataset"], ROOT)
    hashes = {k: file_hash(ROOT / config["dataset"][k]) for k in ("cube_path", "labels_path")}
    assert hashes == manifest["dataset_files_sha256"]
    assert experiment_fingerprint(config, hashes, versions()) == manifest["fingerprint"]
    checked, candidates = [], 0
    methods = [f"E{i}" for i in range(7)]
    if config["svm"]["linear"]:
        methods.append("E1_linear")
    for protocol in config["protocols"]:
        for seed in config["seeds"]:
            folder = output / "metrics" / protocol / f"seed_{seed}"
            with np.load(folder / "split_indices.npz") as saved:
                split = {k: saved[k] for k in saved.files}
            expected = make_split(labels, config, seed, protocol)
            for key in split:
                np.testing.assert_array_equal(split[key], expected[key])
            assert_disjoint(split["train_pool"], split["validation"], split["test"])
            assert np.isin(split["train"], split["train_pool"]).all()
            for method in methods:
                result = json.loads((folder / f"{method}.json").read_text())
                assert result["fingerprint"] == manifest["fingerprint"]
                assert result["split_digest"] == split_digest(split)
                with np.load(output / "predictions" / protocol / f"seed_{seed}" / f"{method}_test.npz") as prediction:
                    np.testing.assert_array_equal(prediction["indices"], split["test"])
                    measured = classification_metrics(labels.ravel()[prediction["indices"]], prediction["prediction"])
                for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
                    np.testing.assert_allclose(measured[metric], result[metric], rtol=0, atol=1e-14)
                assert measured["confusion_matrix"] == result["confusion_matrix"]
                assert measured["per_class"] == result["per_class"]
                selection = json.loads((folder / f"selection_{method}.json").read_text())
                winner = max(selection, key=lambda row: row["validation_macro_f1"])
                assert winner == result["selected_configuration"]
                candidates += len(selection)
                checked.append(f"{protocol}/{seed}/{method}")
    result = {"status": "PASS", "fingerprint": manifest["fingerprint"],
              "completed_method_seed_results": len(checked), "candidate_fits": candidates,
              "checks": ["data and source fingerprint", "deterministic split reconstruction",
                         "disjoint partitions", "validation-only winning configuration",
                         "all test metrics independently recomputed", "per-class metrics and confusion counts"],
              "verified_results": checked}
    write_json(output / "audit.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    audit()
