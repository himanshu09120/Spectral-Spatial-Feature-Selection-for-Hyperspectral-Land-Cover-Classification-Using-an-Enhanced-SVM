"""CLI entry point. Default mode is a dependency-free plan, with no training."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    __package__ = "src"


def validate_config(config):
    fractions = config["split"]
    if set(fractions) != {"train", "validation", "test"} or any(not 0 < x < 1 for x in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-8:
        raise ValueError("Split must contain positive train/validation/test fractions summing to 1")
    for key in ("band_counts", "component_counts"):
        if not config[key] or any(type(n) is not int or n <= 0 for n in config[key]):
            raise ValueError(f"{key} must contain positive integers")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Use at least one seed and no duplicate seeds")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in config["seeds"]):
        raise ValueError("Seeds must be integers in [0, 2**32)")
    if config["train_cap_per_class"] < 1 or config["execution"]["threads"] < 1 or config["execution"]["prediction_batch_size"] < 1:
        raise ValueError("Training cap, threads and prediction batch size must be positive")
    if not 0 < config["correlation_threshold"] <= 1:
        raise ValueError("Correlation threshold must lie in (0, 1]")
    if config["spatial_window"] < 1 or config["spatial_window"] % 2 != 1:
        raise ValueError("Spatial window must be positive and odd")
    if not config["protocols"] or not set(config["protocols"]) <= {"pixel", "spatial_block"}:
        raise ValueError("Supported protocols: pixel, spatial_block")
    if not config["svm"]["C"] or any(x <= 0 for x in config["svm"]["C"]):
        raise ValueError("SVM C candidates must be positive")
    if not config["svm"]["gamma"] or any(g not in ("scale", "auto") and (not isinstance(g, (int, float)) or g <= 0) for g in config["svm"]["gamma"]):
        raise ValueError("SVM gamma candidates must be scale, auto or positive numbers")
    if not config["random_forest"]["n_estimators"] or any(n < 1 for n in config["random_forest"]["n_estimators"]):
        raise ValueError("Forest tree counts must be positive")


def plan(config):
    svm_grid = len(config["svm"]["C"]) * len(config["svm"]["gamma"])
    bands, components = len(config["band_counts"]), len(config["component_counts"])
    per_seed = 1 + svm_grid * (1 + 2 * components + 2 * bands) + bands * len(config["random_forest"]["n_estimators"])
    if config["svm"]["linear"]:
        per_seed += len(config["svm"]["C"])
    return {"mode": "plan_only_no_training", "training_enabled_in_config": config["execution"]["allow_training"],
            "methods": ["E0 majority", "E1 full-spectrum SVM", "E2 PCA-SVM", "E3 MNF-SVM", "E4 selected-band SVM", "E5 FRSS-SVM", "E6 spatial Random Forest"],
            "seeds": config["seeds"], "protocols": config["protocols"],
            "maximum_candidate_fits_per_seed_protocol": per_seed,
            "maximum_total_candidate_fits": per_seed * len(config["seeds"]) * len(config["protocols"]),
            "train_cap_per_class": config["train_cap_per_class"], "threads": config["execution"]["threads"],
            "note": "No classifiers or transforms are fitted by this command. Fits are skipped for infeasible feature counts."}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions():
    return {"python": sys.version, "platform": platform.platform(), "executable": sys.executable,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "scikit-learn", "matplotlib", "pandas", "h5py", "joblib", "threadpoolctl")}}


def experiment_fingerprint(config, dataset_files, environment):
    compatible_config = copy.deepcopy(config)
    for key in ("allow_training", "resume"):
        compatible_config["execution"].pop(key, None)
    content = {"config": compatible_config, "dataset_files": dataset_files,
               "python_version": platform.python_version(), "packages": environment["packages"],
               "source": {p.name: file_hash(p) for p in sorted((ROOT / "src").glob("*.py"))}}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".run.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Run lock exists: {path}. Check that its PID is inactive before removing a stale lock.") from error
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def synthetic_scene():
    import numpy as np
    rng = np.random.default_rng(999)
    labels = np.tile(np.arange(1, 10, dtype=np.int16), (36, 4)).reshape(36, 36)
    cube = rng.normal(0, .1, (36, 36, 12)).astype(np.float32)
    cube += rng.normal(size=(10, 12)).astype(np.float32)[labels]
    return cube, labels


def smoke_test():
    """Structural synthetic smoke check; never fits a classifier or transformation."""
    import numpy as np
    from .data_loader import validate_scene
    from .evaluation import classification_metrics
    from .spatial_features import spatial_features
    cube, labels = synthetic_scene()
    summary = validate_scene(cube, labels)
    features = spatial_features(cube, [0, 2, 4])
    assert features.shape == (labels.size, 9)
    assert np.isfinite(features).all()
    # Hand-constructed labels verify metrics; these are not model predictions.
    assert classification_metrics(np.arange(1, 10), np.arange(1, 10))["accuracy"] == 1.0
    print(json.dumps({"status": "PASS", "synthetic_only": True, "classifiers_fitted": 0,
                      "checks": ["scene validation", "spatial dimensions", "finite features", "metric arithmetic"],
                      "shape": summary["cube_shape"]}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "config.json")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Print plan only; default")
    modes.add_argument("--validate-data", action="store_true", help="Load and validate real files; no fitting")
    modes.add_argument("--smoke-test", action="store_true", help="Synthetic structural checks; no training")
    modes.add_argument("--run", action="store_true", help="Run experiments; training must also be explicitly enabled")
    modes.add_argument("--smoke-train", action="store_true", help="Optional synthetic E0–E6 integration run; requires --allow-training")
    parser.add_argument("--allow-training", action="store_true", help="Explicitly permit model fitting for this invocation")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    if not any((args.validate_data, args.smoke_test, args.run, args.smoke_train)):
        print(json.dumps(plan(config), indent=2))
        return 0
    if args.smoke_test:
        smoke_test()
        return 0
    if args.run or args.smoke_train:
        if args.smoke_train and not args.allow_training:
            parser.error("--smoke-train requires --allow-training")
        if not (args.allow_training or config["execution"]["allow_training"]):
            parser.error("Training is disabled by your code-only setting. Use --allow-training only when ready to train.")
        config["execution"]["allow_training"] = True
    from .data_loader import load_scene, validate_scene
    from .evaluation import write_json
    if args.smoke_train:
        cube, labels = synthetic_scene()
        config.update(seeds=[999], band_counts=[3, 5], component_counts=[3, 5],
                      train_cap_per_class=24, protocols=["pixel"], correlation_threshold=.999)
        config["svm"].update(C=[1], gamma=["scale"])
        config["random_forest"]["n_estimators"] = [5]
        config["mnf"]["min_noise_pairs"] = 2
        config["execution"].update(map_seeds=[999], figure_dpi=90)
        config["output_dir"] = "outputs/synthetic_smoke_NOT_RESEARCH"
        summary = validate_scene(cube, labels)
        dataset_files = {"synthetic": "fixed generator seed 999; NOT LongKou"}
    else:
        cube, labels, summary = load_scene(config["dataset"], ROOT)
        dataset_files = {key: file_hash(ROOT / config["dataset"][key]) for key in ("cube_path", "labels_path")}
    output = ROOT / config["output_dir"]
    if args.validate_data:
        write_json(output / "data_validation.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    environment = versions()
    fingerprint = experiment_fingerprint(config, dataset_files, environment)
    with run_lock(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Output directory contains an incompatible run. Set a new output_dir; never mix results.")
        if not manifest_path.exists() and list((output / "metrics").glob("*/seed_*/E*.json")):
            raise ValueError("Results exist without a provenance manifest; use a fresh output_dir")
        write_json(manifest_path, {"fingerprint": fingerprint, "config": config, "environment": environment,
                                  "dataset_files_sha256": dataset_files, "synthetic_only": args.smoke_train})
        write_json(output / "data_validation.json", summary)
        logs = output / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            handlers=[logging.StreamHandler(), logging.FileHandler(logs / "run.log")])
        from .experiment import run
        from .reporting import write_results
        _, aggregate = run(cube, labels, config, output, fingerprint)
        write_results(output, aggregate, args.smoke_train)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as error:
        print(f"Missing/incompatible dependency: {error}. Install requirements.txt in .venv.", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        logging.exception("Project failed: %s", error)
        sys.exit(1)
