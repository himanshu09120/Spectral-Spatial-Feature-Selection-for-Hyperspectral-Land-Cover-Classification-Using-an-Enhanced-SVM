"""Shared experimental logic. Imported safely; fitting happens only in run()."""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import logging
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .classifiers import make_classifier
from .data_loader import CLASS_IDS
from .dimensionality_reduction import MNF
from .evaluation import aggregate_results, classification_metrics, write_json
from .feature_selection import FisherSelector
from .preprocessing import make_split, training_noise_pairs, training_scaler
from .spatial_features import spatial_features
from .visualization import (confusion_figures, dataset_figures, fisher_figure,
                            prediction_figures, result_figures)

LOG = logging.getLogger(__name__)


@dataclass
class Representation:
    kind: str
    count: int
    bands: object = None
    scaler: object = None
    reducer: object = None
    component_scaler: object = None
    window: int = 3

    def raw(self, cube):
        flat = cube.reshape(-1, cube.shape[2])
        if self.kind == "spatial":
            return spatial_features(cube, self.bands, self.window)
        if self.kind == "selected":
            return flat[:, self.bands]
        if self.kind == "dummy":
            return flat[:, :1]
        return flat

    def transform(self, raw, indices):
        values = raw[indices]
        if self.scaler is not None:
            values = self.scaler.transform(values)
        if self.reducer is not None:
            if isinstance(self.reducer, MNF):
                values = self.reducer.transform(values, self.count)
            else:
                values = self.reducer.transform(values)[:, :self.count]
        if self.component_scaler is not None:
            values = self.component_scaler.transform(values)
        return np.asarray(values, dtype=np.float32)


class FeatureFactory:
    """Fit transformations on the shared capped training set and reuse decompositions."""
    def __init__(self, cube, labels, split, config, seed):
        self.cube, self.labels, self.split, self.config, self.seed = cube, labels, split, config, seed
        self.flat = cube.reshape(-1, cube.shape[2])
        self.train = split["train"]
        self.selector = None
        self.base_scaler = None
        self.pca = None
        self.mnf = None
        self.noise_pairs = None
        self.spatial_cache = None
        self.spatial_key = None

    def select(self):
        if self.selector is None:
            self.selector = FisherSelector(self.config["correlation_threshold"]).fit(
                self.flat[self.train], self.labels.ravel()[self.train])
        return self.selector

    def counts(self, method):
        if method in ("E0", "E1", "E1_linear"):
            return [1 if method == "E0" else self.cube.shape[2]]
        if method in ("E2", "E3"):
            counts = [n for n in self.config["component_counts"] if n <= min(len(self.train), self.flat.shape[1])]
        else:
            available = len(self.select().selected_)
            counts = [n for n in self.config["band_counts"] if n <= available]
            if available and not counts:
                # Predeclared fallback, independent of validation/test performance.
                counts = [available]
            LOG.info("%s: strict correlation pruning retained %d bands; candidates=%s", method, available, counts)
        if not counts:
            raise ValueError(f"No feasible feature counts for {method}")
        return sorted(set(counts))

    def raw(self, representation):
        if representation.kind != "spatial":
            return representation.raw(self.cube)
        key = tuple(representation.bands)
        if key != self.spatial_key:
            self.spatial_cache = None
            self.spatial_cache = representation.raw(self.cube)
            self.spatial_key = key
        return self.spatial_cache

    def build(self, method, count):
        rep = Representation("full", count, window=self.config["spatial_window"])
        if method == "E0":
            return Representation("dummy", 1)
        if method in ("E4", "E5", "E6"):
            rep.bands = self.select().bands(count)
            rep.kind = "selected" if method == "E4" else "spatial"
            rep.count = count if method == "E4" else 3 * count
            rep.scaler = training_scaler(self.raw(rep), self.train)
            return rep
        if self.base_scaler is None:
            self.base_scaler = training_scaler(self.flat, self.train)
        rep.scaler = self.base_scaler
        if method in ("E2", "E3"):
            standardized = rep.scaler.transform(self.flat[self.train])
            if method == "E2":
                if self.pca is None:
                    self.pca = PCA(n_components=max(self.counts(method)), svd_solver="full").fit(standardized)
                rep.reducer = self.pca
            else:
                if self.mnf is None:
                    settings = self.config["mnf"]
                    self.noise_pairs = training_noise_pairs(self.cube, self.train,
                        max_pairs=settings["max_noise_pairs"], min_pairs=settings["min_noise_pairs"], seed=self.seed)
                    a, b = self.noise_pairs.T
                    differences = (self.flat[a].astype(np.float64) - self.flat[b]) / (np.sqrt(2) * rep.scaler.scale_)
                    self.mnf = MNF(settings["regularization"]).fit(standardized, differences)
                rep.reducer = self.mnf
            transformed = rep.transform(self.flat, self.train)
            rep.component_scaler = StandardScaler().fit(transformed)
        return rep


def hyperparameters(method, config):
    if method == "E0":
        return [{}]
    if method == "E6":
        return [{"n_estimators": n} for n in config["random_forest"]["n_estimators"]]
    if method == "E1_linear":
        return [{"C": c} for c in config["svm"]["C"]]
    return [{"C": c, "gamma": gamma} for c, gamma in itertools.product(config["svm"]["C"], config["svm"]["gamma"])]


def batched_predict(model, rep, raw, indices, batch_size, workers=1):
    """Predict independent batches with bounded CPU workers and stable row order."""
    result = np.empty(len(indices), dtype=np.int16)
    def batch(start):
        stop = start + batch_size
        return start, model.predict(rep.transform(raw, indices[start:stop]))

    starts = range(0, len(indices), batch_size)
    if workers == 1:
        for start in starts:
            offset, values = batch(start)
            result[offset:offset + len(values)] = values
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for offset, values in executor.map(batch, starts):
                result[offset:offset + len(values)] = values
    return result


def split_digest(split):
    digest = hashlib.sha256()
    for key in ("train", "train_pool", "validation", "test"):
        digest.update(np.asarray(split[key], dtype="<i8").tobytes())
    return digest.hexdigest()


def execute_method(method, factory, output, protocol, seed, fingerprint):
    config, split, labels = factory.config, factory.split, factory.labels.ravel()
    directory = output / "metrics" / protocol / f"seed_{seed}"
    path = directory / f"{method}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if not config["execution"]["resume"]:
            raise FileExistsError(f"{path} exists; use a new output_dir or enable resume")
        if saved["fingerprint"] != fingerprint or saved["split_digest"] != split_digest(split):
            raise ValueError("Refusing incompatible cached results")
        LOG.info("Resume: %s/%s/%s already complete", protocol, seed, method)
        return
    search_start, preprocessing_seconds, best_score = time.perf_counter(), 0.0, -1.0
    audit, best = [], None
    start = time.perf_counter()
    counts = factory.counts(method)
    preprocessing_seconds += time.perf_counter() - start
    for count in counts:
        start = time.perf_counter()
        rep = factory.build(method, count)
        raw = factory.raw(rep)
        train = split["train_pool"] if method == "E0" else split["train"]
        Xtrain = rep.transform(raw, train)
        preprocessing_seconds += time.perf_counter() - start
        for params in hyperparameters(method, config):
            model = make_classifier(method, params, config, seed)
            start = time.perf_counter()
            model.fit(Xtrain, labels[train])
            fit_seconds = time.perf_counter() - start
            predicted = batched_predict(model, rep, raw, split["validation"], config["execution"]["prediction_batch_size"],
                                        config["execution"].get("prediction_workers", 1))
            score = float(f1_score(labels[split["validation"]], predicted, labels=CLASS_IDS, average="macro", zero_division=0))
            candidate = dict(protocol=protocol, seed=seed, method=method, requested_count=count,
                             feature_count=rep.count, parameters=params, validation_macro_f1=score, fit_seconds=fit_seconds)
            audit.append(candidate)
            # Candidates are checkpointed after every fit; completed methods are resumable.
            write_json(directory / f"selection_{method}.json", audit)
            LOG.info("%s seed=%s %s features=%d %s validation F1=%.4f", protocol, seed, method, rep.count, params, score)
            # Deterministic tie handling: sorted smaller counts and input parameter order win.
            if score > best_score:
                best_score, best = score, (model, rep, candidate)
    search_seconds = time.perf_counter() - search_start
    model, rep, selected = best
    raw = factory.raw(rep)
    # Test measurements and labels are accessed for scoring only after selection finishes.
    start = time.perf_counter()
    prediction = batched_predict(model, rep, raw, split["test"], config["execution"]["prediction_batch_size"],
                                 config["execution"].get("prediction_workers", 1))
    prediction_seconds = time.perf_counter() - start
    result = classification_metrics(labels[split["test"]], prediction)
    result.update(protocol=protocol, seed=seed, method=method, fingerprint=fingerprint,
                  split_digest=split_digest(split), feature_count=rep.count,
                  preprocessing_seconds=preprocessing_seconds, search_seconds=search_seconds,
                  fit_seconds=selected["fit_seconds"], prediction_seconds=prediction_seconds,
                  validation_macro_f1=best_score, selected_configuration=selected,
                  selected_bands_zero_based=rep.bands.tolist() if rep.bands is not None else None,
                  training_counts={str(int(c)): int(np.sum(labels[train] == c)) for c in CLASS_IDS},
                  majority_uses_uncapped_training_priors=(method == "E0"))
    figure_dir = output / "figures" / protocol / f"seed_{seed}"
    confusion_figures(result["confusion_matrix"], figure_dir, method, config["execution"]["figure_dpi"])
    prediction_dir = output / "predictions" / protocol / f"seed_{seed}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_dir / f"{method}_test.npz", indices=split["test"], prediction=prediction)
    if rep.bands is not None:
        selector = factory.select()
        write_json(directory / f"bands_{method}.json", {"selected_zero_based": rep.bands.tolist(),
                   "selected_one_based": (rep.bands + 1).tolist(), "fisher_scores": selector.scores_.tolist(),
                   "ranking_zero_based": selector.ranking_.tolist(), "pruned_order_zero_based": selector.selected_.tolist()})
        fisher_figure(selector, rep.bands, figure_dir / f"{method}_fisher.png", config["execution"]["figure_dpi"])
    if method == "E3":
        np.savez_compressed(directory / "mnf_diagnostics.npz", noise_pairs=factory.noise_pairs,
                            noise_covariance=factory.mnf.noise_covariance_, eigenvalues=factory.mnf.eigenvalues_,
                            components=factory.mnf.components_)
    if config["execution"]["save_models"]:
        model_path = output / "models" / protocol / f"seed_{seed}"
        model_path.mkdir(parents=True, exist_ok=True)
        temporary = model_path / f"{method}.joblib.tmp"
        joblib.dump({"classifier": model, "representation": rep, "fingerprint": fingerprint}, temporary)
        temporary.replace(model_path / f"{method}.joblib")
    if seed in config["execution"]["map_seeds"] and method in ("E1", "E5"):
        indices = np.flatnonzero(labels > 0)
        land_cover = np.zeros_like(labels)
        land_cover[indices] = batched_predict(model, rep, raw, indices, config["execution"]["prediction_batch_size"],
                                             config["execution"].get("prediction_workers", 1))
        land_cover = land_cover.reshape(factory.labels.shape)
        np.save(prediction_dir / f"{method}_map.npy", land_cover)
        prediction_figures(factory.labels, land_cover, split["test"], figure_dir, method, config["execution"]["figure_dpi"])
    # Completion marker is written last, so partial artifacts are never counted as results.
    write_json(path, result)


def run(cube, labels, config, output, fingerprint):
    if not config["execution"]["allow_training"]:
        raise PermissionError("Training is disabled. Explicitly enable it when you intend to run experiments.")
    methods = [f"E{i}" for i in range(7)]
    if config["svm"]["linear"]:
        methods.insert(2, "E1_linear")
    with threadpool_limits(limits=config["execution"]["threads"]):
        dataset_figures(cube, labels, output / "figures" / "dataset", config["execution"]["figure_dpi"])
        for protocol in config["protocols"]:
            for seed in config["seeds"]:
                split = make_split(labels, config, seed, protocol)
                folder = output / "metrics" / protocol / f"seed_{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "split_indices.npz", **split)
                write_json(folder / "split_counts.json", {
                    name: {str(int(c)): int(np.sum(labels.ravel()[indices] == c)) for c in CLASS_IDS}
                    for name, indices in split.items()})
                factory = FeatureFactory(cube, labels, split, config, seed)
                for method in methods:
                    try:
                        execute_method(method, factory, output, protocol, seed, fingerprint)
                    except Exception:
                        LOG.exception("Failed %s seed %s method %s; completed methods remain saved", protocol, seed, method)
                        aggregate_results(output, config["seeds"])
                        raise
                    aggregate_results(output, config["seeds"])
        raw, summary = aggregate_results(output, config["seeds"])
        result_figures(output, summary, config["execution"]["figure_dpi"])
        LOG.info("Final test results:\n%s", summary[["protocol", "method", "accuracy_mean", "macro_f1_mean", "n_seeds", "complete"]].to_string(index=False))
        return raw, summary
