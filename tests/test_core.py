"""Small deterministic numerical tests. No classifier training occurs here."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy.io import savemat

from src.data_loader import load_scene, read_array, validate_scene
from src.dimensionality_reduction import MNF
from src.evaluation import aggregate_results, classification_metrics, write_json
from src.feature_selection import FisherSelector, fisher_scores
from src.main import ROOT, experiment_fingerprint, main, plan, run_lock, validate_config
from src.preprocessing import assert_disjoint, make_split, training_noise_pairs, training_scaler
from src.spatial_features import spatial_features


class DataTests(unittest.TestCase):
    def test_standard_mat_and_background(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cube = np.arange(120, dtype=float).reshape(3, 4, 10)
            labels = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 1, 2]])
            savemat(root / "cube.mat", {"scene": cube.transpose(2, 0, 1)})
            savemat(root / "gt.mat", {"truth": labels})
            config = dict(cube_path="cube.mat", labels_path="gt.mat", expected_shape=[3, 4, 10])
            actual, truth, summary = load_scene(config, root)
            np.testing.assert_array_equal(actual, cube)
            self.assertEqual(summary["labelled_pixels"], 11)
            self.assertEqual(truth.dtype, np.int16)

    def test_hdf5_matlab_orientation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v73.mat"
            cube = np.zeros((3, 4, 10))
            with h5py.File(path, "w") as handle:
                ds = handle.create_dataset("cube", data=cube.T)
                ds.attrs["MATLAB_class"] = np.bytes_("double")
            result, _ = read_array(path, "cube", 3)
            self.assertEqual(result.shape, cube.shape)

    def test_validation_rejects_invalid_data(self):
        cube = np.zeros((3, 3, 10))
        labels = np.arange(1, 10).reshape(3, 3)
        validate_scene(cube, labels)
        for bad in [labels.astype(float) + .1, labels * 2, np.zeros_like(labels)]:
            with self.assertRaises(ValueError):
                validate_scene(cube, bad)
        cube[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            validate_scene(cube, labels)
        with self.assertRaises(FileNotFoundError):
            read_array(Path("missing_scene.mat"), None, 3)

    def test_ambiguous_keys_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "two.mat"
            savemat(path, {"a": np.zeros((2, 3)), "b": np.zeros((2, 3))})
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                read_array(path, None, 2)


class NumericalTests(unittest.TestCase):
    def test_blas_finite_arithmetic_without_float_errors(self):
        rng = np.random.default_rng(0)
        left, right = rng.normal(size=(216, 12)), rng.normal(size=(12, 3))
        with np.errstate(divide="raise", over="raise", invalid="raise"):
            actual = left @ right
        np.testing.assert_allclose(actual, np.einsum("ij,jk->ik", left, right), atol=1e-12)

    def test_fisher_formula(self):
        X = np.array([[0, 1], [2, 1], [4, 1], [6, 1]])
        # Global mean 3: between = 2*(1-3)^2 + 2*(5-3)^2 = 16; within=4.
        np.testing.assert_allclose(fisher_scores(X, np.array([1, 1, 2, 2])), [4, 0], atol=1e-10)

    def test_pruning_handles_duplicates_and_constants(self):
        x = np.arange(20, dtype=float)
        X = np.column_stack([x, x, -x, np.ones(20)])
        selector = FisherSelector(.95).fit(X, np.repeat([1, 2], 10))
        self.assertEqual(len(selector.selected_), 1)
        self.assertEqual(selector.bands(1).tolist(), [0])
        with self.assertRaises(ValueError):
            selector.bands(2)

    def test_training_only_scaling_and_selection(self):
        X = np.array([[0., 2], [1, 1], [3, 4], [4, 3], [999, -999]])
        train, labels = np.arange(4), np.array([1, 1, 2, 2])
        scaler = training_scaler(X, train)
        selector = FisherSelector().fit(X[train], labels)
        X[-1] *= -1e8
        np.testing.assert_array_equal(scaler.mean_, [2, 2.5])
        np.testing.assert_array_equal(selector.scores_, fisher_scores(X[train], labels))
        with self.assertRaisesRegex(ValueError, "Leakage"):
            assert_disjoint(np.array([1, 2]), np.array([2, 3]))

    def test_spatial_features_shape_and_values(self):
        cube = np.zeros((3, 3, 2), dtype=float)
        cube[:, :, 0] = np.arange(9).reshape(3, 3)
        cube[:, :, 1] = 2
        result = spatial_features(cube, [0, 1])
        self.assertEqual(result.shape, (9, 6))
        np.testing.assert_allclose(result[4], [4, 2, 4, 2, np.std(np.arange(9)), 0], atol=1e-6)
        # Reflected upper-left neighbourhood: [[0,0,1],[0,0,1],[3,3,4]].
        self.assertAlmostEqual(float(result[0, 2]), 12 / 9, places=6)
        with self.assertRaises(ValueError):
            spatial_features(cube, [0], 2)

    def test_mnf_noise_metric_and_ordering(self):
        rng = np.random.default_rng(10)
        X = rng.normal(size=(80, 4)) * [1, 2, 3, 4]
        noise = rng.normal(size=(60, 4)) * [.5, 2, 1, .1]
        mnf = MNF().fit(X, noise)
        np.testing.assert_allclose(mnf.components_.T @ mnf.noise_covariance_ @ mnf.components_, np.eye(4), atol=1e-9)
        projected = mnf.transform(X)
        np.testing.assert_allclose(np.diag(np.cov(projected, rowvar=False)), mnf.eigenvalues_, rtol=1e-6)
        self.assertTrue(np.all(np.diff(mnf.eigenvalues_) <= 0))
        self.assertEqual(mnf.transform(X, 2).shape, (80, 2))
        # Near-singular noise remains finite under explicit regularization.
        self.assertTrue(np.isfinite(MNF().fit(X, np.zeros_like(noise)).transform(X)).all())

    def test_noise_pairs_only_training_endpoints(self):
        cube = np.zeros((5, 5, 3))
        train = np.array([0, 1, 5, 6, 10, 11, 15, 16])
        pairs = training_noise_pairs(cube, train, min_pairs=2, max_pairs=4)
        self.assertTrue(np.isin(pairs, train).all())
        self.assertEqual(pairs.shape, (4, 2))
        self.assertTrue(np.isin(pairs[:, 1] - pairs[:, 0], [1, 5]).all())
        with self.assertRaisesRegex(ValueError, "neighbour pairs"):
            training_noise_pairs(cube, [0, 24], min_pairs=2)

    def test_metrics_use_nine_classes_and_zero_division(self):
        result = classification_metrics(np.arange(1, 10), np.ones(9))
        self.assertAlmostEqual(result["accuracy"], 1 / 9)
        self.assertAlmostEqual(result["macro_recall"], 1 / 9)
        self.assertEqual(result["per_class"][1]["precision"], 0)
        self.assertEqual(np.asarray(result["confusion_matrix"]).shape, (9, 9))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config" / "config.json").read_text())

    def test_splits_reproducible_disjoint_and_capped(self):
        labels = np.tile(np.arange(10), (100, 10))
        self.config["train_cap_per_class"] = 10
        split = make_split(labels, self.config, 42)
        again = make_split(labels, self.config, 42)
        for name in split:
            np.testing.assert_array_equal(split[name], again[name])
            self.assertTrue(np.all(labels.ravel()[split[name]] > 0))
        self.assertEqual(len(split["train"]), 90)
        assert_disjoint(split["train_pool"], split["validation"], split["test"])
        self.assertTrue(np.isin(split["train"], split["train_pool"]).all())
        for name in ("train_pool", "validation", "test"):
            self.assertAlmostEqual(len(split[name]) / np.count_nonzero(labels), self.config["split"].get(name, .7), places=2)

    def test_block_buffer_separates_neighbourhoods(self):
        from scipy.ndimage import binary_dilation
        labels = np.tile(np.arange(1, 10), (90, 10))
        self.config["spatial_block"] = {"block_size": 10, "buffer": 2}
        split = make_split(labels, self.config, 7, "spatial_block")
        footprints = []
        for name in ("train_pool", "validation", "test"):
            mask = np.zeros(labels.shape, dtype=bool)
            mask.ravel()[split[name]] = True
            footprints.append(binary_dilation(mask, structure=np.ones((3, 3))))
        self.assertFalse(np.any(footprints[0] & footprints[1]))
        self.assertFalse(np.any(footprints[0] & footprints[2]))
        self.assertFalse(np.any(footprints[1] & footprints[2]))

    def test_config_and_budget(self):
        validate_config(self.config)
        self.assertFalse(self.config["execution"]["allow_training"])
        self.assertEqual(plan(self.config)["maximum_total_candidate_fits"], 305)
        bad = copy.deepcopy(self.config)
        bad["split"]["test"] = .5
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_cli_refuses_training_before_importing_runner(self):
        from unittest.mock import patch
        with patch("src.experiment.run", side_effect=AssertionError("must not train")) as run:
            with self.assertRaises(SystemExit) as context:
                main(["--run"])
            self.assertEqual(context.exception.code, 2)
            run.assert_not_called()

    def test_lock_rejects_duplicate_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with run_lock(path):
                with self.assertRaises(RuntimeError):
                    with run_lock(path):
                        self.fail("Second lock acquired")
            self.assertFalse((path / ".run.lock").exists())

    def test_sample_standard_deviation_and_paired_ablation(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            for seed, accuracy in [(7, .4), (19, .8)]:
                for method in ("E1", "E4", "E5"):
                    record = classification_metrics(np.arange(1, 10), np.arange(1, 10))
                    record.update(protocol="pixel", seed=seed, method=method, split_digest=str(seed),
                                  accuracy=accuracy, feature_count=10, preprocessing_seconds=1,
                                  search_seconds=2, fit_seconds=1, prediction_seconds=.1)
                    write_json(output / "metrics" / "pixel" / f"seed_{seed}" / f"{method}.json", record)
            raw, summary = aggregate_results(output, [7, 19])
            self.assertEqual(len(raw), 6)
            self.assertTrue(summary.complete.all())
            np.testing.assert_allclose(summary.accuracy_std, np.std([.4, .8], ddof=1))
            self.assertTrue((output / "metrics" / "ablation_deltas.csv").is_file())

    def test_matlab_launcher_static_contract(self):
        text = (ROOT / "Main.m").read_text()
        for required in ("mfilename('fullpath')", "onCleanup", "FRSS_PYTHON", "status ~= 0", "shellQuote", "--config"):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
