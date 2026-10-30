"""Exercise all experiment branches using a fixed-output, non-learning stub.

No real classifier is trained. Artifacts exist only in a temporary test directory.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.experiment import FeatureFactory, Representation, run
from src.main import ROOT, synthetic_scene
from src.preprocessing import make_split
from src.reporting import write_results


class NonLearningStub:
    """No parameters, fitting logic or state; always predicts class 1."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=np.int16)


class OrchestrationTests(unittest.TestCase):
    def config(self):
        config = json.loads((ROOT / "config" / "config.json").read_text())
        config.update(seeds=[42], band_counts=[2, 3], component_counts=[2, 3],
                      train_cap_per_class=24, correlation_threshold=.999)
        config["svm"].update(C=[1], gamma=["scale"])
        config["random_forest"]["n_estimators"] = [2]
        config["mnf"]["min_noise_pairs"] = 2
        config["execution"].update(allow_training=True, figure_dpi=35, map_seeds=[42])
        return config

    def test_all_branches_with_no_learning_stub_and_resume(self):
        cube, labels = synthetic_scene()
        config = self.config()
        with tempfile.TemporaryDirectory() as directory, \
                patch("src.experiment.make_classifier", return_value=NonLearningStub()) as factory, \
                patch("sklearn.svm.SVC.fit", side_effect=AssertionError("No SVM training permitted")), \
                patch("sklearn.ensemble.RandomForestClassifier.fit", side_effect=AssertionError("No RF training permitted")), \
                patch("sklearn.dummy.DummyClassifier.fit", side_effect=AssertionError("No dummy training permitted")):
            output = Path(directory)
            raw, summary = run(cube, labels, config, output, "synthetic-stub-test")
            self.assertEqual(set(raw.method), {f"E{i}" for i in range(7)})
            self.assertEqual(raw.split_digest.nunique(), 1)
            self.assertTrue(summary.complete.all())
            self.assertTrue(summary.accuracy_std.isna().all())
            self.assertTrue((output / "predictions/pixel/seed_42/E5_map.npy").exists())
            self.assertTrue((output / "metrics/pixel/seed_42/mnf_diagnostics.npz").exists())
            self.assertTrue((output / "figures/pixel/validation_feature_curves.png").exists())
            write_results(output, summary, synthetic=True)
            self.assertIn("NOT RESEARCH", (output / "results.md").read_text())
            calls = factory.call_count
            run(cube, labels, config, output, "synthetic-stub-test")
            self.assertEqual(factory.call_count, calls)
            with self.assertLogs("src.experiment", level="ERROR"), self.assertRaisesRegex(ValueError, "incompatible"):
                run(cube, labels, config, output, "changed-fingerprint")

    def test_pipeline_guard_and_transform_isolation(self):
        cube, labels = synthetic_scene()
        config = self.config()
        config["execution"]["allow_training"] = False
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PermissionError):
                run(cube, labels, config, Path(directory), "disabled")
        split = make_split(labels, config, 42)
        feature_factory = FeatureFactory(cube, labels, split, config, 42)
        rep = feature_factory.build("E3", 3)
        before = rep.transform(cube.reshape(-1, 12), split["train"])
        changed = cube.copy().reshape(-1, 12)
        changed[split["validation"]] += 10000
        changed[split["test"]] -= 10000
        second = FeatureFactory(changed.reshape(cube.shape), labels, split, config, 42)
        other = second.build("E3", 3)
        np.testing.assert_allclose(before, other.transform(changed, split["train"]))
        np.testing.assert_array_equal(rep.reducer.noise_covariance_, other.reducer.noise_covariance_)
        # Same isolation check for PCA and Fisher/correlation-selected band order.
        first_pca, second_pca = feature_factory.build("E2", 3), second.build("E2", 3)
        np.testing.assert_allclose(first_pca.reducer.components_, second_pca.reducer.components_)
        np.testing.assert_array_equal(feature_factory.select().selected_, second.select().selected_)


if __name__ == "__main__":
    unittest.main()
