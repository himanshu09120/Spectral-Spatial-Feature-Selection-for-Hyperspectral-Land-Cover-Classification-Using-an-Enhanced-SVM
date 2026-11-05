"""Check actual fitted classifiers produce identical serial/parallel predictions."""
import unittest

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from src.experiment import Representation, batched_predict


class PredictionTests(unittest.TestCase):
    def test_parallel_predictions_preserve_values_and_order(self):
        rng = np.random.default_rng(381)
        features = rng.normal(size=(93, 7)).astype(np.float32)
        labels = np.tile(np.arange(1, 4), 31)
        order = rng.permutation(len(labels))
        rep = Representation("full", 7)
        for model in [SVC(), RandomForestClassifier(n_estimators=5, n_jobs=1, random_state=7),
                      DummyClassifier(strategy="most_frequent")]:
            model.fit(features[:60], labels[:60])
            expected = model.predict(features[order])
            for workers in [1, 4]:
                actual = batched_predict(model, rep, features, order, 11, workers)
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
