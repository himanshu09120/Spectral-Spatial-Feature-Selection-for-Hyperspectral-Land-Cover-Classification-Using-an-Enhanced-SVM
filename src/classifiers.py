"""Bounded CPU classifiers. Construction never starts fitting."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def make_classifier(method, parameters, config, seed):
    if method == "E0":
        return DummyClassifier(strategy="most_frequent")
    if method == "E6":
        return RandomForestClassifier(**parameters, random_state=seed, n_jobs=config["execution"]["threads"],
                                      max_depth=config["random_forest"]["max_depth"],
                                      min_samples_leaf=config["random_forest"]["min_samples_leaf"])
    return SVC(**parameters, kernel="linear" if method == "E1_linear" else "rbf",
               cache_size=config["svm"]["cache_size_mb"], probability=False,
               random_state=seed)
