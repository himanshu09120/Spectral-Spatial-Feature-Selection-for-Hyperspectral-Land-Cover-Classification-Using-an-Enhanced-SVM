"""Multiclass Fisher ranking followed by strict correlation pruning."""
from __future__ import annotations

import numpy as np


def fisher_scores(X, y, epsilon=1e-12):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or len(y) != len(X) or len(np.unique(y)) < 2:
        raise ValueError("Fisher scores require a feature matrix and at least two labelled classes")
    mean = X.mean(axis=0)
    between, within = np.zeros(X.shape[1]), np.zeros(X.shape[1])
    for label in np.unique(y):
        group = X[y == label]
        between += len(group) * (group.mean(axis=0) - mean) ** 2
        within += len(group) * group.var(axis=0, ddof=0)
    return between / (within + epsilon)


class FisherSelector:
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y):
        if not 0 < self.threshold <= 1:
            raise ValueError("Correlation threshold must be in (0, 1]")
        self.scores_ = fisher_scores(X, y)
        self.ranking_ = np.argsort(-self.scores_, kind="stable")
        # Normalize explicitly so constant bands never produce NaN correlations.
        centered = np.asarray(X, dtype=np.float64) - np.mean(X, axis=0)
        norms = np.linalg.norm(centered, axis=0)
        normalized = centered / np.where(norms > 0, norms, 1)
        correlation = np.clip(normalized.T @ normalized, -1, 1)
        selected = []
        for j in self.ranking_:
            if norms[j] > 0 and (not selected or np.all(np.abs(correlation[j, selected]) <= self.threshold)):
                selected.append(int(j))
        self.selected_ = np.asarray(selected, dtype=int)
        return self

    def bands(self, count):
        if count <= 0 or count > len(self.selected_):
            raise ValueError(f"Requested {count} bands, but strict pruning retained only {len(self.selected_)}")
        return self.selected_[:count]
