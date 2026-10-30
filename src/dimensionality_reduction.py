"""Noise-adjusted PCA (MNF) using a genuine, regularized noise covariance."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


class MNF:
    """Descending total/noise variance ordering, i.e. highest signal quality first.

    Solve C_x v = lambda C_n v; this is equivalent to noise whitening and PCA.
    Noise differences must be expressed in the same (training-standardized) units
    as X. The regularized noise metric gives V.T @ C_n @ V = I.
    """
    def __init__(self, regularization=1e-3):
        self.regularization = regularization

    def fit(self, X, noise_differences):
        X = np.asarray(X, dtype=np.float64)
        noise_differences = np.asarray(noise_differences, dtype=np.float64)
        if X.ndim != 2 or noise_differences.ndim != 2 or X.shape[1] != noise_differences.shape[1]:
            raise ValueError("MNF data and noise dimensions must match")
        if len(noise_differences) < 2 or self.regularization <= 0:
            raise ValueError("MNF needs >=2 noise pairs and positive regularization")
        self.mean_ = X.mean(axis=0)
        raw_noise = np.atleast_2d(np.cov(noise_differences, rowvar=False))
        scale = max(float(np.trace(raw_noise) / X.shape[1]), np.finfo(float).eps)
        self.noise_covariance_ = raw_noise + self.regularization * scale * np.eye(X.shape[1])
        covariance = np.atleast_2d(np.cov(X, rowvar=False))
        values, vectors = eigh(covariance, self.noise_covariance_, check_finite=True)
        order = np.argsort(values)[::-1]
        self.eigenvalues_, self.components_ = values[order], vectors[:, order]
        return self

    def transform(self, X, n_components=None):
        n = self.components_.shape[1] if n_components is None else n_components
        if not 1 <= n <= self.components_.shape[1]:
            raise ValueError("Invalid MNF component count")
        return ((X - self.mean_) @ self.components_[:, :n]).astype(np.float32)
