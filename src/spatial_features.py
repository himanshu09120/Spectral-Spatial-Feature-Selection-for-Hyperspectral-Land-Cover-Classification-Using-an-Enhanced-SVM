"""Measurement-only spatial statistics with reflected image borders."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def spatial_features(cube, bands, window=3):
    if window < 1 or window % 2 != 1:
        raise ValueError("Spatial window must be a positive odd integer")
    bands = np.asarray(bands, dtype=int)
    if bands.ndim != 1 or not len(bands) or np.any((bands < 0) | (bands >= cube.shape[2])):
        raise ValueError("Invalid selected bands")
    h, w = cube.shape[:2]
    result = np.empty((h, w, 3 * len(bands)), dtype=np.float32)
    # Per-band float64 accumulation avoids cancellation in nearly constant regions.
    for position, band in enumerate(bands):
        values = cube[:, :, band].astype(np.float64)
        mean = uniform_filter(values, size=window, mode="reflect")
        second = uniform_filter(values * values, size=window, mode="reflect")
        result[:, :, position] = values
        result[:, :, len(bands) + position] = mean
        result[:, :, 2 * len(bands) + position] = np.sqrt(np.maximum(second - mean * mean, 0))
    return result.reshape(h * w, -1)
