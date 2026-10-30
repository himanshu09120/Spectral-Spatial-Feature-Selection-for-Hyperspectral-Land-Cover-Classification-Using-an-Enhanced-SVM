"""MATLAB scene loading, orientation checks and label validation."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy.io import loadmat, whosmat

CLASS_NAMES = ["Corn", "Cotton", "Sesame", "Broad-leaf soybean", "Narrow-leaf soybean",
               "Rice", "Water", "Roads and houses", "Mixed weed"]
CLASS_IDS = np.arange(1, 10)
DOCUMENTED_COUNTS = [34511, 8374, 3031, 63212, 4151, 11854, 67056, 7124, 5229]


def read_array(path: Path, key: Optional[str], ndim: int) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}. Run scripts/download_data.py; see data/README.md.")
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            shapes = {}
            handle.visititems(lambda name, obj: shapes.update({name: obj.shape})
                              if isinstance(obj, h5py.Dataset) and not name.startswith("#") else None)
            candidates = [name for name, shape in shapes.items() if len(shape) == ndim]
            selected = choose_key(key, candidates, path)
            # MATLAB v7.3 stores dimension order in reverse. Generic HDF5 does not.
            dataset = handle[selected]
            array = dataset[()]
            if "MATLAB_class" in dataset.attrs:
                array = array.transpose(tuple(reversed(range(array.ndim))))
            return array, selected
    inventory = whosmat(str(path))
    candidates = [name for name, shape, kind in inventory if len(shape) == ndim
                  and kind not in ("cell", "struct", "char")]
    selected = choose_key(key, candidates, path)
    return loadmat(str(path), variable_names=[selected])[selected], selected


def choose_key(key, candidates, path):
    if key is not None:
        if key not in candidates:
            raise ValueError(f"Configured variable {key!r} is absent or has wrong dimensions in {path}; candidates={candidates}")
        return key
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous variables in {path}: {candidates}. Set cube_key/labels_key in config.")
    return candidates[0]


def validate_scene(cube, labels, expected_shape=None, require_all_classes=True):
    if cube.ndim != 3 or labels.ndim != 2 or cube.shape[:2] != labels.shape:
        raise ValueError(f"Cube {cube.shape} and labels {labels.shape} must have matching spatial dimensions")
    if expected_shape is not None and tuple(cube.shape) != tuple(expected_shape):
        raise ValueError(f"Expected cube shape {expected_shape}; observed {cube.shape}. Inspect files/configuration.")
    if cube.dtype.kind not in "iuf" or labels.dtype.kind not in "iuf":
        raise ValueError("Cube and labels must be real numeric arrays")
    for start in range(0, cube.shape[0], 32):
        if not np.isfinite(cube[start:start + 32]).all():
            raise ValueError("Cube contains NaN or infinite measurements")
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Labels must be finite integers")
    values, counts = np.unique(labels, return_counts=True)
    if np.any((values < 0) | (values > 9)):
        raise ValueError(f"Expected background 0 and class IDs 1–9, found {values.tolist()}")
    missing = [int(c) for c in sorted(set(CLASS_IDS) - set(values))]
    if require_all_classes and missing:
        raise ValueError(f"Missing classes: {missing}")
    return {"cube_shape": list(cube.shape), "label_shape": list(labels.shape),
            "class_counts": {str(int(k)): int(v) for k, v in zip(values, counts)},
            "labelled_pixels": int(np.count_nonzero(labels)), "missing_classes": missing}


def load_scene(config: dict, root: Path):
    cube, cube_key = read_array(root / config["cube_path"], config.get("cube_key"), 3)
    labels, labels_key = read_array(root / config["labels_path"], config.get("labels_key"), 2)
    axes = config.get("cube_axes")
    if axes is not None:
        if sorted(axes) != [0, 1, 2]:
            raise ValueError("cube_axes must be a permutation of [0, 1, 2]")
        cube = cube.transpose(axes)
    elif cube.shape[:2] != labels.shape:
        eligible = [p for p in itertools.permutations(range(3))
                    if tuple(cube.shape[a] for a in p[:2]) == labels.shape]
        if len(eligible) != 1:
            raise ValueError("Cannot uniquely align cube and label dimensions; set cube_axes explicitly")
        cube = cube.transpose(eligible[0])
    summary = validate_scene(cube, labels, config.get("expected_shape"), config.get("require_all_classes", True))
    summary.update(cube_key=cube_key, labels_key=labels_key,
                   documented_counts_match=[summary["class_counts"].get(str(i), 0)
                                            for i in CLASS_IDS] == DOCUMENTED_COUNTS)
    return np.ascontiguousarray(cube, dtype=np.float32), np.ascontiguousarray(labels, dtype=np.int16), summary
