"""Partition generation and auditable training-only preprocessing."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def assert_disjoint(*partitions):
    combined = np.concatenate(partitions)
    if np.unique(combined).size != combined.size:
        raise ValueError("Leakage: partitions overlap or contain repeated indices")


def make_split(labels, config, seed, protocol="pixel"):
    flat = labels.ravel()
    labelled = np.flatnonzero(flat > 0)
    fractions = config["split"]
    if protocol == "pixel":
        train_pool, rest = train_test_split(labelled, train_size=fractions["train"],
                                            stratify=flat[labelled], random_state=seed)
        validation, test = train_test_split(rest, test_size=fractions["test"] / (1 - fractions["train"]),
                                            stratify=flat[rest], random_state=seed)
    elif protocol == "spatial_block":
        settings = config["spatial_block"]
        buffer = settings["buffer"]
        if buffer < config["spatial_window"] - 1:
            raise ValueError("Block buffer must be >= window-1 to separate spatial feature footprints")
        rows, cols = np.indices(labels.shape)
        blocks = rows // settings["block_size"] * (1 + (labels.shape[1] - 1) // settings["block_size"]) + cols // settings["block_size"]
        ids = np.unique(blocks)
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
        ntrain, nval = int(len(ids) * fractions["train"]), int(len(ids) * fractions["validation"])
        groups = [ids[:ntrain], ids[ntrain:ntrain + nval], ids[ntrain + nval:]]
        partitions = []
        for group in groups:
            mask = np.isin(blocks, group)
            if buffer:
                mask = binary_erosion(mask, iterations=buffer, border_value=0)
            partitions.append(np.flatnonzero(mask.ravel() & (flat > 0)))
        train_pool, validation, test = partitions
    else:
        raise ValueError(f"Unknown protocol {protocol}")
    assert_disjoint(train_pool, validation, test)
    classes = np.unique(flat[labelled])
    for name, indices in [("train", train_pool), ("validation", validation), ("test", test)]:
        if not np.array_equal(np.unique(flat[indices]), classes):
            raise ValueError(f"{protocol} split seed {seed}: {name} lacks classes; choose block design before evaluation")
    # One shared capped subset is used by all methods. Discarded training pixels never enter evaluation.
    rng = np.random.default_rng(seed)
    train = np.concatenate([rng.choice(train_pool[flat[train_pool] == c],
                                       min(config["train_cap_per_class"], np.sum(flat[train_pool] == c)),
                                       replace=False) for c in classes])
    rng.shuffle(train)
    assert_disjoint(train, validation, test)
    return {"train": train, "train_pool": train_pool, "validation": validation, "test": test}


def training_scaler(features, train_indices):
    """Only the explicitly supplied training rows can determine fitted statistics."""
    scaler = StandardScaler()
    scaler.fit(features[train_indices])
    return scaler


def training_noise_pairs(cube, train_indices, max_pairs=20000, seed=0, min_pairs=10):
    """Horizontal/vertical one-step differences, both endpoints in capped training set.

    Cov((x_p-x_q)/sqrt(2)) estimates noise under independent equal-covariance
    neighbour noise and locally constant signal. Signal edges can bias it upward.
    """
    mask = np.zeros(cube.shape[:2], dtype=bool)
    mask.ravel()[train_indices] = True
    grid = np.arange(mask.size).reshape(mask.shape)
    hp = mask[:, :-1] & mask[:, 1:]
    vp = mask[:-1, :] & mask[1:, :]
    first = np.concatenate([grid[:, :-1][hp], grid[:-1, :][vp]])
    second = np.concatenate([grid[:, 1:][hp], grid[1:, :][vp]])
    if len(first) < min_pairs:
        raise ValueError(f"MNF needs >= {min_pairs} training-only neighbour pairs; found {len(first)}. Increase training cap or predeclare a lower pair minimum.")
    keep = np.random.default_rng(seed).choice(len(first), min(len(first), max_pairs), replace=False)
    pairs = np.column_stack([first[keep], second[keep]])
    if not np.isin(pairs, train_indices).all():
        raise AssertionError("Noise pair includes a non-training pixel")
    return pairs
