"""Noninteractive figures using one class palette and actual measurements only."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .data_loader import CLASS_IDS, CLASS_NAMES

# Distinct hues plus luminance differences; inspect printed figures for accessibility.
COLORS = ["#eeeeee", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
          "#D55E00", "#332288", "#999933", "#882255"]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(np.arange(-0.5, 10.5), CMAP.N)


def save(fig, path, dpi=300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def map_axis(ax, array, title):
    ax.imshow(array, cmap=CMAP, norm=NORM, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Column (pixel)")
    ax.set_ylabel("Row (pixel)")


def legend(fig):
    fig.legend(handles=[Patch(color=COLORS[c], label=name) for c, name in zip(CLASS_IDS, CLASS_NAMES)],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))


def dataset_figures(cube, labels, output, dpi=300):
    bands = np.linspace(0, cube.shape[2] - 1, 3).astype(int)[::-1]
    rgb = cube[:, :, bands].astype(np.float64)
    lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-12), 0, 1)
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.set(title=f"False colour (zero-based bands {bands.tolist()})", xlabel="Column (pixel)", ylabel="Row (pixel)")
    save(fig, output / "false_colour.png", dpi)
    fig, ax = plt.subplots(figsize=(8, 8))
    map_axis(ax, labels, "Ground truth; grey = unlabelled")
    legend(fig)
    save(fig, output / "ground_truth.png", dpi)
    fig, ax = plt.subplots(figsize=(9, 5))
    for c, name in zip(CLASS_IDS, CLASS_NAMES):
        if np.any(labels == c):
            # Class means are descriptive plots only and never feed model selection.
            ax.plot(cube[labels == c].mean(axis=0), label=name, color=COLORS[c])
    ax.set(xlabel="Spectral band (zero-based index)", ylabel="Mean recorded measurement", title="Descriptive class spectra (entire scene)")
    ax.legend(fontsize=7)
    save(fig, output / "spectral_signatures.png", dpi)


def fisher_figure(selector, bands, path, dpi=300):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    indices = np.arange(len(selector.scores_))
    axes[0].plot(indices, selector.scores_)
    axes[0].scatter(bands, selector.scores_[bands], color="#D55E00", s=12, label="Selected")
    axes[0].set(xlabel="Band (zero-based index)", ylabel="Fisher score", title="Training-only band scores")
    axes[0].legend()
    axes[1].plot(np.arange(len(indices)) + 1, selector.scores_[selector.ranking_])
    axes[1].set(xlabel="Fisher rank (one-based)", ylabel="Fisher score", title="Informative band ranking")
    save(fig, path, dpi)


def confusion_figures(matrix, output, name, dpi=300):
    matrix = np.asarray(matrix)
    for normalized in [False, True]:
        values = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1) if normalized else matrix
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        for i in range(9):
            for j in range(9):
                text = f"{100 * values[i, j]:.1f}" if normalized else str(matrix[i, j])
                ax.text(j, i, text, ha="center", va="center", fontsize=7,
                        color="white" if values[i, j] > values.max() * .55 else "black")
        ax.set_xticks(range(9), CLASS_NAMES, rotation=60, ha="right")
        ax.set_yticks(range(9), CLASS_NAMES)
        ax.set(xlabel="Predicted class", ylabel="True class", title=name + (" (%)" if normalized else " (counts)"))
        fig.colorbar(im, ax=ax, label="Fraction" if normalized else "Pixels")
        save(fig, output / f"{name}_confusion_{'normalized' if normalized else 'counts'}.png", dpi)


def prediction_figures(labels, prediction, test_indices, output, name, dpi=300):
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    map_axis(axes[0], labels, "Ground truth")
    map_axis(axes[1], prediction, name + " (all labelled pixels; includes training)")
    legend(fig)
    save(fig, output / f"{name}_comparison.png", dpi)
    fig, ax = plt.subplots(figsize=(7, 8))
    map_axis(ax, prediction, name + " descriptive map (includes training)")
    legend(fig)
    save(fig, output / f"{name}_prediction.png", dpi)
    error = np.full(labels.size, np.nan)
    error[test_indices] = (prediction.ravel()[test_indices] != labels.ravel()[test_indices]).astype(int)
    fig, ax = plt.subplots()
    cmap = ListedColormap(["#009E73", "#D55E00"])
    im = ax.imshow(error.reshape(labels.shape), cmap=cmap, vmin=0, vmax=1)
    ax.set(title=name + " held-out test errors", xlabel="Column (pixel)", ylabel="Row (pixel)")
    fig.colorbar(im, ax=ax, ticks=[0, 1], label="0 correct; 1 incorrect; blank excluded")
    save(fig, output / f"{name}_test_errors.png", dpi)


def result_figures(output, summary, dpi=300):
    for protocol, frame in summary.groupby("protocol"):
        target = output / "figures" / protocol
        for metric in ["accuracy", "macro_f1"]:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(frame.method, frame[f"{metric}_mean"], yerr=frame[f"{metric}_std"].fillna(0), capsize=4)
            ax.set(xlabel="Method", ylabel=metric.replace("_", " "), ylim=(0, 1),
                   title=f"{protocol}: mean ± sample SD (see n_seeds in summary.csv)")
            save(fig, target / f"{metric}_comparison.png", dpi)
        fig, ax = plt.subplots()
        ax.errorbar(frame.feature_count_mean, frame.fit_seconds_mean,
                    yerr=frame.fit_seconds_std.fillna(0), fmt="o", capsize=4)
        for row in frame.itertuples():
            ax.annotate(row.method, (row.feature_count_mean, row.fit_seconds_mean))
        ax.set(xlabel="Retained feature count", ylabel="Selected model fit time (s)", title=protocol + " runtime and dimensionality")
        save(fig, target / "runtime_vs_features.png", dpi)
    paths = list((output / "metrics").glob("*/seed_*/selection_*.json"))
    if paths:
        import json
        rows = [r for p in paths for r in json.loads(p.read_text())]
        candidates = pd.DataFrame(rows)
        for protocol, group in candidates.groupby("protocol"):
            fig, ax = plt.subplots()
            for method, methods in group.groupby("method"):
                if method == "E0":
                    continue
                # Best hyperparameter per feature count per seed, strictly validation scores.
                best = methods.groupby(["seed", "feature_count"]).validation_macro_f1.max().reset_index()
                curves = best.groupby("feature_count").validation_macro_f1.agg(["mean", "std"])
                ax.errorbar(curves.index, curves["mean"], yerr=curves["std"].fillna(0), label=method, marker="o")
            ax.set(xlabel="Feature count", ylabel="Validation macro-F1", title=protocol + " feature-count selection curves")
            ax.legend()
            save(fig, output / "figures" / protocol / "validation_feature_curves.png", dpi)
