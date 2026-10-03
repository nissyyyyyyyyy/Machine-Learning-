"""
visualization.py
------------------
All charts live here, and only here - keeping plotting code out of the menu
logic makes both easier to read and test independently.

Every plot function does two things:
  1. Saves a PNG into the plots/ folder (so you always have a copy for your
     report/slides, even if you're running in a terminal that can't pop up
     a window).
  2. Tries to display the plot interactively with plt.show(). If the
     environment has no display (e.g. a headless server), this fails
     silently and you still get the saved PNG.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src import config


def _save_and_show(fig, filename: str) -> str:
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(config.PLOTS_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    try:
        plt.show()
    except Exception:
        pass  # no display available - the saved file is still there
    plt.close(fig)
    return out_path


def plot_quality_distribution(df, target_column: str) -> str:
    counts = df[target_column].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#7b2d26")
    ax.set_title("Wine Quality Distribution")
    ax.set_xlabel(target_column)
    ax.set_ylabel("Number of samples")
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return _save_and_show(fig, "quality_distribution.png")


def plot_feature_importance(importance_pairs) -> str:
    names = [p[0] for p in importance_pairs]
    values = [p[1] for p in importance_pairs]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color="#4a7c59")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # highest importance at the top
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Random Forest)")
    fig.tight_layout()
    return _save_and_show(fig, "feature_importance.png")


def plot_confusion_matrix(confusion, class_names) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = confusion.max() / 2 if confusion.size else 0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            value = confusion[i, j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return _save_and_show(fig, "confusion_matrix.png")


def plot_correlation_heatmap(df, feature_columns) -> str:
    corr = df[feature_columns].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_columns)))
    ax.set_yticks(range(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=90)
    ax.set_yticklabels(feature_columns)
    ax.set_title("Feature Correlation Heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return _save_and_show(fig, "correlation_heatmap.png")
