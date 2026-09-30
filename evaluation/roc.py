"""ROC curve / AUC computation and plotting, shared across phases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve


def compute_roc(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> dict:
    """Returns fpr, tpr, thresholds, and AUC for a genuine/impostor score set."""
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    y_score = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": float(auc(fpr, tpr))}


def plot_roc(roc_results: dict[str, dict], save_path: str | Path | None = None):
    """Plot one or more ROC curves on the same axes.

    `roc_results` maps a label (e.g. "Face", "Voice") to the
    dict returned by `compute_roc`.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, result in roc_results.items():
        ax.plot(result["fpr"], result["tpr"], label=f"{label} (AUC = {result['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig
