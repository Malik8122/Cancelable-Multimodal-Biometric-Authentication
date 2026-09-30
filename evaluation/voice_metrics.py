"""Voice-specific evaluation report, built on the same shared primitives
every other modality's evaluation already uses (evaluation/metrics.py,
evaluation/roc.py), including `precision_recall_f1`/`confusion_matrix_at_threshold`
(moved to `evaluation/metrics.py` so there's exactly one implementation).

Mirrors evaluation/experiments.py::run_modality_experiment's shape and
extends it - same genuine/impostor cosine-similarity scoring, plus the
additional metrics table and CSV/confusion-matrix export the spec wants
specifically for voice.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from evaluation.experiments import build_genuine_impostor_scores
from evaluation.metrics import accuracy_at_threshold, compute_eer, compute_far_frr, confusion_matrix_at_threshold, precision_recall_f1
from evaluation.roc import compute_roc


def run_voice_experiment(embeddings: list[np.ndarray], labels: list[str]) -> dict:
    """Full Voice evaluation report: accuracy, precision/recall/F1, FAR/FRR, EER, ROC/AUC.

    Threshold is chosen at the EER point (`evaluation/metrics.py::compute_eer`),
    the same convention `evaluation/experiments.py::run_modality_experiment`
    uses for the other three modalities, so results are directly comparable.
    """
    genuine_scores, impostor_scores = build_genuine_impostor_scores(embeddings, labels)
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        raise ValueError(
            "Need at least one genuine pair and one impostor pair to evaluate voice; "
            "make sure the sample set has multiple utterances per speaker and multiple speakers."
        )

    eer, eer_threshold = compute_eer(genuine_scores, impostor_scores)
    accuracy = accuracy_at_threshold(genuine_scores, impostor_scores, eer_threshold)
    far, frr = compute_far_frr(genuine_scores, impostor_scores, eer_threshold)
    prf1 = precision_recall_f1(genuine_scores, impostor_scores, eer_threshold)
    roc = compute_roc(genuine_scores, impostor_scores)
    confusion = confusion_matrix_at_threshold(genuine_scores, impostor_scores, eer_threshold)

    return {
        "modality": "voice",
        "num_samples": len(embeddings),
        "num_genuine_pairs": len(genuine_scores),
        "num_impostor_pairs": len(impostor_scores),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "accuracy": accuracy,
        "far": far,
        "frr": frr,
        "precision": prf1["precision"],
        "recall": prf1["recall"],
        "f1": prf1["f1"],
        "auc": roc["auc"],
        "roc": roc,
        "confusion_matrix": confusion,
    }


def save_voice_metrics_csv(report: dict, path: str | Path) -> None:
    """Write the scalar metrics from `run_voice_experiment`'s report to a CSV.

    Only scalar fields are written as rows - `roc` (arrays) and
    `confusion_matrix` are saved to their own sibling files by
    `save_roc_csv`/`save_confusion_matrix_csv` instead, so this file stays a
    simple, spreadsheet-friendly metrics table per the spec's "Save: CSV
    metrics" ask.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_fields = (
        "modality", "num_samples", "num_genuine_pairs", "num_impostor_pairs",
        "eer", "eer_threshold", "accuracy", "far", "frr", "precision", "recall", "f1", "auc",
    )
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        for field in scalar_fields:
            writer.writerow([field, report[field]])


def save_roc_csv(report: dict, path: str | Path) -> None:
    """Write the ROC curve (fpr, tpr, threshold triples) from a `run_voice_experiment` report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    roc = report["roc"]
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["fpr", "tpr", "threshold"])
        for fpr, tpr, threshold in zip(roc["fpr"], roc["tpr"], roc["thresholds"]):
            writer.writerow([fpr, tpr, threshold])


def save_confusion_matrix_csv(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = report["confusion_matrix"]
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["", "predicted_impostor", "predicted_genuine"])
        writer.writerow(["actual_impostor", matrix[0][0], matrix[0][1]])
        writer.writerow(["actual_genuine", matrix[1][0], matrix[1][1]])
