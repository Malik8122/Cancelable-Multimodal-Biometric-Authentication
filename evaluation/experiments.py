"""Experiment 1: individual modality recognition performance.

Given a list of (embedding, identity_label) pairs for a single modality,
builds genuine/impostor cosine-similarity score distributions and reports
accuracy, FAR, FRR, EER, and AUC.

Experiments 2-5 (protected-vs-unprotected, fusion, revocability,
security/privacy analysis) are added in Phase 2 and Phase 3 once the
cancelable transform and fusion layers exist - see docs/ROADMAP.md.
"""

from __future__ import annotations

import itertools

import numpy as np

from evaluation.metrics import accuracy_at_threshold, compute_eer, cosine_similarity
from evaluation.roc import compute_roc


def build_genuine_impostor_scores(
    embeddings: list[np.ndarray],
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-compares every sample against every other sample.

    Returns (genuine_scores, impostor_scores) as flat numpy arrays.
    """
    genuine, impostor = [], []
    for (emb_a, label_a), (emb_b, label_b) in itertools.combinations(zip(embeddings, labels), 2):
        score = cosine_similarity(emb_a, emb_b)
        (genuine if label_a == label_b else impostor).append(score)
    return np.array(genuine), np.array(impostor)


def run_modality_experiment(
    embeddings: list[np.ndarray],
    labels: list[str],
    modality_name: str,
) -> dict:
    """Experiment 1 for a single modality: full recognition-performance report."""
    genuine_scores, impostor_scores = build_genuine_impostor_scores(embeddings, labels)
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        raise ValueError(
            f"Need at least one genuine pair and one impostor pair to evaluate {modality_name}; "
            "make sure the sample set has multiple images per identity and multiple identities."
        )

    eer, eer_threshold = compute_eer(genuine_scores, impostor_scores)
    accuracy = accuracy_at_threshold(genuine_scores, impostor_scores, eer_threshold)
    roc = compute_roc(genuine_scores, impostor_scores)

    return {
        "modality": modality_name,
        "num_samples": len(embeddings),
        "num_genuine_pairs": len(genuine_scores),
        "num_impostor_pairs": len(impostor_scores),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "accuracy_at_eer_threshold": accuracy,
        "auc": roc["auc"],
        "roc": roc,
    }
