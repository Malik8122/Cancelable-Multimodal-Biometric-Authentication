"""evaluation/ module tests using synthetic score distributions."""

from __future__ import annotations

import numpy as np

from evaluation.experiments import build_genuine_impostor_scores, run_modality_experiment
from evaluation.metrics import accuracy_at_threshold, compute_eer, compute_far_frr, cosine_similarity


def test_cosine_similarity_identical_vectors_is_one():
    vector = np.array([1.0, 2.0, 3.0])
    assert np.isclose(cosine_similarity(vector, vector), 1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert np.isclose(cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0)


def test_far_frr_perfect_separation():
    genuine = np.array([0.9, 0.95, 0.8])
    impostor = np.array([0.1, 0.2, 0.05])
    far, frr = compute_far_frr(genuine, impostor, threshold=0.5)
    assert far == 0.0
    assert frr == 0.0


def test_eer_perfect_separation_is_zero():
    genuine = np.array([0.9, 0.95, 0.8, 0.85])
    impostor = np.array([0.1, 0.2, 0.05, 0.15])
    eer, _threshold = compute_eer(genuine, impostor)
    assert eer == 0.0


def test_accuracy_at_threshold_perfect_separation():
    genuine = np.array([0.9, 0.8])
    impostor = np.array([0.1, 0.2])
    assert accuracy_at_threshold(genuine, impostor, threshold=0.5) == 1.0


def test_build_genuine_impostor_scores_splits_correctly():
    embeddings = [np.array([1.0, 0.0]), np.array([0.99, 0.01]), np.array([0.0, 1.0])]
    labels = ["alice", "alice", "bob"]

    genuine, impostor = build_genuine_impostor_scores(embeddings, labels)

    assert len(genuine) == 1  # alice-alice pair
    assert len(impostor) == 2  # alice-bob, alice-bob


def test_run_modality_experiment_end_to_end():
    rng = np.random.default_rng(0)
    embeddings, labels = [], []
    for identity in ["alice", "bob", "carol"]:
        base = rng.standard_normal(16)
        for _ in range(3):
            noisy = base + rng.normal(scale=0.01, size=16)
            embeddings.append(noisy / np.linalg.norm(noisy))
            labels.append(identity)

    report = run_modality_experiment(embeddings, labels, modality_name="synthetic")

    assert report["modality"] == "synthetic"
    assert 0.0 <= report["eer"] <= 1.0
    assert 0.0 <= report["auc"] <= 1.0
    assert report["num_genuine_pairs"] > 0
    assert report["num_impostor_pairs"] > 0
