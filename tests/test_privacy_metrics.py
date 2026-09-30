"""Offline tests for evaluation/privacy_metrics.py.

Synthetic embeddings only - this is the end-to-end smoke test of the whole
template_protection package (key derivation -> template generation ->
comparison) exercised through the same experiment functions a real report
would use, before any backend/database/HTTP code exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evaluation.privacy_metrics import (
    experiment_diversity,
    experiment_protected_far_frr,
    experiment_revocability,
    experiment_similarity_preservation,
    save_csv_report,
)
from template_protection.utils import l2_normalize

MASTER_SECRET = "unit-test-master-secret-not-for-production"
APPLICATION_ID = "capstone-demo"


def _random_embedding(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.standard_normal(dim).astype(np.float32))


@pytest.fixture
def synthetic_identities():
    """3 identities x 3 samples each, with per-identity samples clustered
    (small perturbations of a base vector) so genuine pairs are more similar
    than impostor pairs - mirrors tests/conftest.py's "no GPU/dataset needed"
    philosophy for evaluation/experiments.py's existing tests."""
    embeddings, labels = [], []
    for identity in range(3):
        rng = np.random.default_rng(identity)
        base = rng.standard_normal(512).astype(np.float32)
        for sample in range(3):
            noise = np.random.default_rng(identity * 100 + sample).normal(scale=0.05, size=512).astype(np.float32)
            embeddings.append(l2_normalize(base + noise))
            labels.append(f"identity-{identity}")
    return embeddings, labels


def test_similarity_preservation_reports_a_correlation(synthetic_identities):
    embeddings, labels = synthetic_identities
    result = experiment_similarity_preservation(
        embeddings, labels, MASTER_SECRET, APPLICATION_ID, modality="face", output_bits=256
    )
    assert result["num_pairs"] == len(embeddings) * (len(embeddings) - 1) // 2
    assert -1.0 <= result["correlation"] <= 1.0
    # Protected similarity should track raw similarity reasonably well for
    # recognition to remain usable, not perfectly (it's a lossy transform).
    assert result["correlation"] > 0.3


def test_revocability_pairwise_distances_are_near_random(synthetic_identities):
    embeddings, _ = synthetic_identities
    result = experiment_revocability(
        embeddings[0], MASTER_SECRET, APPLICATION_ID, user_id="U001", modality="face", num_rotations=4
    )
    assert len(result["pairwise_hamming_distances"]) == 6  # C(4, 2)
    assert 0.3 < result["mean_hamming_distance"] < 0.7


def test_diversity_pairwise_distances_are_near_random(synthetic_identities):
    embeddings, _ = synthetic_identities
    result = experiment_diversity(
        embeddings[0],
        MASTER_SECRET,
        user_id="U001",
        modality="face",
        application_ids=["app-a", "app-b", "app-c"],
    )
    assert len(result["pairwise_hamming_distances"]) == 3
    assert 0.3 < result["mean_hamming_distance"] < 0.7


def test_protected_far_frr_reports_a_plausible_eer(synthetic_identities):
    embeddings, labels = synthetic_identities
    result = experiment_protected_far_frr(embeddings, labels, MASTER_SECRET, APPLICATION_ID, modality="face")
    assert result["num_genuine_pairs"] == 3 * 3  # C(3,2) genuine pairs per identity * 3 identities
    assert 0.0 <= result["eer"] <= 1.0


def test_protected_far_frr_requires_genuine_and_impostor_pairs():
    single_identity_embeddings = [_random_embedding(512, seed) for seed in range(3)]
    single_identity_labels = ["only-identity"] * 3
    with pytest.raises(ValueError):
        experiment_protected_far_frr(single_identity_embeddings, single_identity_labels, MASTER_SECRET, APPLICATION_ID, "face")


def test_save_csv_report_writes_a_readable_csv(tmp_path: Path, synthetic_identities):
    embeddings, _ = synthetic_identities
    report = experiment_revocability(embeddings[0], MASTER_SECRET, APPLICATION_ID, "U001", "face")
    output_path = tmp_path / "revocability_report.csv"

    save_csv_report(report, output_path)

    assert output_path.exists()
    contents = output_path.read_text(encoding="utf-8")
    assert "experiment" in contents
    assert "revocability" in contents
    assert "mean_hamming_distance" in contents
