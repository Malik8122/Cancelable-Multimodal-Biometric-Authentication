"""Integration test: real embedder output -> protected-template calibration -> CSV/JSON.

tests/test_threshold_calibration.py already covers the calibration *math*
against synthetic embeddings directly. This file instead runs the layer
above that: real `FaceEmbedder` inference (the real trained checkpoint,
guarded by `pytest.importorskip` like the rest of this suite) on synthetic
aligned-size arrays (MTCNN detection is bypassed, as in
tests/test_authentication_debug_sprint.py), proving the real embedder's output
is actually compatible end-to-end with `evaluation/threshold_calibration.py`
and the Bug-2 CSV schema - not that the resulting numbers are biometrically
meaningful (they aren't: synthetic arrays aren't real faces with genuine
identity variation - see the module's own docstring on this).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from evaluation.threshold_calibration import calibrate_modality_threshold, save_protected_metrics_csv, save_threshold_json

MASTER_SECRET = "unit-test-master-secret-not-for-production"
APPLICATION_ID = "capstone-demo"

EXPECTED_CSV_COLUMNS = ["threshold", "accuracy", "far", "frr", "eer", "tp", "fp", "tn", "fn"]


def _synthetic_face_sample(identity: int, sample: int) -> np.ndarray:
    """A distinct base array per identity plus small per-sample noise, so different
    "identities" produce genuinely different real embeddings - not the same
    array reused, which would make every genuine/impostor pair identical."""
    base = np.random.default_rng(identity).integers(0, 256, size=(160, 160, 3)).astype(np.float32)
    noise = np.random.default_rng(identity * 10 + sample).normal(scale=8, size=base.shape)
    return (base + noise).clip(0, 255).astype(np.uint8)


@pytest.fixture
def real_face_embeddings_and_labels():
    pytest.importorskip("facenet_pytorch", reason="facenet-pytorch not installed in this environment")
    from embeddings.constants import DEFAULT_CHECKPOINTS
    from models.face.inference import FaceEmbedder

    embedder = FaceEmbedder(checkpoint_path=DEFAULT_CHECKPOINTS["face"])
    assert embedder.mock_mode is False, "expected the real trained face checkpoint to load"

    embeddings, labels = [], []
    for identity in range(3):
        for sample in range(3):
            embeddings.append(embedder.extract_embedding(_synthetic_face_sample(identity, sample)))
            labels.append(f"identity-{identity}")
    return embeddings, labels


def test_real_embedder_output_calibrates_end_to_end(real_face_embeddings_and_labels):
    embeddings, labels = real_face_embeddings_and_labels
    report = calibrate_modality_threshold(embeddings, labels, MASTER_SECRET, APPLICATION_ID, modality="face")

    assert report["num_genuine_pairs"] == 3 * 3  # C(3,2) per identity * 3 identities
    assert report["num_impostor_pairs"] > 0
    assert 0.5 <= report["threshold"] <= 1.0
    assert 0.0 <= report["eer"] <= 1.0


def test_protected_metrics_csv_has_the_bug2_required_columns(tmp_path: Path, real_face_embeddings_and_labels):
    embeddings, labels = real_face_embeddings_and_labels
    report = calibrate_modality_threshold(embeddings, labels, MASTER_SECRET, APPLICATION_ID, modality="face")
    csv_path = tmp_path / "face_protected_metrics.csv"

    save_protected_metrics_csv(report, csv_path)

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == EXPECTED_CSV_COLUMNS
        rows = list(reader)

    assert len(rows) == len(report["sweep"])
    for row in rows:
        # Every declared column round-trips through the file as a real value,
        # not blank/placeholder.
        for column in EXPECTED_CSV_COLUMNS:
            assert row[column] != ""
        assert int(row["tp"]) + int(row["fn"]) == report["num_genuine_pairs"]
        assert int(row["tn"]) + int(row["fp"]) == report["num_impostor_pairs"]


def test_threshold_json_and_protected_csv_agree_on_the_picked_threshold(tmp_path: Path, real_face_embeddings_and_labels):
    embeddings, labels = real_face_embeddings_and_labels
    report = calibrate_modality_threshold(embeddings, labels, MASTER_SECRET, APPLICATION_ID, modality="face")

    json_path = tmp_path / "face_threshold.json"
    csv_path = tmp_path / "face_protected_metrics.csv"
    save_threshold_json(report, json_path)
    save_protected_metrics_csv(report, csv_path)

    import json

    saved_threshold = json.loads(json_path.read_text(encoding="utf-8"))["threshold"]
    with open(csv_path, newline="", encoding="utf-8") as handle:
        csv_thresholds = {float(row["threshold"]) for row in csv.DictReader(handle)}

    assert saved_threshold in csv_thresholds
