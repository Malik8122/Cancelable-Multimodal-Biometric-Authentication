"""Offline tests for evaluation/threshold_calibration.py.

Synthetic embeddings only (same fixture shape as tests/test_privacy_metrics.py) -
these prove the sweep/calibration *math* and file formats are correct, not
that any particular threshold is biometrically meaningful. Real calibration
needs real, held-out embeddings - see the module's own docstring.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.threshold_calibration import (
    calibrate_modality_threshold,
    load_threshold_report,
    save_protected_metrics_csv,
    save_threshold_json,
    sweep_thresholds,
)
from template_protection.utils import l2_normalize

MASTER_SECRET = "unit-test-master-secret-not-for-production"
APPLICATION_ID = "capstone-demo"


def _random_embedding(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.standard_normal(dim).astype(np.float32))


@pytest.fixture
def synthetic_identities():
    """4 identities x 4 samples each, clustered per identity - enough pairs
    for a stable sweep across every threshold from 0.50 to 1.00."""
    embeddings, labels = [], []
    for identity in range(4):
        rng = np.random.default_rng(identity)
        base = rng.standard_normal(512).astype(np.float32)
        for sample in range(4):
            noise = np.random.default_rng(identity * 100 + sample).normal(scale=0.05, size=512).astype(np.float32)
            embeddings.append(l2_normalize(base + noise))
            labels.append(f"identity-{identity}")
    return embeddings, labels


def test_sweep_thresholds_covers_the_full_range_and_sums_to_totals():
    genuine = np.array([0.95, 0.9, 0.85])
    impostor = np.array([0.6, 0.55, 0.5])
    sweep = sweep_thresholds(genuine, impostor, start=0.5, stop=1.0, step=0.1)

    assert sweep[0]["threshold"] == pytest.approx(0.5)
    assert sweep[-1]["threshold"] == pytest.approx(1.0)
    for row in sweep:
        assert row["tp"] + row["fn"] == len(genuine)
        assert row["tn"] + row["fp"] == len(impostor)


def test_sweep_thresholds_far_and_frr_move_in_opposite_directions():
    genuine = np.array([0.95, 0.9, 0.85, 0.8])
    impostor = np.array([0.6, 0.55, 0.5, 0.45])
    sweep = sweep_thresholds(genuine, impostor, start=0.5, stop=1.0, step=0.05)

    # A stricter (higher) threshold must never increase FAR or decrease FRR.
    for previous, current in zip(sweep, sweep[1:]):
        assert current["far"] <= previous["far"] + 1e-9
        assert current["frr"] >= previous["frr"] - 1e-9


def test_calibrate_modality_threshold_reports_a_plausible_operating_point(synthetic_identities):
    embeddings, labels = synthetic_identities
    report = calibrate_modality_threshold(embeddings, labels, MASTER_SECRET, APPLICATION_ID, modality="face")

    assert report["modality"] == "face"
    assert 0.5 <= report["threshold"] <= 1.0
    assert 0.0 <= report["eer"] <= 1.0
    assert 0.0 <= report["auc"] <= 1.0
    assert report["num_genuine_pairs"] == 4 * 6  # C(4,2) genuine pairs per identity * 4 identities
    assert len(report["sweep"]) > 0
    # The picked threshold must itself be one of the swept rows, so the JSON
    # summary and the CSV sweep can never silently disagree.
    assert any(row["threshold"] == report["threshold"] for row in report["sweep"])


def test_calibrate_modality_threshold_requires_genuine_and_impostor_pairs():
    single_identity_embeddings = [_random_embedding(512, seed) for seed in range(3)]
    single_identity_labels = ["only-identity"] * 3
    with pytest.raises(ValueError):
        calibrate_modality_threshold(single_identity_embeddings, single_identity_labels, MASTER_SECRET, APPLICATION_ID, "voice")


def test_save_threshold_json_writes_the_bug1_shape(tmp_path: Path, synthetic_identities):
    embeddings, labels = synthetic_identities
    report = calibrate_modality_threshold(embeddings, labels, MASTER_SECRET, APPLICATION_ID, modality="voice")
    output_path = tmp_path / "voice_threshold.json"

    save_threshold_json(report, output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["threshold"] == report["threshold"]
    assert saved["eer"] == report["eer"]
    assert saved["far"] == report["far"]
    assert saved["frr"] == report["frr"]
    assert saved["auc"] == report["auc"]


def test_save_protected_metrics_csv_writes_one_row_per_threshold(tmp_path: Path, synthetic_identities):
    embeddings, labels = synthetic_identities
    report = calibrate_modality_threshold(embeddings, labels, MASTER_SECRET, APPLICATION_ID, modality="face")
    output_path = tmp_path / "face_protected_metrics.csv"

    save_protected_metrics_csv(report, output_path)

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "threshold,accuracy,far,frr,eer,tp,fp,tn,fn"
    assert len(lines) - 1 == len(report["sweep"])


def test_load_threshold_report_round_trips(tmp_path: Path, synthetic_identities):
    embeddings, labels = synthetic_identities
    report = calibrate_modality_threshold(embeddings, labels, MASTER_SECRET, APPLICATION_ID, modality="face")
    save_threshold_json(report, tmp_path / "face_threshold.json")

    loaded = load_threshold_report("face", results_dir=tmp_path)
    assert loaded is not None
    assert loaded["threshold"] == report["threshold"]


def test_load_threshold_report_returns_none_when_missing(tmp_path: Path):
    assert load_threshold_report("voice", results_dir=tmp_path) is None
