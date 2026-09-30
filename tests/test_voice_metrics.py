"""Offline tests for evaluation/voice_metrics.py. Synthetic embeddings only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evaluation.voice_metrics import run_voice_experiment, save_confusion_matrix_csv, save_roc_csv, save_voice_metrics_csv


@pytest.fixture
def synthetic_speaker_embeddings():
    embeddings, labels = [], []
    for speaker in range(4):
        rng = np.random.default_rng(speaker)
        base = rng.standard_normal(192).astype(np.float32)
        for _ in range(4):
            noise = np.random.default_rng(speaker * 10 + _).normal(scale=0.05, size=192).astype(np.float32)
            vector = base + noise
            embeddings.append(vector / np.linalg.norm(vector))
            labels.append(f"speaker-{speaker}")
    return embeddings, labels


def test_run_voice_experiment_reports_plausible_metrics(synthetic_speaker_embeddings):
    embeddings, labels = synthetic_speaker_embeddings
    report = run_voice_experiment(embeddings, labels)

    assert report["modality"] == "voice"
    assert 0.0 <= report["eer"] <= 1.0
    assert 0.0 <= report["accuracy"] <= 1.0
    assert 0.0 <= report["precision"] <= 1.0
    assert 0.0 <= report["recall"] <= 1.0
    assert 0.0 <= report["f1"] <= 1.0
    assert report["confusion_matrix"].shape == (2, 2)


def test_run_voice_experiment_requires_genuine_and_impostor_pairs():
    single_speaker = [np.random.default_rng(i).standard_normal(192) for i in range(3)]
    with pytest.raises(ValueError):
        run_voice_experiment(single_speaker, ["only-speaker"] * 3)


def test_save_voice_metrics_csv_writes_scalar_fields(tmp_path: Path, synthetic_speaker_embeddings):
    embeddings, labels = synthetic_speaker_embeddings
    report = run_voice_experiment(embeddings, labels)
    path = tmp_path / "voice_metrics.csv"

    save_voice_metrics_csv(report, path)

    contents = path.read_text(encoding="utf-8")
    assert "eer" in contents
    assert "precision" in contents


def test_save_roc_csv_writes_curve_points(tmp_path: Path, synthetic_speaker_embeddings):
    embeddings, labels = synthetic_speaker_embeddings
    report = run_voice_experiment(embeddings, labels)
    path = tmp_path / "voice_roc.csv"

    save_roc_csv(report, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "fpr,tpr,threshold"
    assert len(lines) > 1


def test_save_confusion_matrix_csv_writes_a_2x2_table(tmp_path: Path, synthetic_speaker_embeddings):
    embeddings, labels = synthetic_speaker_embeddings
    report = run_voice_experiment(embeddings, labels)
    path = tmp_path / "voice_confusion.csv"

    save_confusion_matrix_csv(report, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
