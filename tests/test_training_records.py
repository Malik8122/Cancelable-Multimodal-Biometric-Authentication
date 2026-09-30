"""Training record infrastructure (training/recorder.py), training configurations and checkpoint loading."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from training.recorder import LOG_COLUMNS, NA, TrainingRecorder, environment, plot_run


def test_epoch_log_is_written_after_every_epoch_and_fills_unknowns(tmp_path):
    rec = TrainingRecorder("unit", {"model": "m", "random_seed": 1}, experiment_id="run1", root=tmp_path / "unit" / "runs")
    rec.log_epoch(1, train_loss=1.5, validation_EER=0.2)
    rows = list(csv.DictReader((rec.dir / "training_log.csv").open(encoding="utf-8")))
    assert len(rows) == 1  # on disk before the next epoch starts
    assert list(rows[0]) == LOG_COLUMNS
    assert rows[0]["train_loss"] == "1.5" and rows[0]["validation_AUC"] == NA and rows[0]["train_accuracy"] == NA
    rec.log_epoch(2, train_loss=1.0)
    assert len(list(csv.DictReader((rec.dir / "training_log.csv").open(encoding="utf-8")))) == 2


def test_unknown_metric_names_are_rejected(tmp_path):
    rec = TrainingRecorder("unit", {}, experiment_id="run2", root=tmp_path / "unit" / "runs")
    with pytest.raises(KeyError):
        rec.log_epoch(1, made_up_metric=0.1)


def test_config_environment_manifest_and_best_files(tmp_path):
    rec = TrainingRecorder("unit", {"model": "m", "random_seed": 7}, experiment_id="run3", root=tmp_path / "unit" / "runs")
    cfg = json.loads((rec.dir / "config.json").read_text())
    assert cfg["experiment_id"] == "run3" and cfg["modality"] == "unit" and cfg["git_commit"]
    env = json.loads((rec.dir / "environment.json").read_text())
    assert {"python", "pytorch", "cuda", "gpu", "cpu", "ram_gb", "git_commit", "git_branch"} <= set(env)
    rec.record_checkpoint("best.pt", 3, validation_EER=0.05)
    rec.record_checkpoint("best.pt", 5, validation_EER=0.04)  # same file re-saved -> one manifest entry
    manifest = json.loads((rec.dir / "checkpoints" / "checkpoint_manifest.json").read_text())
    assert len(manifest) == 1 and manifest[0]["epoch"] == 5 and manifest[0]["random_seed"] == 7
    rec.record_best(5, 0.04, "validation_EER (minimum)", "x/best.pt", "min EER")
    assert json.loads((rec.dir / "best_checkpoint.json").read_text())["best_epoch"] == 5


def test_figures_only_for_metrics_that_exist(tmp_path):
    rec = TrainingRecorder("unit", {}, experiment_id="run4", root=tmp_path / "unit" / "runs")
    for e in (1, 2, 3):
        rec.log_epoch(e, train_loss=1 / e, validation_loss=1.2 / e)
    rec.finish("COMPLETED")
    figs = {p.name for p in (rec.dir / "figures").glob("*.png")}
    assert figs == {"loss_vs_epoch.png"}  # no EER / AUC / accuracy values -> no such figures
    assert json.loads((rec.dir / "run_status.json").read_text())["status"] == "COMPLETED"


def test_environment_is_measured():
    env = environment()
    import platform

    assert env["python"] == platform.python_version()


def test_training_configurations_match_the_code():
    from models.voice.config import VoiceConfig

    v = VoiceConfig()
    assert (v.embedding_dim, v.num_epochs, v.batch_size, v.learning_rate, v.augmentation_enabled) == (192, 30, 64, 1e-3, False)
    assert (v.arcface_margin, v.arcface_scale, v.train_val_test_split) == (0.5, 30.0, (0.7, 0.15, 0.15))


@pytest.mark.parametrize("modality, dim", [("face", 512), ("voice", 192)])
def test_deployed_checkpoints_load_and_have_the_documented_dimension(modality, dim):
    from embeddings.constants import DEFAULT_CHECKPOINTS

    path = Path(DEFAULT_CHECKPOINTS[modality])
    if not path.exists() or path.stat().st_size < 1_000_000:
        pytest.skip("checkpoint not pulled from Git LFS")
    from models.face.inference import FaceEmbedder
    from models.voice.inference import VoiceEmbedder

    emb = {"face": FaceEmbedder, "voice": VoiceEmbedder}[modality](checkpoint_path=path)
    assert emb.mock_mode is False and emb.embedding_dim == dim
    shape = {"face": (160, 160, 3), "voice": (80, 401, 3)}[modality]
    v = emb.extract_embedding(np.random.default_rng(0).standard_normal(shape).astype(np.float32))
    assert v.shape == (dim,) and abs(float(np.linalg.norm(v)) - 1.0) < 1e-4


def test_training_loops_expose_an_observation_only_epoch_callback():
    import inspect

    from models.voice.train import train as voice_train

    for fn in (voice_train,):
        p = inspect.signature(fn).parameters["epoch_callback"]
        assert p.default is None  # optional: existing callers are unchanged
