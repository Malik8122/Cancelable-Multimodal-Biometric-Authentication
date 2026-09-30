"""One embedding dimension per modality, everywhere: models, calibration, evaluation scripts (FACE 512, VOICE 192)."""

from __future__ import annotations

import json

import numpy as np
import pytest

EXPECTED = {"face": 512, "voice": 192}


def test_model_constants():
    from models.face.inference import FACE_EMBEDDING_DIM
    from models.voice.inference import VOICE_EMBEDDING_DIM

    assert {"face": FACE_EMBEDDING_DIM, "voice": VOICE_EMBEDDING_DIM} == EXPECTED


def test_calibration_script_dims_match_models():
    from scripts.calibrate_biohash_metric_mapping import MODALITY_DIMS

    assert MODALITY_DIMS == EXPECTED


def test_template_set_experiment_dims_match_models():
    from evaluation.template_set_experiments import _MODALITY_DIMS

    assert _MODALITY_DIMS == EXPECTED


def test_committed_calibration_was_fitted_at_the_model_dims():
    from template_protection.metric_estimation import CALIBRATION_PATH

    report = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    assert report["template_bits"] == 256
    assert {m: e["embedding_dim"] for m, e in report["modalities"].items()} == EXPECTED
    for entry in report["modalities"].values():
        assert entry["fit_statistics"]["evidence_label"] == "SYNTHETIC CALIBRATION"
        assert entry["fit_statistics"]["R2"] > 0.999


@pytest.mark.parametrize("modality", sorted(EXPECTED))
def test_mock_embedders_emit_the_model_dim(modality):
    """Without a checkpoint each embedder falls back to a deterministic mock of the SAME dimension."""
    if modality == "face":
        from models.face.inference import FaceEmbedder as E
        sample = np.zeros((160, 160, 3), dtype=np.uint8)
    else:
        from models.voice.inference import VoiceEmbedder as E
        sample = np.zeros((80, 200, 3), dtype=np.float32)  # (n_mels, n_frames, 3) pseudo-RGB log-mel
    embedder = E(checkpoint_path=None)
    assert embedder.mock_mode
    assert embedder.extract_embedding(sample).shape == (EXPECTED[modality],)


@pytest.mark.parametrize("modality", sorted(EXPECTED))
def test_biohash_accepts_each_dim_and_emits_256_bits(modality):
    from template_protection.biohash import generate_template
    from template_protection.hkdf_keys import derive_key

    key = derive_key("dimension-test-secret", application_id="t", user_id="u", modality=modality, key_version=1)
    v = np.random.default_rng(0).standard_normal(EXPECTED[modality])
    assert generate_template(v / np.linalg.norm(v), key, 256).shape == (256,)
