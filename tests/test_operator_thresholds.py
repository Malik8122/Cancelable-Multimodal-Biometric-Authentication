"""Which threshold each modality is judged against.

Face and voice are decided in their own metrics (backend/services/modality_metrics.py): face on the calibrated
estimate of the embedding cosine similarity (>= Settings.face_cosine_threshold, 0.80), voice on the calibrated estimate
of the Euclidean distance (<= Settings.voice_euclidean_threshold, 0.90). Face's is a teacher-requested project setting; voice's was raised from 0.75,
not genuine/impostor calibrations. They replaced the operator-specified evaluation/results/{face,voice}_threshold.json
files (0.80 on the raw Hamming similarity), which were removed so each modality has exactly one authoritative
threshold. The Hamming similarity rule against the 0.90 fallback remains only for a template length with no metric
calibration (see backend/services/modality_metrics.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

RESULTS = Path(__file__).resolve().parents[1] / "evaluation" / "results"


@pytest.fixture(autouse=True)
def _fresh_threshold_cache():
    from backend import threshold_loader

    threshold_loader.clear_cache()
    yield
    threshold_loader.clear_cache()


class _Stub:
    is_mock = False

    def embed(self, raw):
        return raw


def _pair(dim: int, cosine: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(dim)
    base /= np.linalg.norm(base)
    noise = rng.standard_normal(dim)
    noise -= (noise @ base) * base
    noise /= np.linalg.norm(noise)
    return base, cosine * base + np.sqrt(1 - cosine**2) * noise


def test_face_and_voice_have_no_second_hamming_threshold_file():
    assert not (RESULTS / "face_threshold.json").exists()
    assert not (RESULTS / "voice_threshold.json").exists()


def test_the_hamming_fallback_rule_keeps_the_0_9_threshold():
    from backend.threshold_loader import get_modality_threshold

    assert get_modality_threshold("face", 0.9) == 0.9
    assert get_modality_threshold("voice", 0.9) == 0.9


def test_each_modality_is_judged_in_its_own_metric(db_session):
    from backend.config import get_settings
    from backend.services.base_service import ModalityService

    settings = get_settings()
    face = ModalityService("face", _Stub(), settings)
    voice = ModalityService("voice", _Stub(), settings)
    voice.capture_gate = False  # stub input is an embedding/seed, not audio: no capture gate

    face_enrolled, face_live = _pair(512, 0.93)
    voice_enrolled, voice_live = _pair(192, 0.93, seed=1)
    face.enroll(db_session, face_enrolled, "u", "app")
    voice.enroll(db_session, voice_enrolled, "u", "app")

    face_result = face.authenticate(db_session, face_live, "u", "app")
    voice_result = voice.authenticate(db_session, voice_live, "u", "app")

    assert face_result.metric == "cosine_estimate" and face_result.metric_threshold == 0.80
    assert voice_result.metric == "euclidean_estimate" and voice_result.metric_threshold == 0.90
    # A true embedding cosine of 0.93 (Euclidean distance 0.37) is a genuine-looking capture for face and voice.
    assert face_result.authenticated is True and voice_result.authenticated is True


def test_an_unrelated_person_is_far_from_both_thresholds(db_session):
    """Impostor-level embeddings (cosine ~0) estimate far below 0.80 (face) and far above 0.90 (voice)."""
    from backend.config import get_settings
    from backend.services.base_service import ModalityService

    rng = np.random.default_rng(1)
    face = ModalityService("face", _Stub(), get_settings())
    voice = ModalityService("voice", _Stub(), get_settings())
    voice.capture_gate = False  # stub input is an embedding/seed, not audio: no capture gate
    face.enroll(db_session, rng.standard_normal(512), "u", "app")
    voice.enroll(db_session, rng.standard_normal(192), "u", "app")
    for _ in range(10):
        face_result = face.authenticate(db_session, rng.standard_normal(512), "u", "app")
        voice_result = voice.authenticate(db_session, rng.standard_normal(192), "u", "app")
        assert face_result.metric_value < 0.5 and face_result.authenticated is False
        assert voice_result.metric_value > 1.0 and voice_result.authenticated is False


def test_metrics_endpoint_reports_only_what_was_measured(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.config import get_settings
    from backend.database.session import get_engine, get_session_factory

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'm.db'}")
    for fn in (get_settings, get_engine, get_session_factory):
        fn.cache_clear()
    from backend.main import app

    with TestClient(app) as client:
        body = client.get("/metrics/face").json()
        health = client.get("/system/health").json()
    for fn in (get_settings, get_engine, get_session_factory):
        fn.cache_clear()
    # No protected-template genuine/impostor calibration exists for face: nothing calibrated is claimed.
    assert body["calibrated"] is False
    assert not {"calibrated_threshold", "calibrated_far", "calibrated_frr", "calibrated_eer", "calibrated_auc"} & set(body["metrics"])
    assert health["thresholds_loaded"] is False


def test_settings_carry_the_teacher_requested_thresholds():
    from backend.config import get_settings

    settings = get_settings()
    assert settings.face_cosine_threshold == 0.80 and settings.voice_euclidean_threshold == 0.90
    assert json.loads((RESULTS / "biohash_metric_calibration.json").read_text(encoding="utf-8"))["template_bits"] == settings.template_bits
