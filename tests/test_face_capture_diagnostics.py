"""Debug-only face capture-quality diagnostics: logged only under DEBUG_SCORES, after the decision, scalars only,
and never able to change a score or a decision."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.config import Settings
from backend.database.models import Base
from backend.services.base_service import ModalityService

QUALITY = {"verdict": "TOO_SMALL", "detection_confidence": 0.99, "sharpness": 80.0, "face_size_ratio": 0.08,
           "center_offset": 0.1, "roll_degrees": 2.0, "yaw_ratio": 0.05}


class _Pipeline:
    is_mock = False

    def __init__(self):
        self.quality_calls = 0

    def embed(self, raw):
        v = np.asarray(raw, dtype=np.float64)[::20, ::20].ravel()[:512] - 127.5
        return v / np.linalg.norm(v)

    def embed_poses(self, captures):
        return [self.embed(img) for _, img in captures], [{"pose": p, "status": "VALID"} for p, _ in captures]

    def capture_quality(self, image):
        self.quality_calls += 1
        return dict(QUALITY)


def _service(debug: bool):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    pipeline = _Pipeline()
    return db, pipeline, ModalityService("face", pipeline, Settings(master_secret="diag-test", debug_scores=debug))


def _images():
    rng = np.random.default_rng(3)
    enrolled = rng.integers(0, 256, (320, 320, 3), dtype=np.uint8)
    probe = np.clip(enrolled.astype(int) + rng.integers(-40, 40, enrolled.shape), 0, 255).astype(np.uint8)
    return enrolled, probe


@pytest.mark.parametrize("debug", [False, True])
def test_decision_is_identical_with_and_without_diagnostics(debug):
    enrolled, probe = _images()
    results = {}
    for flag in (False, True):
        db, _, svc = _service(flag)
        svc.enroll_poses(db, [(p, enrolled) for p in ("front", "left", "right", "up", "down")], "u", "app")
        r = svc.authenticate(db, probe, "u", "app")
        results[flag] = (r.hamming_similarity, r.metric_value, r.authenticated)
    assert results[False] == results[True]


def test_quality_is_computed_only_in_debug_mode_and_logged_as_scalars(caplog):
    enrolled, probe = _images()
    db, pipeline, svc = _service(False)
    svc.enroll_poses(db, [(p, enrolled) for p in ("front", "left", "right", "up", "down")], "u", "app")
    svc.authenticate(db, probe, "u", "app")
    assert pipeline.quality_calls == 0  # no extra work, nothing logged, when DEBUG_SCORES is off

    db, pipeline, svc = _service(True)
    with caplog.at_level(logging.INFO, logger="backend.face_debug"):
        svc.enroll_poses(db, [(p, enrolled) for p in ("front", "left", "right", "up", "down")], "u", "app")
        svc.authenticate(db, probe, "u", "app")
    assert pipeline.quality_calls == 6  # 5 enrollment poses + 1 authentication capture
    lines = [r.getMessage() for r in caplog.records if "capture_quality" in r.getMessage()]
    assert any("stage=authentication" in m and "verdict=TOO_SMALL" in m and "face_size_ratio=0.080" in m for m in lines)
    assert sum("stage=enrollment pose=" in m for m in lines) == 5
    for m in lines:  # scalars only: no arrays / landmark coordinates in the log
        body = m.split("capture_quality", 1)[1]
        assert "[" not in body and "array" not in body


def test_face_pipeline_capture_quality_reports_the_enrollment_gate(monkeypatch):
    from embeddings.pipelines import FacePipeline
    from preprocessing.face import FaceDetection

    pipeline = FacePipeline(checkpoint_path=None)
    det = FaceDetection(aligned=np.zeros((160, 160, 3), np.uint8), probability=0.99, sharpness=12.0, face_size_ratio=0.4,
                        center_offset=0.1, roll_degrees=3.0, yaw_ratio=0.05)
    monkeypatch.setattr(pipeline._preprocessor, "detect_and_align", lambda image: det)
    q = pipeline.capture_quality(np.zeros((200, 200, 3), np.uint8))
    assert q["verdict"] == "BLURRY" and q["sharpness"] == 12.0 and set(q) == set(QUALITY)

    def no_face(image):
        raise ValueError("no face")

    monkeypatch.setattr(pipeline._preprocessor, "detect_and_align", no_face)
    assert pipeline.capture_quality(np.zeros((200, 200, 3), np.uint8)) == {"verdict": "NO_FACE"}
