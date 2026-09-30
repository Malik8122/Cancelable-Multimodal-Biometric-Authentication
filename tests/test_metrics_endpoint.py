"""Offline tests for GET /metrics/{modality}.

Reads the real, already-committed `evaluation/results/*_metrics.csv` files -
no fixtures fabricate metrics here, since the whole point of this endpoint is
to surface real numbers (or their real absence) verbatim.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _reset_caches():
    from backend.config import get_settings
    from backend.database.session import get_engine, get_session_factory

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test_backend.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    _reset_caches()

    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client

    _reset_caches()


def test_voice_metrics_are_real_and_complete(client):
    response = client.get("/metrics/voice")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["metrics"]["accuracy"] == pytest.approx(0.9771185086551265)
    assert body["metrics"]["eer"] == pytest.approx(0.022889825855019627)
    for field in ("precision", "recall", "far", "frr"):
        assert field in body["metrics"]


def test_face_metrics_report_only_what_was_actually_computed(client):
    """Face only ever had accuracy/EER/AUC computed (docs/PROJECT_REPORT.md) -
    precision/recall/F1/FAR/FRR must be absent, not zero-filled."""
    response = client.get("/metrics/face")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["metrics"]["accuracy"] == pytest.approx(0.990)
    assert body["metrics"]["eer"] == pytest.approx(0.010)
    assert body["metrics"]["auc"] == pytest.approx(0.999)
    for never_computed_field in ("precision", "recall", "f1", "far", "frr"):
        assert never_computed_field not in body["metrics"]


def test_a_modality_without_a_metrics_csv_is_unavailable_not_fabricated(client, monkeypatch, tmp_path):
    """No `<modality>_metrics.csv` - the endpoint must say so plainly rather than inventing numbers."""
    import backend.api.metrics as metrics_module

    monkeypatch.setattr(metrics_module, "_RESULTS_DIR", tmp_path)
    response = client.get("/metrics/face")
    assert response.status_code == 200
    body = response.json()
    assert body["modality"] == "face"
    assert body["available"] is False
    assert body["metrics"] == {}


def test_invalid_modality_is_rejected(client):
    for modality in ("retina", "fingerprint", "iris"):  # iris and fingerprint were removed from the system
        response = client.get(f"/metrics/{modality}")
        assert response.status_code == 422, modality


def test_calibrated_fields_are_merged_under_distinct_keys_when_present(client, monkeypatch, tmp_path):
    """Bug-7's "expose calibrated FAR/FRR": when a real
    evaluation/threshold_calibration.py report exists for a modality, its
    numbers appear under `calibrated_*` keys - distinct from `far`/`frr`
    above, which are a different score space (raw embeddings, not protected
    templates) and must never be silently conflated with these."""
    import backend.api.metrics as metrics_module

    monkeypatch.setattr(metrics_module, "_RESULTS_DIR", tmp_path)
    (tmp_path / "voice_threshold.json").write_text(
        '{"threshold": 0.87, "eer": 0.05, "far": 0.04, "frr": 0.06, "auc": 0.95}', encoding="utf-8"
    )

    response = client.get("/metrics/voice")
    assert response.status_code == 200
    body = response.json()
    assert body["calibrated"] is True
    assert body["metrics"]["calibrated_threshold"] == pytest.approx(0.87)
    assert body["metrics"]["calibrated_far"] == pytest.approx(0.04)
    assert body["metrics"]["calibrated_frr"] == pytest.approx(0.06)
    # available/available-metrics reflect the (now-missing, since _RESULTS_DIR
    # was swapped to an empty tmp_path) raw-embedding CSV, independently.
    assert body["available"] is False
