"""HTTP-level tests for the TEMPORARY face-debug diagnostics (alignment investigation) - proves
they're actually wired into POST /enroll and POST /authenticate/fusion, gated correctly on
DEBUG_SCORES, and provably inert with respect to the real authentication decision.

Reuses tests/test_face_enrollment.py's fixtures (client, FIVE poses, _enroll_poses, _authenticate)
rather than redefining them.
"""

from __future__ import annotations

import re

import pytest

from tests.test_face_enrollment import APPLICATION_ID, FIVE, _enroll_poses  # noqa: F401  (client is a fixture)
from tests.test_flexible_auth import BUILDING, _authenticate, client  # noqa: F401

FLOAT_TOKEN = re.compile(r"-?\d+\.\d+")


def test_debug_scores_true_logs_face_debug_on_enrollment(client, caplog):
    with caplog.at_level("INFO", logger="backend.face_debug"):
        assert _enroll_poses(client, **FIVE).status_code == 200

    messages = [r.getMessage() for r in caplog.records if r.name == "backend.face_debug"]
    assert any("stage=enrollment" in m for m in messages)
    enrollment_message = next(m for m in messages if "stage=enrollment" in m)
    assert "enrollment_pose_similarity" in enrollment_message
    assert "enrollment_centroid_similarity" in enrollment_message
    assert "poses=5" in enrollment_message


def test_debug_scores_true_logs_face_debug_on_authentication(client, caplog):
    assert _enroll_poses(client, **FIVE).status_code == 200

    with caplog.at_level("INFO", logger="backend.face_debug"):
        response = _authenticate(client, "u", BUILDING, face=FIVE["front"])
    assert response.status_code == 200

    messages = [r.getMessage() for r in caplog.records if r.name == "backend.face_debug"]
    assert any("stage=authentication" in m for m in messages)
    auth_message = next(m for m in messages if "stage=authentication" in m)
    assert "protected_hamming_similarity" in auth_message
    assert "live_vs_centroid_cosine: unavailable" in auth_message
    # The logged Hamming value must be the exact same one the real response carries (the log line
    # rounds to 4 decimal places, so compare with a tolerance matching that rounding).
    logged_value = float(re.search(r"protected_hamming_similarity:\s*(-?\d+\.\d+)", auth_message).group(1))
    assert logged_value == pytest.approx(response.json()["results"]["face"]["hamming_similarity"], abs=5e-5)


def test_debug_scores_false_produces_no_face_debug_logging(client, monkeypatch, caplog):
    monkeypatch.setenv("DEBUG_SCORES", "false")
    from backend.config import get_settings

    get_settings.cache_clear()
    try:
        with caplog.at_level("INFO", logger="backend.face_debug"):
            assert _enroll_poses(client, **FIVE).status_code == 200
            response = _authenticate(client, "u", BUILDING, face=FIVE["front"])
            assert response.status_code == 200

        messages = [r.getMessage() for r in caplog.records if r.name == "backend.face_debug"]
        assert messages == []
        # And the debug-only per-modality `results` field is likewise absent in production mode.
        assert "results" not in response.json()
    finally:
        monkeypatch.setenv("DEBUG_SCORES", "true")
        get_settings.cache_clear()


def test_authentication_result_is_identical_whether_or_not_diagnostics_ran(client, monkeypatch):
    """The diagnostic log call happens strictly after `authenticated`/`score` are already decided
    (backend/services/base_service.py::authenticate_embedding) - proves it end-to-end: the exact
    same live capture against the exact same enrollment must produce the exact same decision and
    score regardless of DEBUG_SCORES."""
    from backend.config import get_settings

    assert _enroll_poses(client, **FIVE).status_code == 200

    monkeypatch.setenv("DEBUG_SCORES", "true")
    get_settings.cache_clear()
    with_diagnostics = _authenticate(client, "u", BUILDING, face=FIVE["front"]).json()

    monkeypatch.setenv("DEBUG_SCORES", "false")
    get_settings.cache_clear()
    without_diagnostics = _authenticate(client, "u", BUILDING, face=FIVE["front"]).json()

    monkeypatch.setenv("DEBUG_SCORES", "true")
    get_settings.cache_clear()

    assert with_diagnostics["authenticated"] == without_diagnostics["authenticated"]
    assert with_diagnostics["fusion_similarity"] == pytest.approx(without_diagnostics["fusion_similarity"])


def test_face_debug_logs_never_contain_vector_sized_data(client, caplog):
    with caplog.at_level("INFO", logger="backend.face_debug"):
        assert _enroll_poses(client, **FIVE).status_code == 200
        assert _authenticate(client, "u", BUILDING, face=FIVE["front"]).status_code == 200

    for record in caplog.records:
        if record.name != "backend.face_debug":
            continue
        message = record.getMessage()
        float_count = len(FLOAT_TOKEN.findall(message))
        assert float_count <= 10, f"possible vector data in log: {message!r}"
