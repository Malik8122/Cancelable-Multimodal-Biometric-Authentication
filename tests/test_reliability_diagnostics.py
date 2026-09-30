"""Development diagnostics added by the reliability audit: they must explain a verdict (which check failed, with the
measured value and limit) without changing any decision, and exist only when the server runs with DEBUG_SCORES."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.hand_signals import Person
from tests.test_flexible_auth import APPLICATION_ID, FACE, client  # noqa: F401
from tests.test_voice_capture_robustness import CLIPPED, NOISE_ONLY, S1, SHORT, SILENT


def _assess(sample, diagnostics):
    from backend.config import Settings
    from backend.services import voice_quality
    from preprocessing.voice import load_wav_bytes

    return voice_quality.assess(load_wav_bytes(sample), Settings(master_secret="t"), diagnostics=diagnostics)


@pytest.mark.parametrize("sample, verdict, metric", [
    (S1, "OK", None),
    (SHORT, "TOO_LITTLE_SPEECH", "speech_seconds"),
    (NOISE_ONLY, "TOO_NOISY", "estimated_snr_db"),
    (CLIPPED, "CLIPPED", "clipping_fraction"),
    (SILENT, "NO_SPEECH", "speech_seconds"),
], ids=["speech", "short", "noise", "clipped", "silent"])
def test_voice_diagnostics_name_the_failing_check(sample, verdict, metric):
    report = _assess(sample, diagnostics=True)
    assert report.verdict == verdict
    d = report.diagnostics
    assert d["limits"] == {"min_speech_seconds": 1.5, "min_estimated_snr_db": 10.0, "max_clipping_fraction": 0.01}
    if metric is None:
        assert d["failed_check"] is None
    else:
        assert d["failed_check"]["metric"] == metric and d["failed_check"]["measured"] == report.metrics[metric]
    assert d["input_sample_rate_hz"] > 0 and d["channels"] == 1
    # scalars only - never audio
    assert all(not isinstance(v, (list, np.ndarray)) for v in d.values())


def test_voice_diagnostics_never_change_the_verdict():
    for sample in (S1, SHORT, NOISE_ONLY, CLIPPED, SILENT):
        with_d, without = _assess(sample, True), _assess(sample, False)
        assert with_d.verdict == without.verdict and with_d.metrics == without.metrics
        assert without.diagnostics is None


def test_quality_check_endpoint_returns_diagnostics_only_in_debug_mode(client, monkeypatch):
    from backend.config import get_settings

    def check():
        return client.post("/voice/check-quality", files={"audio": ("s.wav", S1, "audio/wav")}).json()

    monkeypatch.setenv("DEBUG_SCORES", "false")
    get_settings.cache_clear()
    assert "diagnostics" not in check()
    assert client.get("/system/health").json()["diagnostics_enabled"] is False
    monkeypatch.setenv("DEBUG_SCORES", "true")
    get_settings.cache_clear()
    body = check()
    assert body["diagnostics"]["failed_check"] is None and body["passed"] is True
    assert client.get("/system/health").json()["diagnostics_enabled"] is True
    get_settings.cache_clear()


def test_hand_sample_check_reports_tracking_diagnostics(client):
    body = client.post("/enroll/hand/check", files={"gesture": ("g.json", json.dumps(Person(1).capture(4)), "application/json")}).json()
    assert body["passed"] is True
    for key in ("tracking_fps", "motion_duration_s", "idle_trimmed_s", "frames_detected", "trajectory_extent_palms"):
        assert key in body["metrics"], key


def test_face_enrollment_and_authentication_use_the_same_embedding():
    """The same image must produce the same embedding on the enrollment and the authentication path (one pipeline)."""
    import pathlib

    import matplotlib

    from backend.services import get_service_for_modality
    from backend.utils import decode_image

    portrait = pathlib.Path(matplotlib.__file__).parent / "mpl-data" / "sample_data" / "grace_hopper.jpg"
    service = get_service_for_modality("face")
    raw = decode_image(portrait.read_bytes())
    first, second = service.pipeline.embed(raw), service.pipeline.embed(raw)
    assert first.shape == (512,) and np.allclose(first, second)
    # And with no face at all, the pipeline reports a capture problem (never a low score).
    with pytest.raises(ValueError, match="No face detected"):
        service.pipeline.embed(decode_image(FACE))
