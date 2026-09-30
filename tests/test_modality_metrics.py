"""Face = estimated cosine similarity (>= 0.80), voice = estimated Euclidean distance (<= 0.75), fusion on one
higher-is-better scale, and the cancelable template compared by Hamming distance only.

The face / voice values are CALIBRATED ESTIMATES derived from the 256-bit template Hamming similarity
(template_protection/metric_estimation.py): raw embeddings are never stored, so the exact metrics cannot be
computed. The rule tests use an exactly-invertible stand-in curve so boundaries are exact; the end-to-end tests use
the real calibration file.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.config import Settings
from backend.services import modality_metrics
from template_protection import metric_estimation
from template_protection.matcher import compare
from template_protection.metric_estimation import (
    CALIBRATION_PATH,
    MetricCurve,
    cosine_from_euclidean,
    euclidean_from_cosine,
)
from tests.test_flexible_auth import (  # noqa: F401 - `client` is a fixture
    FACE,
    OTHER_FACE,
    OTHER_VOICE,
    VOICE,
    _authenticate,
    _enroll_user,
    client,
)


def _settings(**overrides) -> Settings:
    return Settings(master_secret="unit-test-master-secret-not-for-production", **overrides)


@pytest.fixture
def linear_curve(monkeypatch):
    """Stand-in curve with estimated cosine == 2 * hamming - 1 exactly, for exact boundary tests."""
    grid = np.linspace(-1.0, 1.0, 201)
    curve = MetricCurve("any", 256, grid, (grid + 1.0) / 2.0, np.full_like(grid, 0.03))
    monkeypatch.setattr(modality_metrics, "get_curve", lambda modality, bits: curve)
    return curve


def _hamming_for(cosine: float) -> float:
    return (cosine + 1.0) / 2.0


# ----------------------------------------------------------------------------- conversions


def test_euclidean_and_cosine_are_two_views_of_the_same_angle_for_unit_vectors():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.standard_normal(192), rng.standard_normal(192)
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        cosine = float(a @ b)
        assert euclidean_from_cosine(cosine) == pytest.approx(float(np.linalg.norm(a - b)))
        assert cosine_from_euclidean(euclidean_from_cosine(cosine)) == pytest.approx(cosine)
    assert euclidean_from_cosine(1.0) == 0.0 and euclidean_from_cosine(-1.0) == pytest.approx(2.0)


# ----------------------------------------------------------------------------- face: estimated cosine, >= 0.80


@pytest.mark.parametrize("cosine, expected", [(0.95, True), (0.80, True), (0.7999, False), (0.30, False)])
def test_face_passes_at_or_above_0_80_estimated_cosine(linear_curve, cosine, expected):
    decision = modality_metrics.decide("face", _hamming_for(cosine), 256, _settings())
    assert decision.metric == modality_metrics.COSINE_ESTIMATE and decision.higher_is_better is True
    assert decision.threshold == 0.80
    assert decision.value == pytest.approx(cosine)
    assert decision.matched is expected


def test_face_threshold_is_configurable(linear_curve):
    strict = modality_metrics.decide("face", _hamming_for(0.85), 256, _settings(face_cosine_threshold=0.90))
    assert strict.threshold == 0.90 and strict.matched is False


# ----------------------------------------------------------------------------- voice: estimated distance, <= 0.90


@pytest.mark.parametrize("distance, expected", [(0.30, True), (0.90, True), (0.9001, False), (1.20, False)])
def test_voice_passes_at_or_below_0_90_estimated_euclidean_distance(linear_curve, distance, expected):
    decision = modality_metrics.decide("voice", _hamming_for(cosine_from_euclidean(distance)), 256, _settings())
    assert decision.metric == modality_metrics.EUCLIDEAN_ESTIMATE and decision.higher_is_better is False
    assert decision.threshold == 0.90
    assert decision.value == pytest.approx(distance, abs=1e-9)
    assert decision.matched is expected


def test_a_larger_voice_distance_is_worse_not_better(linear_curve):
    near = modality_metrics.decide("voice", _hamming_for(0.9), 256, _settings())
    far = modality_metrics.decide("voice", _hamming_for(0.5), 256, _settings())
    assert near.value < far.value  # distance grows as the voices differ...
    assert near.fusion_score > far.fusion_score  # ...while the fusion score (higher = better) falls


def test_voice_enters_fusion_as_a_similarity_never_as_a_raw_distance(linear_curve):
    decision = modality_metrics.decide("voice", _hamming_for(0.9), 256, _settings())
    assert decision.fusion_score == pytest.approx(1 - decision.value**2 / 2) == pytest.approx(0.9)
    assert decision.fusion_threshold == pytest.approx(1 - 0.90**2 / 2)  # 0.595


def test_voice_threshold_is_configurable(linear_curve):
    loose = modality_metrics.decide("voice", _hamming_for(cosine_from_euclidean(0.9)), 256, _settings(voice_euclidean_threshold=1.0))
    assert loose.threshold == 1.0 and loose.matched is True


def test_without_a_calibration_curve_face_and_voice_fall_back_to_the_hamming_rule(monkeypatch):
    monkeypatch.setattr(modality_metrics, "get_curve", lambda modality, bits: None)
    decision = modality_metrics.decide("voice", 0.95, 128, _settings())
    assert decision.metric == modality_metrics.HAMMING and decision.value == 0.95


# ----------------------------------------------------------------------------- Hamming: the template comparison


def test_identical_templates_have_hamming_distance_zero():
    bits = np.random.default_rng(3).integers(0, 2, 256, dtype=np.uint8)
    assert compare(bits, bits.copy(), metric="hamming") == 1.0
    assert int(np.count_nonzero(bits != bits)) == 0


def test_differing_templates_have_the_expected_hamming_distance():
    a = np.zeros(256, dtype=np.uint8)
    b = a.copy()
    b[:32] = 1  # exactly 32 differing bits
    assert int(np.count_nonzero(a != b)) == 32
    assert compare(a, b, metric="hamming") == pytest.approx(1 - 32 / 256)


def test_the_template_bit_length_is_still_256():
    assert _settings().template_bits == 256


# ----------------------------------------------------------------------------- the real calibration file


def test_the_calibration_file_is_256_bit_monotone_and_labelled_as_not_real_biometric_data():
    report = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    assert report["template_bits"] == 256
    assert "Not measured on real biometric data" in report["source"]
    for modality in ("face", "voice"):
        entry = report["modalities"][modality]
        grid = np.asarray(entry["cosine"])
        fitted = metric_estimation.fitted_hamming_similarity(grid, entry["coefficients"])
        assert np.all(np.diff(fitted) > 0), modality  # invertible
        assert entry["max_fit_residual"] < 0.01, modality  # the model matches the sampled means
        # Around the decision thresholds (cosine >= 0.7) one estimate is within ~0.07 (1 sd); wider for impostors.
        assert max(s for c, s in zip(grid, entry["cosine_estimate_std"]) if c >= 0.7) < 0.07, modality


def test_the_real_curve_recovers_a_known_cosine_within_its_stated_uncertainty():
    """Synthetic pairs at a known cosine, through the real BioHash: the mean estimate lands near the truth."""
    from template_protection.biohash import generate_template
    from template_protection.hkdf_keys import derive_key

    metric_estimation.clear_cache()
    rng = np.random.default_rng(42)
    for modality, dim in (("face", 512), ("voice", 192)):
        curve = metric_estimation.get_curve(modality, 256)
        estimates = []
        for trial in range(40):
            a = rng.standard_normal(dim)
            a /= np.linalg.norm(a)
            r = rng.standard_normal(dim)
            r -= (r @ a) * a
            r /= np.linalg.norm(r)
            b = 0.8 * a + 0.6 * r  # cosine exactly 0.8
            key = derive_key("independent-test-secret", application_id="t", user_id=f"{trial}", modality=modality, key_version=1)
            estimates.append(curve.estimate_cosine(compare(generate_template(a, key, 256), generate_template(b, key, 256))))
        assert np.mean(estimates) == pytest.approx(0.8, abs=0.02), modality


# ----------------------------------------------------------------------------- end to end (stubbed embedders)


def test_face_and_voice_both_verify_and_report_their_own_metrics(client):
    _enroll_user(client, "u", face=FACE, voice=VOICE)
    body = _authenticate(client, "u", "defence_research_lab", face=FACE, voice=VOICE).json()
    assert body["authenticated"] is True
    face, voice = body["results"]["face"], body["results"]["voice"]
    assert face["metric"] == "cosine_estimate" and face["metric_threshold"] == 0.80 and face["metric_higher_is_better"] is True
    assert voice["metric"] == "euclidean_estimate" and voice["metric_threshold"] == 0.90 and voice["metric_higher_is_better"] is False
    # Identical samples -> identical templates: Hamming distance 0 of 256 bits, estimated cosine 1, distance 0.
    for r in (face, voice):
        assert r["hamming_distance_bits"] == 0 and r["template_bits"] == 256 and r["hamming_similarity"] == 1.0
    assert face["metric_value"] == 1.0 and voice["metric_value"] == 0.0
    # Fusion: the mean of the higher-is-better scores (voice converted), not of a similarity and a distance.
    assert body["fusion_similarity"] == pytest.approx((face["score"] + voice["score"]) / 2) == pytest.approx(1.0)


def test_a_wrong_face_denies_even_with_a_matching_voice(client):
    _enroll_user(client, "u", face=FACE, voice=VOICE)
    body = _authenticate(client, "u", "defence_research_lab", face=OTHER_FACE, voice=VOICE).json()
    assert body["authenticated"] is False and body["matched_modalities"] == ["voice"]
    face = body["results"]["face"]
    assert face["authenticated"] is False and face["metric_value"] < 0.80 and face["hamming_distance_bits"] > 0


def test_a_wrong_voice_denies_even_with_a_matching_face(client):
    _enroll_user(client, "u", face=FACE, voice=VOICE)
    body = _authenticate(client, "u", "defence_research_lab", face=FACE, voice=OTHER_VOICE).json()
    assert body["authenticated"] is False and body["matched_modalities"] == ["face"]
    voice = body["results"]["voice"]
    assert voice["authenticated"] is False and voice["metric_value"] > 0.90


def test_production_responses_carry_no_per_modality_metrics(client, monkeypatch):
    from backend.config import get_settings

    _enroll_user(client, "u", face=FACE, voice=VOICE)
    monkeypatch.setenv("DEBUG_SCORES", "false")
    get_settings.cache_clear()
    body = _authenticate(client, "u", "defence_research_lab", face=FACE, voice=VOICE).json()
    assert body["authenticated"] is True and "results" not in body and "fusion_diagnostics" not in body
