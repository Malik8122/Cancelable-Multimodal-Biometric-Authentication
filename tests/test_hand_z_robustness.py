"""Z gesture robustness: the normalized representation + rotation search + DTW absorb natural variation (size, position,
speed, frame rate, a few degrees of rotation, jitter, idle time) while a different person or a different movement is
still rejected. SYNTHETIC landmark captures (tests/hand_signals.py) - software tests, not evidence of accuracy.

The service-level distance used here is exactly the one authentication computes (rotation search, DTW per enrolled
sample, median), on plaintext sequences; `test_keyed_matching_equals_plaintext_matching` shows the keyed template gives
the same numbers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.config import get_settings
from backend.services.hand_service import (
    HandGestureService,
    enrollment_analysis,
    gesture_distance,
    rotation_angles,
)
from preprocessing.hand_gesture import (
    CAPTURE_QUALITY_FAILURE,
    INSUFFICIENT_FRAMES,
    INVALID_TRAJECTORY,
    HandCaptureRejected,
    parse_capture,
    process_capture,
    rotate_features,
    weighted,
    z_structure,
)
from tests.hand_signals import Person, Z_WAYPOINTS, at_frame_rate, polyline_capture, transformed, with_idle

ALICE, BOB = Person(11), Person(12)
SETTINGS = get_settings()
THRESHOLD = SETTINGS.hand_gesture_dtw_threshold
BAND = SETTINGS.hand_gesture_dtw_band
ANGLES = rotation_angles(SETTINGS.hand_gesture_rotation_tolerance_deg)


def _seq(capture: dict) -> np.ndarray:
    return weighted(process_capture(parse_capture(json.dumps(capture))).features)


def _distance(candidate: np.ndarray, reference: np.ndarray, angles=ANGLES) -> float:
    return gesture_distance([rotate_features(candidate, a) for a in angles], reference, BAND)


def _match(candidate: np.ndarray, enrolled: list[np.ndarray]) -> float:
    return float(np.median([_distance(candidate, e) for e in enrolled]))


@pytest.fixture(scope="module")
def alice_enrolled() -> list[np.ndarray]:
    return [_seq(ALICE.capture(s)) for s in (1, 2, 3)]


# ----------------------------------------------------------------------------- invariances (same execution, changed)


@pytest.mark.parametrize("change", [
    {"dx": 0.15, "dy": -0.1},             # Z drawn further left / higher in the frame
    {"dx": -0.12, "dy": 0.08},
    {"scale": 0.6},                       # a smaller Z (or further from the camera)
    {"scale": 1.35},                      # a larger Z
])
def test_position_and_size_do_not_change_the_representation(change):
    capture = ALICE.capture(40)
    a, b = _seq(capture), _seq(transformed(capture, **change))
    assert _distance(a, b, [0.0]) < 0.02


@pytest.mark.parametrize("degrees", [-10, -5, 5, 10])
def test_small_rotations_are_absorbed_by_the_rotation_search(degrees):
    capture = ALICE.capture(41)
    a, b = _seq(capture), _seq(transformed(capture, angle_deg=degrees))
    assert _distance(b, a, [0.0]) > _distance(b, a)       # the search is what absorbs it
    assert _distance(b, a) < 0.05


def test_rotation_tolerance_is_bounded_not_full_invariance():
    """A Z rotated far beyond the tolerance is NOT aligned back: orientation stays part of the behaviour."""
    a = _seq(ALICE.capture(42))
    rotated = rotate_features(a, 35.0)
    assert _distance(rotated, a) > 2 * _distance(rotate_features(a, 10.0), a)
    assert _distance(rotated, a) > THRESHOLD


def test_frame_rate_variation_gives_a_comparable_sequence():
    """The same execution tracked at 30 fps and at 15 fps (every 2nd frame) - different frame rates, one representation."""
    capture = ALICE.capture(43)
    assert _distance(_seq(at_frame_rate(capture, 2)), _seq(capture)) < 0.08   # typical genuine-vs-genuine: ~0.15


def test_landmark_jitter_is_smoothed():
    capture = ALICE.capture(44)
    noisy = json.loads(json.dumps(capture))
    rng = np.random.default_rng(0)
    for frame in noisy["frames"]:
        lm = np.asarray(frame["landmarks"])
        lm[:, :2] += rng.normal(0, 1.5 / 480, (21, 2))     # ~1.5 px extra jitter on every landmark
        frame["landmarks"] = lm.tolist()
    assert _distance(_seq(noisy), _seq(capture)) < 0.1


def test_idle_padding_is_removed_without_changing_the_match(alice_enrolled):
    capture = ALICE.capture(45)
    padded = with_idle(capture, 1.5, 1.0, seed=4)
    processed = process_capture(parse_capture(json.dumps(padded)))
    assert processed.metadata["idle_trimmed_s"] > 2.0
    assert abs(_match(weighted(processed.features), alice_enrolled) - _match(_seq(capture), alice_enrolled)) < 0.03


def test_pauses_inside_the_z_are_kept_not_trimmed():
    """Slowing down / pausing at the corners lies between the first and last movement: only the ends are trimmed."""
    person = Person(21)
    person.pauses = np.array([0.35, 0.35])
    processed = process_capture(parse_capture(json.dumps(person.capture(1, duration_s=2.4))))
    assert processed.metadata["idle_trimmed_s"] < 0.2


# ----------------------------------------------------------------------------- natural variation of a genuine user


def test_a_slow_and_a_fast_genuine_z_both_match(alice_enrolled):
    slow, fast = _seq(ALICE.capture(50, duration_s=3.1)), _seq(ALICE.capture(51, duration_s=1.6))
    assert _match(slow, alice_enrolled) <= THRESHOLD
    assert _match(fast, alice_enrolled) <= THRESHOLD


def test_genuine_z_at_12_fps_matches(alice_enrolled):
    assert _match(_seq(ALICE.capture(52, fps=12)), alice_enrolled) <= THRESHOLD


def test_genuine_z_moved_scaled_and_slightly_rotated_matches(alice_enrolled):
    changed = transformed(ALICE.capture(53), dx=-0.08, dy=0.05, scale=0.75, angle_deg=-8)
    assert _match(_seq(changed), alice_enrolled) <= THRESHOLD


def test_genuine_and_impostor_distributions_separate():
    """Several synthetic users: every genuine probe below the threshold, (nearly) every other user's Z above it."""
    people = [Person(300 + k) for k in range(8)]
    enrolled = [[_seq(p.capture(s)) for s in range(3)] for p in people]
    probes = [_seq(p.capture(1000 + k)) for k, p in enumerate(people)]
    genuine = [_match(probes[k], enrolled[k]) for k in range(len(people))]
    impostor = [_match(probes[c], enrolled[t]) for c in range(len(people)) for t in range(len(people)) if c != t]
    assert max(genuine) <= THRESHOLD
    assert np.mean(np.array(impostor) <= THRESHOLD) <= 0.05
    assert np.median(impostor) > 1.5 * np.median(genuine)


# ----------------------------------------------------------------------------- invalid captures (never a mismatch)


@pytest.mark.parametrize("waypoints", [
    Z_WAYPOINTS[:3],                                          # incomplete Z: the bottom stroke is missing
    Z_WAYPOINTS[:2],                                          # only the top stroke
    Z_WAYPOINTS[::-1],                                        # drawn backwards
    [(440, 150), (200, 150), (440, 330), (200, 330)],         # mirrored Z
    [(200, 330), (200, 150), (440, 330), (440, 150)],         # an N
    [(320 + 120 * np.cos(a), 240 + 100 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 40)],   # a circle
])
def test_a_different_or_incomplete_movement_is_an_invalid_capture(waypoints):
    with pytest.raises(HandCaptureRejected) as error:
        process_capture(parse_capture(json.dumps(polyline_capture(waypoints))))
    assert error.value.code == INVALID_TRAJECTORY and not error.value.is_capture_error


def test_a_clean_z_passes_the_structure_check():
    processed = process_capture(parse_capture(json.dumps(polyline_capture(Z_WAYPOINTS))))
    assert processed.metadata["z_validity"] == "PASS"


def test_the_mirrored_flag_decides_left_and_right():
    """The same raw landmarks, declared as an un-mirrored view, are a mirrored Z to the user - rejected."""
    capture = ALICE.capture(60)
    assert process_capture(parse_capture(json.dumps(capture))).metadata["z_validity"] == "PASS"
    with pytest.raises(HandCaptureRejected) as error:
        process_capture(parse_capture(json.dumps({**capture, "mirrored": False})))
    assert error.value.code == INVALID_TRAJECTORY


def test_z_structure_on_a_bare_trajectory():
    t = np.linspace(0, 1, 60)[:, None]
    top, diagonal, bottom = [0, 0] + t[:20] * [1, 0], [1, 0] + t[:20] * [-1, 1], [0, 1] + t[:20] * [1, 0]
    assert z_structure(np.vstack([top, diagonal, bottom])).valid
    assert not z_structure(np.vstack([top, diagonal])).valid
    assert not z_structure(np.zeros((10, 2))).valid


def test_an_extremely_short_gesture_is_rejected():
    with pytest.raises(HandCaptureRejected) as error:
        process_capture(parse_capture(json.dumps(ALICE.capture(61, duration_s=0.35))))
    assert error.value.code in (INSUFFICIENT_FRAMES, CAPTURE_QUALITY_FAILURE)


def test_low_tracking_rate_is_a_capture_quality_failure():
    with pytest.raises(HandCaptureRejected) as error:
        process_capture(parse_capture(json.dumps(ALICE.capture(62, duration_s=3.0, fps=8))))
    assert error.value.code == CAPTURE_QUALITY_FAILURE and "frames per second" in str(error.value)


def test_twelve_fps_is_accepted():
    assert process_capture(parse_capture(json.dumps(ALICE.capture(63, fps=12)))).metadata["tracking_fps"] >= 10


# ----------------------------------------------------------------------------- enrollment robustness


def test_rotation_angles():
    assert rotation_angles(12) == [0.0, 6.0, -6.0, 12.0, -12.0]
    assert rotation_angles(0) == [0.0]
    assert rotation_angles(8) == [0.0, 6.0, -6.0, 8.0, -8.0]


def test_a_genuine_enrollment_has_no_outlier_and_a_medoid():
    analysis = enrollment_analysis([_seq(ALICE.capture(s)) for s in (70, 71, 72)], BAND, ANGLES, THRESHOLD)
    assert analysis["outlier"] is None and analysis["medoid"] in (0, 1, 2) and len(analysis["pairwise"]) == 3


def test_one_foreign_sample_is_flagged_and_nothing_is_stored(db_session):
    service = HandGestureService(SETTINGS)
    captures = [parse_capture(json.dumps(c)) for c in (ALICE.capture(80), BOB.capture(81), ALICE.capture(82))]
    with pytest.raises(HandCaptureRejected) as error:
        service.enroll_attempts(db_session, captures, "alice", "app")
    assert error.value.code == CAPTURE_QUALITY_FAILURE and error.value.attempt == 2
    assert "sample 2" in str(error.value)
    from backend.database import crud

    assert crud.get_set_rows(db_session, "alice", "app") == []


# ----------------------------------------------------------------------------- keyed template consistency


def test_keyed_matching_equals_plaintext_matching(db_session):
    """Enrollment: normalized -> keyed transform -> stored. Authentication: normalized -> rotation -> SAME key ->
    DTW. The distances equal the plaintext ones: same key, dimensionality, feature order, no double transform."""
    service = HandGestureService(SETTINGS)
    enrolled = [ALICE.capture(s) for s in (90, 91, 92)]
    service.enroll_attempts(db_session, [parse_capture(json.dumps(c)) for c in enrolled], "alice", "app")
    probe = ALICE.capture(93)
    result = service.authenticate(db_session, parse_capture(json.dumps(probe)), "alice", "app")
    plain = [_distance(_seq(probe), _seq(c)) for c in enrolled]
    np.testing.assert_allclose(result.diagnostics["dtw_distances"], plain, atol=1e-4)
    assert result.metric_value == pytest.approx(float(np.median(plain)), abs=1e-4)


def test_a_retired_infinity_template_means_re_enroll_not_a_crash(db_session):
    """A user enrolled with the removed infinity gesture (format hand_gesture_v1) is reported as NOT enrolled for the
    hand, so authentication asks for enrollment instead of comparing incompatible representations."""
    from backend.services.enrollment import get_user_enrollment_status

    service = HandGestureService(SETTINGS)
    rows, _ = service.enroll_attempts(db_session, [parse_capture(json.dumps(ALICE.capture(s))) for s in range(3)],
                                      "legacy", "app")
    assert get_user_enrollment_status(db_session, "legacy", "app").modalities["hand"] is True
    for row in rows:
        row.template_format, row.gesture_type = "hand_gesture_v1", "infinity"
    db_session.commit()
    assert get_user_enrollment_status(db_session, "legacy", "app").modalities["hand"] is False


def test_authentication_reports_best_rotation_and_feature_shares(db_session):
    """Development diagnostics: the rotation behind each per-sample distance, and (DEBUG_SCORES) each feature group's
    share of it - the shares sum to 1 and are computed on the same alignment as the reported distance."""
    service = HandGestureService(SETTINGS)
    service.enroll_attempts(db_session, [parse_capture(json.dumps(ALICE.capture(s))) for s in (95, 96, 97)], "alice", "app")
    probe = transformed(ALICE.capture(98), angle_deg=10)
    result = service.authenticate(db_session, parse_capture(json.dumps(probe)), "alice", "app")
    d = result.diagnostics
    assert len(d["dtw_best_rotation_deg"]) == 3 and set(d["dtw_best_rotation_deg"]) <= set(ANGLES)
    if SETTINGS.debug_scores:
        assert len(d["feature_group_shares"]) == 3
        for shares in d["feature_group_shares"]:
            assert set(shares) == {"trajectory", "direction", "speed", "acceleration", "orientation", "fingers"}
            assert abs(sum(shares.values()) - 1.0) < 1e-3
