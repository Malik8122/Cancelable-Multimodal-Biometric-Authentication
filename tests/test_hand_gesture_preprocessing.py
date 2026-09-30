"""Hand gesture preprocessing: capture parsing / validation (the MediaPipe landmark input), normalization, temporal
representation. SYNTHETIC landmark sequences (tests/hand_signals.py) - software tests, not biometric trials."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from preprocessing.hand_gesture import (
    CAPTURE_QUALITY_FAILURE,
    FEATURE_DIM,
    INSUFFICIENT_FRAMES,
    INVALID_TRAJECTORY,
    MALFORMED_CAPTURE,
    NO_HAND_DETECTED,
    RESAMPLED_LENGTH,
    HandCaptureRejected,
    isotropic,
    normalize_shape,
    parse_capture,
    process_capture,
    weighted,
)
from template_protection.dtw import dtw_distance
from tests.hand_signals import Person, no_hand_capture, still_hand_capture


def _process(payload):
    return process_capture(parse_capture(json.dumps(payload)))


def meta_z(processed) -> bool:
    return processed.metadata["z_validity"] == "PASS" and 0.7 <= processed.metadata["z_explained"] <= 1.0


def _reject_code(payload) -> str:
    with pytest.raises(HandCaptureRejected) as error:
        _process(payload)
    return error.value.code


# ----------------------------------------------------------------------------- MediaPipe landmark input


def test_a_valid_hand_capture_is_accepted_with_metadata_only():
    processed = _process(Person(1).capture(1))
    assert processed.features.shape == (RESAMPLED_LENGTH, FEATURE_DIM)
    assert processed.gesture_type == "z" and meta_z(processed)
    meta = processed.metadata
    assert meta["frames_detected"] == meta["frames_total"] and meta["detection_ratio"] == 1.0
    assert 1.0 < meta["duration_s"] < 3.0 and meta["handedness"] == "Right"
    assert meta["client_mediapipe_ms_mean"] == 11.5
    assert not any(isinstance(v, (list, np.ndarray)) for v in meta.values())  # no landmarks in metadata


def test_no_hand_is_a_capture_error_not_a_mismatch():
    assert _reject_code(no_hand_capture()) == NO_HAND_DETECTED
    with pytest.raises(HandCaptureRejected) as error:
        _process(no_hand_capture())
    assert error.value.is_capture_error


def test_insufficient_frames():
    assert _reject_code(Person(1).capture(1, duration_s=0.3)) == INSUFFICIENT_FRAMES


def test_hand_lost_too_often_is_a_quality_failure():
    payload = Person(1).capture(1, duration_s=2.5, dropout=0.6)
    assert _reject_code(payload) in (CAPTURE_QUALITY_FAILURE, INSUFFICIENT_FRAMES)


def test_occasional_dropped_frames_are_interpolated():
    processed = _process(Person(1).capture(1, dropout=0.15))
    assert processed.features.shape == (RESAMPLED_LENGTH, FEATURE_DIM) and np.all(np.isfinite(processed.features))
    assert processed.metadata["detection_ratio"] < 1.0


def test_a_still_hand_has_no_usable_trajectory():
    code = _reject_code(still_hand_capture())
    assert code == INVALID_TRAJECTORY
    with pytest.raises(HandCaptureRejected) as error:
        _process(still_hand_capture())
    assert not error.value.is_capture_error  # quality failure (QUALITY_INSUFFICIENT), not "no sample"


def test_a_too_long_gesture_is_rejected():
    assert _reject_code(Person(1).capture(1, duration_s=12.0, fps=15)) == CAPTURE_QUALITY_FAILURE


@pytest.mark.parametrize("mutate", [
    lambda p: "not json",
    lambda p: {**p, "gesture_type": "circle"},
    lambda p: {**p, "gesture_type": "infinity"},       # the removed gesture is no longer accepted
    lambda p: {**p, "mirrored": "yes"},
    lambda p: {**p, "image_width": 0},
    lambda p: {**p, "frames": []},
    lambda p: {**p, "frames": [{"t": 0, "landmarks": [[0, 0, 0]] * 20}]},
    lambda p: {**p, "frames": [{"t": 0, "landmarks": [[0, 0, float("nan")]] * 21}]},
    lambda p: {**p, "frames": [{"t": 5, "landmarks": None}, {"t": 5, "landmarks": None}]},
    lambda p: {**p, "frames": [{"landmarks": None}]},
])
def test_malformed_captures_are_rejected_as_malformed(mutate):
    payload = mutate(copy.deepcopy(Person(1).capture(1)))
    with pytest.raises(HandCaptureRejected) as error:
        parse_capture(payload if isinstance(payload, str) else json.dumps(payload))
    assert error.value.code == MALFORMED_CAPTURE and error.value.is_capture_error


def test_the_attempt_number_is_reported():
    with pytest.raises(HandCaptureRejected) as error:
        process_capture(parse_capture(json.dumps(no_hand_capture())), attempt=3)
    assert error.value.attempt == 3 and str(error.value).startswith("gesture 3:")


# ----------------------------------------------------------------------------- normalization


def _shifted(payload, dx, dy, scale=1.0, cx=0.5, cy=0.5):
    out = copy.deepcopy(payload)
    for frame in out["frames"]:
        if frame["landmarks"]:
            frame["landmarks"] = [[cx + (x - cx) * scale + dx, cy + (y - cy) * scale + dy, z * scale] for x, y, z in frame["landmarks"]]
    return out


def test_translation_invariance():
    payload = Person(3).capture(7)
    a = _process(payload).features
    b = _process(_shifted(payload, 0.12, -0.08)).features
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_scale_normalization():
    """The same gesture closer to / further from the camera gives (nearly) the same features."""
    payload = Person(3).capture(7)
    a = _process(payload).features
    b = _process(_shifted(payload, 0, 0, scale=0.7)).features
    np.testing.assert_allclose(a, b, atol=1e-3)


def test_normalize_shape_puts_the_wrist_at_the_origin_with_unit_palm():
    frame = np.random.default_rng(0).uniform(0, 1, (21, 3))
    shape = normalize_shape(frame)
    assert np.allclose(shape[0], 0) and np.isclose(np.linalg.norm(shape[9, :2]), 1.0)


def test_isotropic_coordinates_use_the_aspect_ratio():
    lm = np.zeros((21, 3))
    lm[:, 0] = 0.5
    assert np.allclose(isotropic(lm, 640, 480)[:, 0], 0.5 * 640 / 480)


def test_feature_dimensions_and_weights():
    features = _process(Person(1).capture(1)).features
    assert features.dtype == np.float32 and features.shape == (RESAMPLED_LENGTH, FEATURE_DIM)
    assert weighted(features).shape == features.shape


# ----------------------------------------------------------------------------- temporal representation


def test_different_durations_of_the_same_gesture_stay_comparable():
    person = Person(4)
    slow = weighted(_process(person.capture(1, duration_s=2.4)).features)
    fast = weighted(_process(person.capture(2, duration_s=1.2)).features)
    other = weighted(_process(Person(9).capture(3, duration_s=2.4)).features)
    assert dtw_distance(slow, fast) < dtw_distance(slow, other)


def test_every_capture_is_resampled_to_the_same_length():
    lengths = {_process(Person(1).capture(k, duration_s=d)).features.shape for k, d in ((1, 0.9), (2, 3.5), (3, 6.0))}
    assert lengths == {(RESAMPLED_LENGTH, FEATURE_DIM)}
