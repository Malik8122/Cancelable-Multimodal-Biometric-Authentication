"""Hand gesture capture reliability (SYNTHETIC landmark captures - software tests, not accuracy evidence).

Root causes found in real sessions and fixed in preprocessing/hand_gesture.py::process_capture:
- real recordings contain seconds of idle hand before/after a ~2 s gesture, which shifted the gesture inside the
  resampled sequence differently in every sample -> motion segmentation trims the idle hand;
- some real samples were tracked at ~5 fps -> a tracking-rate capture check (a capture failure, never a mismatch).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from preprocessing.hand_gesture import (
    CAPTURE_QUALITY_FAILURE,
    INSUFFICIENT_FRAMES,
    MIN_TRACKING_FPS,
    NO_HAND_DETECTED,
    HandCaptureRejected,
    parse_capture,
    process_capture,
    weighted,
)
from backend.config import get_settings
from template_protection.dtw import dtw_distance
from tests.hand_signals import Person, at_frame_rate, no_hand_capture, with_idle

ALICE, BOB = Person(1), Person(2)


def _process(capture):
    return process_capture(parse_capture(json.dumps(capture).encode()))


def _features(capture):
    return weighted(_process(capture).features)


def test_idle_hand_before_and_after_the_gesture_is_trimmed():
    gesture = ALICE.capture(5)
    padded = with_idle(gesture, lead_s=1.5, tail_s=1.2, seed=5)
    processed = _process(padded)
    assert processed.metadata["idle_trimmed_s"] > 2.0          # ~2.7 s of idle hand removed
    assert processed.metadata["motion_duration_s"] < processed.metadata["duration_s"]
    # The trimmed sample is (nearly) the gesture itself: far closer than the untrimmed shift used to allow.
    assert dtw_distance(_features(padded), _features(gesture)) < 0.1


def test_idle_padding_no_longer_separates_a_genuine_user_from_their_own_enrollment():
    enrolled = [_features(with_idle(ALICE.capture(k), *np.random.default_rng(k).uniform(0, 2, 2), seed=k)) for k in range(3)]
    genuine = _features(with_idle(ALICE.capture(50), 1.8, 0.2, seed=50))
    impostor = _features(with_idle(BOB.capture(51), 0.3, 1.5, seed=51))
    g = np.median([dtw_distance(genuine, e) for e in enrolled])
    i = np.median([dtw_distance(impostor, e) for e in enrolled])
    assert g < get_settings().hand_gesture_dtw_threshold < i


def test_a_sample_tracked_too_slowly_is_a_capture_failure_not_a_mismatch():
    slow = at_frame_rate(with_idle(ALICE.capture(9, duration_s=4.0), 0.5, 0.5, seed=9), keep_every=5)  # ~6 fps
    with pytest.raises(HandCaptureRejected) as error:
        _process(slow)
    assert error.value.code == CAPTURE_QUALITY_FAILURE and "frames per second" in str(error.value)
    assert error.value.is_capture_error is False  # quality failure: the user is asked to retake, nothing compared


def test_a_normally_tracked_sample_reports_its_tracking_rate():
    metadata = _process(ALICE.capture(3)).metadata
    assert metadata["tracking_fps"] >= MIN_TRACKING_FPS


def test_empty_sequence_is_rejected():
    with pytest.raises(HandCaptureRejected) as error:
        _process(no_hand_capture())
    assert error.value.code == NO_HAND_DETECTED


def test_too_short_sequence_is_rejected():
    short = ALICE.capture(4)
    short["frames"] = short["frames"][:8]
    with pytest.raises(HandCaptureRejected) as error:
        _process(short)
    assert error.value.code == INSUFFICIENT_FRAMES


def test_enrollment_and_authentication_produce_the_same_feature_space():
    """The same capture through the enrollment path and the authentication path yields identical features."""
    from backend.services.hand_service import get_hand_service

    get_hand_service.cache_clear()
    service = get_hand_service()
    capture = parse_capture(json.dumps(ALICE.capture(21)).encode())
    via_enrollment = weighted(service.process(capture, attempt=1).features)   # enroll_attempts uses process(capture, attempt=n)
    via_authentication = weighted(service.process(capture).features)          # authenticate uses process(capture)
    assert np.array_equal(via_enrollment, via_authentication)
    assert via_enrollment.shape == (64, 13)


def test_the_keyed_transform_preserves_dtw_distances():
    """Enrollment and authentication are transformed with the same set key; the transform must not change DTW."""
    from template_protection.hkdf_keys import derive_key
    from template_protection.sequence_transform import protect

    a, b = _features(ALICE.capture(1)), _features(ALICE.capture(2))
    key = derive_key("secret", application_id="app", user_id="u", modality="hand", key_version=3)
    pa, pb = protect([a, b], key)
    assert abs(dtw_distance(pa, pb) - dtw_distance(a, b)) < 1e-4
