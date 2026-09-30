"""Tests for the enrollment-quality gates added alongside the mostly-frontal capture protocol
(see preprocessing/face.py and embeddings/pipelines.py::_evaluate_face_quality).

Two levels:
- Pure, deterministic unit tests directly against `_evaluate_face_quality`, using synthetic
  `FaceDetection` instances - no MTCNN needed, exact threshold boundaries pinned.
- Real-MTCNN integration tests reusing tests/test_face_enrollment.py's real-portrait fixtures
  (`_portrait`, `_jpg`, `real_client`), proving the gates are actually wired into
  POST /enroll/face/check-pose and POST /enroll end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

from embeddings.pipelines import _evaluate_face_quality
from preprocessing.face import (
    BLUR_MIN_SHARPNESS,
    MAX_CENTER_OFFSET,
    MAX_ROLL_DEGREES,
    MAX_YAW_RATIO,
    MIN_DETECTION_CONFIDENCE,
    MIN_FACE_SIZE_RATIO,
    FaceDetection,
)
from tests.test_face_enrollment import _jpg, _portrait, real_client  # noqa: F401  (real_client is a fixture)

# A detection that clears every gate, so each test below can override exactly one field.
_GOOD = FaceDetection(
    aligned=np.zeros((160, 160, 3), dtype=np.uint8),
    probability=0.999,
    sharpness=BLUR_MIN_SHARPNESS + 100,
    face_size_ratio=0.35,
    center_offset=0.10,
    roll_degrees=2.0,
    yaw_ratio=0.02,
)


def _replace(**kwargs) -> FaceDetection:
    import dataclasses

    return dataclasses.replace(_GOOD, **kwargs)


def test_a_good_detection_is_valid():
    assert _evaluate_face_quality(_GOOD) == "VALID"


def test_low_confidence_is_rejected_before_anything_else():
    assert _evaluate_face_quality(_replace(probability=MIN_DETECTION_CONFIDENCE - 0.01)) == "LOW_CONFIDENCE"


def test_confidence_at_exactly_the_threshold_passes():
    assert _evaluate_face_quality(_replace(probability=MIN_DETECTION_CONFIDENCE)) == "VALID"


def test_blurry_is_rejected():
    assert _evaluate_face_quality(_replace(sharpness=BLUR_MIN_SHARPNESS - 1)) == "BLURRY"


def test_too_small_is_rejected():
    assert _evaluate_face_quality(_replace(face_size_ratio=MIN_FACE_SIZE_RATIO - 0.01)) == "TOO_SMALL"


def test_face_size_at_exactly_the_threshold_passes():
    assert _evaluate_face_quality(_replace(face_size_ratio=MIN_FACE_SIZE_RATIO)) == "VALID"


def test_off_center_is_rejected():
    assert _evaluate_face_quality(_replace(center_offset=MAX_CENTER_OFFSET + 0.01)) == "OFF_CENTER"


def test_center_offset_at_exactly_the_threshold_passes():
    assert _evaluate_face_quality(_replace(center_offset=MAX_CENTER_OFFSET)) == "VALID"


def test_excessive_roll_is_rejected_as_too_angled():
    assert _evaluate_face_quality(_replace(roll_degrees=MAX_ROLL_DEGREES + 1)) == "TOO_ANGLED"
    assert _evaluate_face_quality(_replace(roll_degrees=-(MAX_ROLL_DEGREES + 1))) == "TOO_ANGLED"  # either direction


def test_excessive_yaw_is_rejected_as_too_angled():
    assert _evaluate_face_quality(_replace(yaw_ratio=MAX_YAW_RATIO + 0.01)) == "TOO_ANGLED"
    assert _evaluate_face_quality(_replace(yaw_ratio=-(MAX_YAW_RATIO + 0.01))) == "TOO_ANGLED"


def test_angle_at_exactly_the_threshold_passes():
    assert _evaluate_face_quality(_replace(roll_degrees=MAX_ROLL_DEGREES, yaw_ratio=MAX_YAW_RATIO)) == "VALID"


def test_gates_are_checked_in_a_fixed_priority_order():
    """A detection failing multiple gates at once reports the first one in the documented order
    (confidence, then sharpness, then size, then centering, then angle) - deterministic, not
    whichever the implementation happens to check last."""
    worst = _replace(probability=0.1, sharpness=0.0, face_size_ratio=0.01, center_offset=0.9, roll_degrees=90.0)
    assert _evaluate_face_quality(worst) == "LOW_CONFIDENCE"


# ------------------------------------------------------------------ real MTCNN integration


def _check(real_client, image):
    return real_client.post(
        "/enroll/face/check-pose", data={"pose": "front"}, files={"image": ("p.jpg", _jpg(image), "image/jpeg")}
    ).json()


def test_real_normal_portrait_still_passes_quality_gating(real_client):
    """Regression guard: the new gates must not reject an ordinary, legitimate capture."""
    assert _check(real_client, _portrait())["status"] == "VALID"


def test_real_two_faces_in_frame_is_rejected(real_client):
    import cv2

    portrait = _portrait()
    h, w = portrait.shape[:2]
    half = cv2.resize(portrait, (w // 2, h))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, : w // 2] = half
    canvas[:, w // 2 :] = half

    result = _check(real_client, canvas)
    assert result["status"] == "MULTIPLE_FACES"
    assert "more than one face" in result["detail"].lower()


def test_real_face_too_small_in_frame_is_rejected(real_client):
    portrait = _portrait()
    h, w = portrait.shape[:2]
    small = cv2_resize_small(portrait, w, h)
    canvas = np.full((h, w, 3), 200, dtype=np.uint8)
    canvas[: small.shape[0], : small.shape[1]] = small

    assert _check(real_client, canvas)["status"] == "TOO_SMALL"


def cv2_resize_small(image, w, h):
    import cv2

    return cv2.resize(image, (int(w * 0.35), int(h * 0.35)))


def test_real_excessive_rotation_is_rejected_as_too_angled(real_client):
    import cv2

    portrait = _portrait()
    m = cv2.getRotationMatrix2D((portrait.shape[1] / 2, portrait.shape[0] / 2), 30, 1)
    rotated = cv2.warpAffine(portrait, m, (portrait.shape[1], portrait.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    assert _check(real_client, rotated)["status"] == "TOO_ANGLED"


def test_real_mild_rotation_within_the_new_protocol_still_passes(real_client):
    """The whole point of the new protocol: small, natural variation must still enroll."""
    import cv2

    portrait = _portrait()
    m = cv2.getRotationMatrix2D((portrait.shape[1] / 2, portrait.shape[0] / 2), 5, 1)
    mild = cv2.warpAffine(portrait, m, (portrait.shape[1], portrait.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    assert _check(real_client, mild)["status"] == "VALID"


def test_a_rejected_capture_never_appears_in_embed_poses_output(real_client):
    """A quality-rejected pose must not silently contribute an embedding - confirms the
    centroid can never be contaminated by a rejected capture."""
    import cv2

    portrait = _portrait()
    m = cv2.getRotationMatrix2D((portrait.shape[1] / 2, portrait.shape[0] / 2), 30, 1)
    too_angled = cv2.warpAffine(portrait, m, (portrait.shape[1], portrait.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    response = real_client.post(
        "/enroll",
        data={"user_id": "u-quality", "modality": "face"},
        files={
            "pose_front": ("front.jpg", _jpg(portrait), "image/jpeg"),
            "pose_left": ("left.jpg", _jpg(portrait), "image/jpeg"),
            "pose_right": ("right.jpg", _jpg(portrait), "image/jpeg"),
            "pose_up": ("up.jpg", _jpg(too_angled), "image/jpeg"),  # deliberately bad
            "pose_down": ("down.jpg", _jpg(portrait), "image/jpeg"),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["poses_valid"] == 4
    verdicts = {r["pose"]: r["status"] for r in body["pose_results"]}
    assert verdicts["up"] == "TOO_ANGLED"
    assert verdicts["front"] == verdicts["left"] == verdicts["right"] == verdicts["down"] == "VALID"
