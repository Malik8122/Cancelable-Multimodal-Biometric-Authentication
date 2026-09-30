"""Landmark-based face alignment (preprocessing/face.py): transform maths, template, failure handling, gating."""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing.face import (
    ALIGNMENT_FAILED,
    ALIGNMENT_TEMPLATE_160,
    LANDMARK_FAILURE,
    AlignmentFailed,
    FacePreprocessor,
    FacePreprocessorAligned,
    FacePreprocessorBaseline,
    LandmarkFailure,
    align_face,
    estimate_similarity_transform,
    validate_landmarks,
)


def _apply(M, pts):
    return pts @ M[:, :2].T + M[:, 2]


def _similarity(angle_deg, scale, tx, ty):
    a = np.radians(angle_deg)
    return np.array([[scale * np.cos(a), -scale * np.sin(a), tx], [scale * np.sin(a), scale * np.cos(a), ty]])


def test_template_ordering_and_geometry():
    le, re_, nose, ml, mr = ALIGNMENT_TEMPLATE_160
    assert le[0] < re_[0] and ml[0] < mr[0]  # image-left first (MTCNN order)
    assert le[1] == re_[1] and ml[1] == mr[1]  # level eyes and mouth
    assert nose[0] == 80.0 and np.isclose((le[0] + re_[0]) / 2, 80.0)  # centred, symmetric
    assert le[1] < nose[1] < ml[1]  # eyes above nose above mouth
    assert np.all((ALIGNMENT_TEMPLATE_160 > 0) & (ALIGNMENT_TEMPLATE_160 < 160))


def test_similarity_transform_recovers_a_known_transform_exactly():
    true = _similarity(23.0, 1.7, -40.0, 12.5)
    src = np.array([[10, 20], [60, 18], [35, 45], [15, 70], [55, 72]], dtype=float)
    M = estimate_similarity_transform(src, _apply(true, src))
    assert np.allclose(M, true, atol=1e-9)


def test_template_onto_itself_is_the_identity():
    M = estimate_similarity_transform(ALIGNMENT_TEMPLATE_160, ALIGNMENT_TEMPLATE_160)
    assert np.allclose(M, [[1, 0, 0], [0, 1, 0]], atol=1e-9)


def test_transform_is_a_similarity_without_reflection():
    rng = np.random.default_rng(0)
    for _ in range(20):
        src = ALIGNMENT_TEMPLATE_160 + rng.normal(0, 6, (5, 2))
        M = estimate_similarity_transform(src, ALIGNMENT_TEMPLATE_160)
        A = M[:, :2]
        s = np.sqrt(abs(np.linalg.det(A)))
        assert np.linalg.det(A) > 0  # no mirror image
        assert np.allclose(A / s @ (A / s).T, np.eye(2), atol=1e-9)  # rotation x uniform scale


def test_mirrored_input_is_not_turned_into_a_reflection():
    mirrored = np.c_[160 - ALIGNMENT_TEMPLATE_160[:, 0], ALIGNMENT_TEMPLATE_160[:, 1]]
    assert np.linalg.det(estimate_similarity_transform(mirrored, ALIGNMENT_TEMPLATE_160)[:, :2]) > 0


def test_aligned_landmarks_land_on_the_template():
    true = _similarity(-17.0, 0.8, 30.0, -5.0)
    landmarks = _apply(true, ALIGNMENT_TEMPLATE_160)  # a rotated/scaled/shifted face
    M = estimate_similarity_transform(landmarks, ALIGNMENT_TEMPLATE_160)
    assert np.allclose(_apply(M, landmarks), ALIGNMENT_TEMPLATE_160, atol=1e-8)


def test_align_face_output_shape_dtype_and_determinism():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (250, 250, 3), dtype=np.uint8)
    landmarks = _apply(_similarity(10.0, 1.2, 20.0, 15.0), ALIGNMENT_TEMPLATE_160)
    a, b = align_face(image, landmarks), align_face(image, landmarks)
    assert a.shape == (160, 160, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)


def test_align_face_moves_image_content_to_the_template():
    """A bright dot at the left-eye landmark must end up at the template's left-eye position."""
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    landmarks = _apply(_similarity(25.0, 1.5, 60.0, 40.0), ALIGNMENT_TEMPLATE_160)
    x, y = np.round(landmarks[0]).astype(int)
    image[y - 2:y + 3, x - 2:x + 3] = 255
    out = align_face(image, landmarks)
    ys, xs = np.nonzero(out[:, :, 0] > 100)
    assert abs(xs.mean() - ALIGNMENT_TEMPLATE_160[0, 0]) < 2.0 and abs(ys.mean() - ALIGNMENT_TEMPLATE_160[0, 1]) < 2.0


@pytest.mark.parametrize("landmarks", [
    None,
    np.zeros((4, 2)),
    np.full((5, 2), np.nan),
    np.array([[50, 60], [51, 60], [50, 80], [45, 100], [55, 100]]),  # eyes 1 px apart
    np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]),    # collinear
    "not landmarks",
])
def test_invalid_landmarks_raise_alignment_failed(landmarks):
    with pytest.raises(AlignmentFailed):
        validate_landmarks(landmarks)


def test_alignment_failed_is_a_value_error_so_callers_reject_cleanly():
    assert issubclass(AlignmentFailed, ValueError)


def test_implausible_scale_is_rejected():
    tiny = ALIGNMENT_TEMPLATE_160 * 0.001 + 5
    with pytest.raises(AlignmentFailed):
        estimate_similarity_transform(tiny, ALIGNMENT_TEMPLATE_160)


def test_baseline_is_the_unchanged_preprocessor():
    assert FacePreprocessorBaseline is FacePreprocessor
    assert issubclass(FacePreprocessorAligned, FacePreprocessor)


def test_quality_gate_reports_alignment_failure_distinctly(monkeypatch):
    from embeddings.pipelines import FacePipeline

    pipeline = FacePipeline(checkpoint_path=None, alignment="similarity")

    def boom(image):
        raise AlignmentFailed("degenerate")

    monkeypatch.setattr(pipeline._preprocessor, "detect_and_align", boom)
    assert pipeline.check_capture(np.zeros((100, 100, 3), dtype=np.uint8)) == ALIGNMENT_FAILED
    _, report = pipeline.embed_poses([("front", np.zeros((100, 100, 3), dtype=np.uint8))])
    assert report == [{"pose": "front", "status": ALIGNMENT_FAILED}]


def test_unknown_alignment_mode_is_rejected():
    from embeddings.pipelines import FacePipeline

    with pytest.raises(ValueError):
        FacePipeline(checkpoint_path=None, alignment="affine")


def test_production_default_is_unchanged():
    from backend.config import Settings

    assert Settings(master_secret="x").face_alignment == "bbox"


def test_real_face_alignment_levels_the_eyes():
    """On a real public LFW image (skipped if the dataset is absent), the aligned landmarks sit on the template."""
    from pathlib import Path

    import cv2
    from PIL import Image

    root = Path(__file__).resolve().parents[1] / "data" / "lfw" / "lfw_home" / "lfw_funneled"
    images = sorted(root.glob("*/*.jpg"))[:3] if root.exists() else []
    if not images:
        pytest.skip("LFW not downloaded")
    pre = FacePreprocessorAligned()
    for path in images:
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        boxes, _, points = pre._get_detector().detect(Image.fromarray(rgb), landmarks=True)
        k = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        M = estimate_similarity_transform(validate_landmarks(points[k]), ALIGNMENT_TEMPLATE_160)
        le, re_ = _apply(M, validate_landmarks(points[k]))[:2]
        assert abs(np.degrees(np.arctan2(re_[1] - le[1], re_[0] - le[0]))) < 5.0
        assert pre.preprocess(rgb).shape == (160, 160, 3)


@pytest.mark.parametrize("angle,scale,shift", [(30.0, 1.0, (0, 0)), (-45.0, 1.0, (0, 0)), (0.0, 0.5, (0, 0)), (0.0, 2.5, (0, 0)),
                                               (0.0, 1.0, (70, -30)), (12.0, 1.8, (40, 55))])
def test_rotation_scale_and_translation_are_normalized(angle, scale, shift):
    """Whatever rotation / scale / offset the face has in the capture, the aligned landmarks sit on the template."""
    landmarks = _apply(_similarity(angle, scale, *shift), ALIGNMENT_TEMPLATE_160) + 100.0
    M = estimate_similarity_transform(landmarks, ALIGNMENT_TEMPLATE_160)
    A = M[:, :2]
    assert np.isclose(np.sqrt(np.linalg.det(A)), 1.0 / scale)  # scale undone
    assert np.isclose(np.degrees(np.arctan2(A[1, 0], A[0, 0])), -angle, atol=1e-6)  # rotation undone
    assert np.allclose(_apply(M, landmarks), ALIGNMENT_TEMPLATE_160, atol=1e-8)  # translation undone


def test_rgb_channels_are_preserved():
    """Alignment is purely geometric: a pure-red / green / blue image stays that colour (no BGR swap)."""
    landmarks = _apply(_similarity(8.0, 1.1, 30.0, 30.0), ALIGNMENT_TEMPLATE_160)
    for channel in range(3):
        image = np.zeros((260, 260, 3), dtype=np.uint8)
        image[:, :, channel] = 200
        out = align_face(image, landmarks)
        inside = out[60:100, 60:100]
        assert np.all(inside[:, :, channel] == 200) and np.all(np.delete(inside, channel, axis=2) == 0)


@pytest.mark.parametrize("landmarks", [None, np.zeros((4, 2)), np.full((5, 2), np.nan), "not landmarks"])
def test_missing_or_malformed_landmarks_are_a_landmark_failure(landmarks):
    with pytest.raises(LandmarkFailure):
        validate_landmarks(landmarks)


@pytest.mark.parametrize("landmarks", [
    np.array([[50, 60], [51, 60], [50, 80], [45, 100], [55, 100]]),
    np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]),
])
def test_degenerate_landmarks_are_an_alignment_failure_not_a_landmark_failure(landmarks):
    with pytest.raises(AlignmentFailed) as info:
        validate_landmarks(landmarks)
    assert not isinstance(info.value, LandmarkFailure)


def test_quality_gate_reports_landmark_failure_distinctly(monkeypatch):
    from embeddings.pipelines import FacePipeline

    pipeline = FacePipeline(checkpoint_path=None, alignment="similarity")

    def boom(image):
        raise LandmarkFailure("none")

    monkeypatch.setattr(pipeline._preprocessor, "detect_and_align", boom)
    assert pipeline.check_capture(np.zeros((100, 100, 3), dtype=np.uint8)) == LANDMARK_FAILURE
    _, report = pipeline.embed_poses([("front", np.zeros((100, 100, 3), dtype=np.uint8))])
    assert report == [{"pose": "front", "status": LANDMARK_FAILURE}]


def test_every_face_status_has_a_user_hint():
    from backend.services import face_enrollment as fe

    for status in (fe.NO_FACE, fe.BLURRY, fe.POSE_INVALID, fe.ALIGNMENT_FAILED, fe.LANDMARK_FAILURE):
        assert status in fe.POSE_HINTS
    assert fe.POSE_INVALID == fe.TOO_ANGLED


def test_quality_gates_are_measured_before_alignment():
    """Roll/yaw come from the ORIGINAL capture's landmarks: an aligned crop is level by construction, so gating on it
    would never reject a tilted head."""
    from preprocessing.face import FacePreprocessor as P

    tilted = _apply(_similarity(35.0, 1.0, 40.0, 40.0), ALIGNMENT_TEMPLATE_160)
    box = np.array([20.0, 20.0, 240.0, 240.0])
    det = P._with_quality_signals(np.zeros((160, 160, 3), np.uint8), np.zeros((260, 260, 3), np.uint8), box, 0.99, tilted)
    assert abs(det.roll_degrees - 35.0) < 1e-6


@pytest.mark.parametrize("value,expected", [("FACE_BASELINE", "bbox"), ("FACE_ALIGNED", "similarity"), ("bbox", "bbox"), ("similarity", "similarity")])
def test_face_mode_names_are_accepted(value, expected):
    from backend.config import Settings

    assert Settings(master_secret="x", face_alignment=value).face_alignment == expected


def test_threshold_source_defaults_to_the_teacher_baseline():
    from backend.config import Settings

    s = Settings(master_secret="x")
    assert s.threshold_source == "TEACHER_REQUESTED_BASELINE"
    # Face keeps the teacher-requested 0.80; voice was raised to 0.90 on 2026-09-30 (docs/VOICE_MODEL.md).
    assert (s.face_cosine_threshold, s.voice_euclidean_threshold) == (0.80, 0.90)
