"""Preprocessing pipeline smoke tests using synthetic images (no real biometric data)."""

from __future__ import annotations

import pytest


def test_face_preprocessing_requires_facenet_pytorch_and_detects_no_face_on_noise(random_rgb_image):
    facenet_pytorch = pytest.importorskip(
        "facenet_pytorch", reason="facenet-pytorch not installed in this environment"
    )
    from preprocessing.face import FacePreprocessor

    with pytest.raises(ValueError):
        FacePreprocessor().preprocess(random_rgb_image)
