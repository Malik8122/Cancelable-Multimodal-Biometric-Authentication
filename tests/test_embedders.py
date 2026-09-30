"""Embedding-interface smoke tests, run in MOCK_MODE (no trained checkpoint required).

These prove that every modality embedder honors the same
BaseEmbedder contract (models/common/base_embedder.py) - fixed dimension,
L2-normalized output, deterministic given the same input - independent of
whether real weights have been trained yet.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.face.inference import FACE_EMBEDDING_DIM, FaceEmbedder
from models.voice.inference import VOICE_EMBEDDING_DIM, VoiceEmbedder

EMBEDDER_CASES = [
    (FaceEmbedder, FACE_EMBEDDING_DIM),
    (VoiceEmbedder, VOICE_EMBEDDING_DIM),
]


@pytest.mark.parametrize("embedder_cls,expected_dim", EMBEDDER_CASES)
def test_mock_embedding_shape_and_normalization(embedder_cls, expected_dim, random_rgb_image):
    embedder = embedder_cls(checkpoint_path=None)  # no checkpoint -> MOCK_MODE
    assert embedder.mock_mode is True

    embedding = embedder.extract_embedding(random_rgb_image)

    assert embedding.shape == (expected_dim,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


@pytest.mark.parametrize("embedder_cls,expected_dim", EMBEDDER_CASES)
def test_mock_embedding_is_deterministic_for_same_image(embedder_cls, expected_dim, random_rgb_image):
    embedder = embedder_cls(checkpoint_path=None)

    first = embedder.extract_embedding(random_rgb_image)
    second = embedder.extract_embedding(random_rgb_image)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("embedder_cls,expected_dim", EMBEDDER_CASES)
def test_mock_embedding_differs_for_different_images(embedder_cls, expected_dim, random_rgb_image):
    embedder = embedder_cls(checkpoint_path=None)
    other_image = 255 - random_rgb_image

    first = embedder.extract_embedding(random_rgb_image)
    second = embedder.extract_embedding(other_image)

    assert not np.allclose(first, second)


def test_rejects_non_rgb_input(random_rgb_image):
    embedder = FaceEmbedder(checkpoint_path=None)
    grayscale = random_rgb_image[:, :, 0]

    with pytest.raises(ValueError):
        embedder.extract_embedding(grayscale)


def test_nonexistent_checkpoint_path_falls_back_to_mock_mode(tmp_path):
    missing_checkpoint = tmp_path / "does_not_exist.pt"
    embedder = FaceEmbedder(checkpoint_path=missing_checkpoint)
    assert embedder.mock_mode is True
