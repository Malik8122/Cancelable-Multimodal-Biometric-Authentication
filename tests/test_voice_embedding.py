"""Offline tests for models/voice/inference.py.

Mock mode only (no checkpoint, no speechbrain/torchaudio needed) - proves the
VoiceEmbedder <-> BaseEmbedder contract and the embeddings/pipelines.py
integration are wired correctly, independent of whether real ECAPA-TDNN
weights exist yet. See docs/VOICE_MODEL.md's "Integration" section for why
`image` here is a pseudo-RGB log-mel spectrogram, not a photograph.
"""

from __future__ import annotations

import numpy as np
import pytest

from embeddings.pipelines import VoicePipeline
from models.voice.inference import VOICE_EMBEDDING_DIM, VoiceEmbedder


@pytest.fixture
def pseudo_rgb_mel_image():
    rng = np.random.default_rng(7)
    mel = rng.standard_normal((80, 400)).astype(np.float32)
    return np.stack([mel] * 3, axis=-1)


def test_mock_embedding_shape_and_normalization(pseudo_rgb_mel_image):
    embedder = VoiceEmbedder(checkpoint_path=None)
    assert embedder.mock_mode is True

    embedding = embedder.extract_embedding(pseudo_rgb_mel_image)

    assert embedding.shape == (VOICE_EMBEDDING_DIM,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_mock_embedding_is_deterministic_for_same_input(pseudo_rgb_mel_image):
    embedder = VoiceEmbedder(checkpoint_path=None)

    first = embedder.extract_embedding(pseudo_rgb_mel_image)
    second = embedder.extract_embedding(pseudo_rgb_mel_image)

    np.testing.assert_array_equal(first, second)


def test_mock_embedding_differs_for_different_input(pseudo_rgb_mel_image):
    embedder = VoiceEmbedder(checkpoint_path=None)
    other_image = -pseudo_rgb_mel_image

    first = embedder.extract_embedding(pseudo_rgb_mel_image)
    second = embedder.extract_embedding(other_image)

    assert not np.allclose(first, second)


def test_rejects_non_3_channel_input():
    embedder = VoiceEmbedder(checkpoint_path=None)
    grayscale = np.zeros((80, 400), dtype=np.float32)

    with pytest.raises(ValueError):
        embedder.extract_embedding(grayscale)


def test_nonexistent_checkpoint_path_falls_back_to_mock_mode(tmp_path):
    missing_checkpoint = tmp_path / "does_not_exist.pt"
    embedder = VoiceEmbedder(checkpoint_path=missing_checkpoint)
    assert embedder.mock_mode is True


def test_voice_pipeline_embeds_a_raw_waveform_end_to_end():
    pipeline = VoicePipeline(checkpoint_path=None)
    assert pipeline.is_mock is True
    assert pipeline.embedding_dim == VOICE_EMBEDDING_DIM

    t = np.linspace(0, 2.0, 32_000, endpoint=False)
    waveform = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    embedding = pipeline.embed(waveform, sample_rate=16_000)

    assert embedding.shape == (VOICE_EMBEDDING_DIM,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_voice_checkpoint_is_registered_in_default_checkpoints():
    from embeddings.constants import DEFAULT_CHECKPOINTS

    assert "voice" in DEFAULT_CHECKPOINTS
    assert DEFAULT_CHECKPOINTS["voice"].name == "voice_embedder.pt"
