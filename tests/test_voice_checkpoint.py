"""Checkpoint save/load round-trip for the Voice module.

Mirrors tests/test_checkpoint_io.py's pattern (a tiny real nn.Module, not the
full ECAPA-TDNN backbone) for the dependency-free .h5 round-trip, then
separately exercises the actual VoiceEmbedder .pt loading path and, only if
`speechbrain`/`torchaudio` are installed, a real end-to-end inference run -
skipped cleanly otherwise (see tests/test_preprocessing.py's
`pytest.importorskip` convention, already used for face's real
checkpoint in tests/test_services.py).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.common.checkpoint_io import load_h5_as_state_dict, save_state_dict_as_h5
from models.voice.inference import VOICE_EMBEDDING_DIM, VoiceEmbedder


class _TinyVoiceNet(nn.Module):
    """Stand-in for the real ECAPA-TDNN backbone - same round-trip guarantees
    (.pt save/load, .h5 export/import) without needing speechbrain installed."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(80, VOICE_EMBEDDING_DIM)

    def forward(self, x):
        return self.fc(x.mean(dim=1))


def test_pt_and_h5_checkpoint_round_trip(tmp_path):
    model = _TinyVoiceNet()
    original_state_dict = model.state_dict()

    pt_path = tmp_path / "voice_embedder.pt"
    h5_path = tmp_path / "voice_embedder.h5"
    torch.save(original_state_dict, pt_path)
    save_state_dict_as_h5(original_state_dict, h5_path)

    assert pt_path.exists()
    assert h5_path.exists()

    restored_from_pt = torch.load(pt_path)
    restored_from_h5 = load_h5_as_state_dict(h5_path)

    for key in original_state_dict:
        torch.testing.assert_close(restored_from_pt[key], original_state_dict[key])
        torch.testing.assert_close(restored_from_h5[key], original_state_dict[key])


def test_voice_embedder_falls_back_to_mock_mode_without_a_checkpoint():
    embedder = VoiceEmbedder(checkpoint_path=None)
    assert embedder.mock_mode is True


def test_voice_embedder_falls_back_to_mock_mode_for_a_missing_checkpoint_path(tmp_path):
    embedder = VoiceEmbedder(checkpoint_path=tmp_path / "does_not_exist.pt")
    assert embedder.mock_mode is True


def test_real_checkpoint_loads_and_runs_inference(tmp_path):
    """Only runs with the real ECAPA-TDNN backbone available; skips cleanly otherwise."""
    pytest.importorskip("speechbrain", reason="speechbrain not installed in this environment")
    pytest.importorskip("torchaudio", reason="torchaudio not installed in this environment")

    from models.voice.model import VoiceEmbeddingNet

    model = VoiceEmbeddingNet()
    checkpoint_path = tmp_path / "voice_embedder.pt"
    torch.save(model.state_dict(), checkpoint_path)

    embedder = VoiceEmbedder(checkpoint_path=checkpoint_path)
    assert embedder.mock_mode is False

    pseudo_rgb_mel = np.stack([np.random.default_rng(0).standard_normal((80, 400)).astype(np.float32)] * 3, axis=-1)
    embedding = embedder.extract_embedding(pseudo_rgb_mel)

    assert embedding.shape == (VOICE_EMBEDDING_DIM,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
