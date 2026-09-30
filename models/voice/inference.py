"""Voice embedding model: SpeechBrain ECAPA-TDNN speaker embedding.

Produces a 192-dimensional embedding from an 80-bin log-mel filterbank (see
preprocessing/voice.py). `models/voice/saved/` holds any fine-tuned
checkpoint produced by notebooks/04_voice_training.ipynb or
kaggle_kernels/voice_training/.

**Interface note (see docs/VOICE_MODEL.md's "Integration" section for the
full rationale):** `VoiceEmbedder.extract_embedding` has the exact same
signature as every other modality - `extract_embedding(image: np.ndarray)`
where `image` is a preprocessed `(H, W, 3)` array - *not*
`extract_embedding(audio_path)` as an early draft of the spec suggested.
`image` here is really an (n_mels, n_frames) log-mel spectrogram replicated
to 3 identical channels (by `embeddings/pipelines.py::VoicePipeline`, a
"2D -> pseudo-RGB" trick) so
that `BaseEmbedder`'s shared, deliberately narrow contract - the thing that
lets `embeddings/pipelines.py` and later fusion/backend code treat every
modality uniformly - never needs a voice-specific carve-out. The
path-based convenience functions the spec actually wants
(`load_model`/`extract_embedding`/`compare_embeddings`) are the free
functions below, built on top of that contract rather than replacing it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from models.common.base_embedder import BaseEmbedder
from models.voice.config import VoiceConfig

VOICE_EMBEDDING_DIM = 192


class VoiceEmbedder(BaseEmbedder):
    embedding_dim = VOICE_EMBEDDING_DIM

    def __init__(self, checkpoint_path: str | Path | None = None, mock_mode: bool = False, device: str = "cpu"):
        self.device = device
        self._model = None
        super().__init__(checkpoint_path=checkpoint_path, mock_mode=mock_mode)

    def _build_model(self):
        from models.voice.model import VoiceEmbeddingNet

        return VoiceEmbeddingNet(VoiceConfig(embedding_dim=self.embedding_dim)).to(self.device)

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        import torch

        self._model = self._build_model()
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def _extract_embedding_impl(self, image: np.ndarray) -> np.ndarray:
        """`image` is a (n_mels, n_frames, 3) pseudo-RGB log-mel spectrogram.

        All 3 channels are identical (see this module's docstring), so
        channel 0 recovers the original (n_mels, n_frames) spectrogram
        losslessly. ECAPA_TDNN expects (batch, time, n_mels), hence the
        transpose before adding the batch dimension.
        """
        import torch

        mel = image[:, :, 0]  # (n_mels, n_frames)
        tensor = torch.from_numpy(mel).float().transpose(0, 1).unsqueeze(0).to(self.device)  # (1, n_frames, n_mels)
        with torch.no_grad():
            embedding = self._model(tensor)
        return embedding.squeeze(0).cpu().numpy()


def load_model(checkpoint_path: str | Path | None = None, device: str | None = None) -> VoiceEmbedder:
    """Load (or mock-mode-initialize) a VoiceEmbedder. `device=None` auto-detects (models/voice/utils.py)."""
    from models.voice.utils import detect_device

    resolved_device = device or detect_device()
    return VoiceEmbedder(checkpoint_path=checkpoint_path, device=resolved_device)


def _embed_audio_file(audio_path: str | Path, model: VoiceEmbedder) -> np.ndarray:
    """Load `audio_path`, preprocess it, and embed it through `model`.

    Builds a fresh `VoicePreprocessor` per call - it's cheap (no model
    weights, just a mel-filterbank matrix) and keeps this function stateless,
    unlike `model` itself which is expensive to reload.
    """
    from preprocessing.voice import VoicePreprocessor, load_wav_file

    waveform, sample_rate = load_wav_file(audio_path)
    mel = VoicePreprocessor().preprocess(waveform, sample_rate=sample_rate, training=False)
    pseudo_rgb = np.stack([mel] * 3, axis=-1)
    return model.extract_embedding(pseudo_rgb)


def extract_embedding(audio_path: str | Path, model: VoiceEmbedder | None = None) -> np.ndarray:
    """Path-based convenience wrapper: preprocess `audio_path` end-to-end and embed it.

    This is the actual `extract_embedding(audio_path)` entry point the spec
    asks for (see this module's docstring for why it isn't a `VoiceEmbedder`
    method). `model` can be reused across multiple calls (e.g. from
    `compare_embeddings`) to avoid reloading the checkpoint each time.
    """
    model = model or load_model()
    return _embed_audio_file(audio_path, model)


def compare_embeddings(audio_path_1: str | Path, audio_path_2: str | Path, model: VoiceEmbedder | None = None) -> float:
    """Cosine similarity between the speaker embeddings of two audio files."""
    from evaluation.metrics import cosine_similarity

    model = model or load_model()
    embedding_1 = extract_embedding(audio_path_1, model=model)
    embedding_2 = extract_embedding(audio_path_2, model=model)
    return cosine_similarity(embedding_1, embedding_2)
