"""Voice biometric service: wraps `embeddings.pipelines.VoicePipeline`.

`ModalityService` (base_service.py, shared by every modality) calls
`self.pipeline.embed(raw_image)` with exactly one positional argument - true
for Face, whose pipeline takes just an image array. Voice's
pipeline needs a second argument (`sample_rate`) because a raw waveform
alone doesn't say what rate it was captured at - see
`embeddings/pipelines.py::VoicePipeline`'s own docstring for why its
`embed()` is overridden with that extra parameter.

Rather than changing `ModalityService` (shared by all four modalities, and
explicitly out of scope to redesign), `_VoicePipelineAdapter` below satisfies
the *same* single-argument `embed(raw_input)` contract by accepting
`raw_input` as a `(waveform, sample_rate)` tuple instead of a bare array -
exactly what `backend/utils.py::decode_audio` already returns. This is a
voice-specific adapter, invisible to `ModalityService` and to every other
modality's code path.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from backend.config import get_settings
from backend.services.base_service import ModalityService
from embeddings.pipelines import VoicePipeline


class _VoicePipelineAdapter:
    """Adapts `VoicePipeline.embed(waveform, sample_rate)` to `embed(raw_input)`."""

    def __init__(self, voice_pipeline: VoicePipeline):
        self._voice_pipeline = voice_pipeline

    def embed(self, raw_input: tuple[np.ndarray, int]) -> np.ndarray:
        waveform, sample_rate = raw_input
        return self._voice_pipeline.embed(waveform, sample_rate=sample_rate)

    @property
    def embedding_dim(self) -> int:
        return self._voice_pipeline.embedding_dim

    @property
    def is_mock(self) -> bool:
        return self._voice_pipeline.is_mock


@lru_cache
def get_voice_service() -> ModalityService:
    settings = get_settings()
    pipeline = VoicePipeline(checkpoint_path=settings.voice_model_path)
    return ModalityService(modality="voice", pipeline=_VoicePipelineAdapter(pipeline), settings=settings)
