"""Voice embedding backbone: SpeechBrain ECAPA-TDNN.

Uses a `__new__`-based lazy-import trick: `speechbrain`/`torch.nn` are only
imported the first time a real (non-mock-mode) VoiceEmbedder is actually
constructed, so importing this module - or models/voice/inference.py, or
models/voice/config.py etc. - never requires `speechbrain` to be installed.
See docs/VOICE_MODEL.md's "Zero hard dependencies for preprocessing/testing"
note for why that matters on a dev machine without these optional deps.

Takes precomputed log-mel features directly (shape (batch, time, n_mels)),
not raw waveform - `speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN` supports
this out of the box, which is what lets preprocessing/voice.py (not
SpeechBrain's own built-in feature extractor) own log-mel extraction, the
same way preprocessing/face.py owns its
modality's classical preprocessing rather than delegating it into the model.
"""

from __future__ import annotations

from models.voice.config import VoiceConfig


class VoiceEmbeddingNet:
    """Thin wrapper so this module has no hard torch/speechbrain import at module load time."""

    def __new__(cls, config: VoiceConfig | None = None):
        import torch.nn as nn
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

        config = config or VoiceConfig()

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = ECAPA_TDNN(input_size=config.n_mels, lin_neurons=config.embedding_dim)

            def forward(self, features):
                # ECAPA_TDNN returns (batch, 1, lin_neurons); squeeze the
                # middle dim so callers get a plain (batch, lin_neurons)
                # embedding, matching every other modality's backbone output shape.
                return self.backbone(features).squeeze(1)

        return _Net()
