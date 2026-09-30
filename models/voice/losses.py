"""Voice's training loss: the same ArcFace head every other modality uses.

Deliberately does not reimplement additive angular margin loss - Deng et
al.'s ArcFace math is already implemented once, reviewed, and used by all
three existing modalities' training notebooks in
`models/common/arcface.py::ArcMarginProduct`. This module is a thin,
voice-specific factory around it so `models/voice/train.py` gets the same
margin/scale configurability the spec asks for without a second
implementation to keep numerically consistent with the first.
"""

from __future__ import annotations

from models.common.arcface import ArcMarginProduct
from models.voice.config import VoiceConfig


def build_arcface_head(num_classes: int, config: VoiceConfig | None = None) -> ArcMarginProduct:
    """ArcFace head sized for `num_classes` speakers, using `config`'s margin/scale.

    `in_features` is always `config.embedding_dim` (192 by default) - the
    dimensionality `models/voice/model.py::VoiceEmbeddingNet` actually
    outputs, so this head and the backbone can't silently drift apart.
    """
    config = config or VoiceConfig()
    return ArcMarginProduct(
        in_features=config.embedding_dim,
        out_features=num_classes,
        s=config.arcface_scale,
        m=config.arcface_margin,
    )
