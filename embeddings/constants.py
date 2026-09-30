"""Default checkpoint locations for each modality.

Pointing at a path that doesn't exist yet is fine and expected before the
Colab notebooks have been run - BaseEmbedder falls back to MOCK_MODE in that
case (see models/common/base_embedder.py).
"""

from pathlib import Path

_MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"

DEFAULT_CHECKPOINTS = {
    "face": _MODELS_ROOT / "face" / "saved" / "face_embedder.pt",
    "voice": _MODELS_ROOT / "voice" / "saved" / "voice_embedder.pt",
}
