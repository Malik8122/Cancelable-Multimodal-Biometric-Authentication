"""Common interface every modality's embedding model must implement.

Keeping this interface identical across face / voice is what lets
the rest of the system (template protection in Phase 2, fusion in Phase 3)
treat every modality uniformly:

    embedding = embedder.extract_embedding(image)

`image` is always an RGB `numpy.ndarray` of shape (H, W, 3), dtype uint8 —
callers are responsible for running the modality-specific preprocessing
(see `preprocessing/`) before calling this.
"""

from __future__ import annotations

import abc
from pathlib import Path

import numpy as np


class ModelNotTrainedError(RuntimeError):
    """Raised when a checkpoint is required but hasn't been produced yet.

    Real biometric weights are trained on Colab (see notebooks/) and committed
    back via Git LFS into models/<modality>/saved/. Until that checkpoint
    exists, embedders can optionally run in MOCK_MODE (a deterministic
    pseudo-random embedding derived from the image) purely so the rest of the
    pipeline (preprocessing -> embedding -> [protection -> fusion later]) is
    provably wired end-to-end without requiring GPU access.
    """


class BaseEmbedder(abc.ABC):
    """Abstract base class for a modality's feature-extraction model."""

    #: Fixed-length embedding dimensionality produced by this modality.
    embedding_dim: int

    def __init__(self, checkpoint_path: str | Path | None = None, mock_mode: bool = False):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.mock_mode = mock_mode or self.checkpoint_path is None or not self.checkpoint_path.exists()
        if not self.mock_mode:
            self._load_checkpoint(self.checkpoint_path)

    @abc.abstractmethod
    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load trained weights from `checkpoint_path` into the model."""

    @abc.abstractmethod
    def _extract_embedding_impl(self, image: np.ndarray) -> np.ndarray:
        """Real forward pass. Only called when a checkpoint has been loaded."""

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Return an L2-normalized, fixed-length embedding for `image`.

        In mock mode (no trained checkpoint present), returns a deterministic
        pseudo-random unit vector derived from the image content, so the
        surrounding pipeline can be tested without a GPU. Mock embeddings are
        NOT biometrically meaningful and must never be used for real
        enrollment/authentication decisions.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an RGB image of shape (H, W, 3), got {image.shape}")

        if self.mock_mode:
            return self._mock_embedding(image)

        embedding = self._extract_embedding_impl(image)
        return _l2_normalize(embedding)

    def _mock_embedding(self, image: np.ndarray) -> np.ndarray:
        seed = int(np.sum(image.astype(np.uint64)) % (2**32 - 1))
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.embedding_dim).astype(np.float32)
        return _l2_normalize(vector)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
