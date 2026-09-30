"""Protected-template comparison.

Templates from `biohash.py` are bit vectors, so the natural similarity
measure is Hamming distance; cosine similarity is offered too (on the same
bits cast to floats) since the spec calls it out as an optional metric, and
because `evaluation/privacy_metrics.py`'s Experiment 1 wants to compare
against the raw-embedding cosine similarity `evaluation/metrics.py` already
uses for unprotected recognition.
"""

from __future__ import annotations

import numpy as np

from template_protection.utils import constant_time_equals

_SUPPORTED_METRICS = ("hamming", "cosine")


def compare(template_a: np.ndarray, template_b: np.ndarray, metric: str = "hamming") -> float:
    """Return a similarity score in [0, 1] for two equal-length protected templates.

    - `"hamming"`: fraction of matching bit positions (1.0 = identical,
      0.0 = every bit differs). This is the primary metric - it's what
      `evaluation/privacy_metrics.py`'s FAR/FRR experiment and
      `backend/services/*.py`'s authentication flow use.
    - `"cosine"`: cosine similarity of the two bit vectors treated as
      float vectors, rescaled from [-1, 1] to [0, 1] so both metrics share a
      comparable range. Offered as the spec's optional secondary metric;
      Hamming distance is the better-suited measure for binary templates and
      is what threshold tuning should be based on.

    An exact byte-for-byte match (the case that matters most for "did
    revocation actually invalidate the old template") is additionally checked
    via `utils.constant_time_equals`, so that check itself doesn't leak timing
    information about *where* two templates first diverge.
    """
    if template_a.shape != template_b.shape:
        raise ValueError(f"Templates must be the same shape to compare, got {template_a.shape} vs {template_b.shape}")
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric {metric!r}; expected one of {_SUPPORTED_METRICS}")

    if constant_time_equals(template_a.astype(np.uint8).tobytes(), template_b.astype(np.uint8).tobytes()):
        return 1.0

    if metric == "hamming":
        return float(np.mean(template_a == template_b))

    # metric == "cosine"
    a, b = template_a.astype(np.float64), template_b.astype(np.float64)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    return (cosine + 1.0) / 2.0  # rescale [-1, 1] -> [0, 1]


def accept(score: float, threshold: float, metric: str = "hamming") -> bool:
    """Accept/reject helper: `score >= threshold` for both supported metrics.

    Both `compare()` metrics are defined so that higher means "more likely
    the same biometric" - callers pick `threshold` empirically (see
    `evaluation/privacy_metrics.py`'s EER-based threshold selection, mirroring
    `evaluation/metrics.py::compute_eer` for unprotected embeddings).
    """
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric {metric!r}; expected one of {_SUPPORTED_METRICS}")
    return score >= threshold
