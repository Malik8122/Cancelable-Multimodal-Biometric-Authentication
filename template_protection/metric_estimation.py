"""Calibrated ESTIMATES of embedding-space metrics from a protected-template Hamming comparison.

The only comparison the system can perform is between two cancelable
templates: Hamming similarity of two 256-bit BioHash codes
(`matcher.compare(..., metric="hamming")`). Raw embeddings are never stored,
so the true face cosine similarity or voice Euclidean distance between the
enrolled and the live embedding cannot be computed - and cannot be recovered
exactly from the templates either.

What this module offers instead is an ESTIMATE: the embedding cosine that, on
average, produces the observed Hamming similarity for that modality's
embedding dimension and template length. The curve is fitted offline by
`scripts/calibrate_biohash_metric_mapping.py` (see its docstring for why the
fit is valid without real biometric data) and stored in
`evaluation/results/biohash_metric_calibration.json`; only that curve is used
here. Every estimate carries its calibrated standard deviation.

Directions (never mix them):
- Hamming similarity: higher = more similar (1.0 = identical templates).
- Estimated cosine similarity: higher = more similar.
- Estimated Euclidean distance: LOWER = more similar (0 = identical).

For L2-normalized embeddings (every embedder here returns unit vectors,
`models/common/base_embedder.py`), Euclidean distance and cosine similarity
are two views of the same angle:

    d = sqrt(2 - 2 * cos)        cos = 1 - d**2 / 2

so the Euclidean estimate is derived from the cosine estimate, and a distance
is turned back into a higher-is-better similarity with the inverse formula
(used for score fusion) - no extra fitted parameter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "results" / "biohash_metric_calibration.json"


def fitted_hamming_similarity(cosine: np.ndarray, coefficients: list[float]) -> np.ndarray:
    """The calibration model: expected Hamming similarity of two templates whose embeddings have this cosine.

    1 - h = (p1*theta + p2*theta**2 + p3*theta**3) / pi with theta = arccos(cosine) - exactly 1.0 at cosine 1.0.
    """
    theta = np.arccos(np.clip(np.asarray(cosine, dtype=np.float64), -1.0, 1.0))
    p1, p2, p3 = coefficients
    return 1.0 - (p1 * theta + p2 * theta**2 + p3 * theta**3) / np.pi


@dataclass(frozen=True)
class MetricCurve:
    """One modality's Hamming-similarity -> cosine curve (strictly increasing), tabulated from the fitted model."""

    modality: str
    template_bits: int
    cosine: np.ndarray
    hamming_similarity_mean: np.ndarray
    cosine_estimate_std: np.ndarray

    def estimate_cosine(self, hamming_similarity: float) -> float:
        """Estimated embedding cosine for an observed Hamming similarity.

        Identical templates (similarity 1.0) map to 1.0. Values outside the
        calibrated range are clamped to its ends rather than extrapolated.
        """
        if hamming_similarity >= 1.0:
            return 1.0
        return float(np.interp(hamming_similarity, self.hamming_similarity_mean, self.cosine))

    def estimate_std(self, hamming_similarity: float) -> float:
        """Calibrated standard deviation of `estimate_cosine` around this point."""
        if hamming_similarity >= 1.0:
            return 0.0
        return float(np.interp(hamming_similarity, self.hamming_similarity_mean, self.cosine_estimate_std))

    def hamming_for_cosine(self, cosine: float) -> float:
        """Inverse: the Hamming similarity whose estimate equals `cosine` (the threshold in template space)."""
        return float(np.interp(cosine, self.cosine, self.hamming_similarity_mean))


def euclidean_from_cosine(cosine: float) -> float:
    """Euclidean distance between two unit vectors with this cosine similarity (0 = identical, 2 = opposite)."""
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * cosine)))


def cosine_from_euclidean(distance: float) -> float:
    """Inverse of `euclidean_from_cosine`: a higher-is-better similarity for a unit-vector distance."""
    return 1.0 - distance * distance / 2.0


@lru_cache
def _load(path: str) -> dict:
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8"))


def get_curve(modality: str, template_bits: int, path: Path = CALIBRATION_PATH) -> MetricCurve | None:
    """The calibrated curve for `modality` at `template_bits`, or None if that combination was never calibrated."""
    report = _load(str(path))
    entry = report.get("modalities", {}).get(modality)
    if entry is None or report.get("template_bits") != template_bits:
        return None
    calibrated = np.asarray(entry["cosine"], dtype=np.float64)
    cosine = np.linspace(calibrated[0], 1.0, 2001)
    return MetricCurve(
        modality=modality,
        template_bits=template_bits,
        cosine=cosine,
        hamming_similarity_mean=fitted_hamming_similarity(cosine, entry["coefficients"]),
        cosine_estimate_std=np.interp(cosine, calibrated, np.asarray(entry["cosine_estimate_std"], dtype=np.float64)),
    )


def clear_cache() -> None:
    """Test hook."""
    _load.cache_clear()
