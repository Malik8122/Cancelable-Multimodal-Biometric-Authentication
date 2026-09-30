"""Per-modality match decision on top of the protected-template Hamming comparison.

The template comparison is always Hamming similarity of two 256-bit cancelable
templates (`template_protection.matcher.compare`). What differs per modality is
the metric the decision is expressed in:

| modality            | metric (calibrated ESTIMATE)      | rule                               | direction        |
|---------------------|-----------------------------------|------------------------------------|------------------|
| face                | cosine similarity                 | estimate >= face_cosine_threshold  | higher = better  |
| voice               | Euclidean distance (unit vectors) | estimate <= voice_euclidean_threshold | LOWER = better |
| (no calibration)    | Hamming similarity (fallback)     | similarity >= per-modality threshold | higher = better |

Both estimates are monotone functions of the Hamming similarity
(`template_protection/metric_estimation.py`), so each rule is exactly a Hamming
threshold in template space (`hamming_threshold` below) - the stored template
is never compared any other way, and no embedding is stored to make it possible.

Fusion needs every score higher-is-better and on one scale, so each modality
also reports a `fusion_score` on the ESTIMATED-COSINE scale: face's estimate as
is; voice's distance turned back into a similarity with cos = 1 - d**2 / 2
(exact for unit vectors, so the voice threshold 0.90 becomes 1 - 0.90**2 / 2 =
0.595 on that scale). A raw distance is never averaged with a similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.config import Settings
from backend.threshold_loader import get_modality_threshold
from template_protection.metric_estimation import cosine_from_euclidean, euclidean_from_cosine, get_curve

logger = logging.getLogger("backend.services.modality_metrics")

COSINE_ESTIMATE = "cosine_estimate"
EUCLIDEAN_ESTIMATE = "euclidean_estimate"
HAMMING = "hamming"


@dataclass(frozen=True)
class ModalityDecision:
    #: COSINE_ESTIMATE, EUCLIDEAN_ESTIMATE or HAMMING.
    metric: str
    #: The value in the metric's own units (a distance for EUCLIDEAN_ESTIMATE).
    value: float
    #: The configured threshold in the same units.
    threshold: float
    higher_is_better: bool
    matched: bool
    #: Calibrated standard deviation of `value` (0.0 for HAMMING, which is measured, not estimated).
    uncertainty: float
    #: Higher-is-better score on the estimated-cosine scale, used for fusion.
    fusion_score: float
    fusion_threshold: float
    #: The Hamming similarity this decision is equivalent to (template-space threshold).
    hamming_threshold: float


def _hamming_rule_threshold(modality: str, settings: Settings) -> float:
    return get_modality_threshold(modality, settings.match_threshold)


def decide(modality: str, hamming_similarity: float, template_bits: int, settings: Settings) -> ModalityDecision:
    """The match decision for one modality from its template Hamming similarity."""
    curve = get_curve(modality, template_bits)

    if modality == "face" and curve is not None:
        threshold = settings.face_cosine_threshold
        cosine = curve.estimate_cosine(hamming_similarity)
        return ModalityDecision(
            metric=COSINE_ESTIMATE,
            value=cosine,
            threshold=threshold,
            higher_is_better=True,
            matched=cosine >= threshold,
            uncertainty=curve.estimate_std(hamming_similarity),
            fusion_score=cosine,
            fusion_threshold=threshold,
            hamming_threshold=curve.hamming_for_cosine(threshold),
        )

    if modality == "voice" and curve is not None:
        threshold = settings.voice_euclidean_threshold
        cosine = curve.estimate_cosine(hamming_similarity)
        distance = euclidean_from_cosine(cosine)
        spread = curve.estimate_std(hamming_similarity)
        return ModalityDecision(
            metric=EUCLIDEAN_ESTIMATE,
            value=distance,
            threshold=threshold,
            higher_is_better=False,
            matched=distance <= threshold,
            # Half the distance range covered by cosine +/- one standard deviation.
            uncertainty=(euclidean_from_cosine(cosine - spread) - euclidean_from_cosine(min(1.0, cosine + spread))) / 2,
            fusion_score=cosine_from_euclidean(distance),
            fusion_threshold=cosine_from_euclidean(threshold),
            hamming_threshold=curve.hamming_for_cosine(cosine_from_euclidean(threshold)),
        )

    if modality in ("face", "voice"):
        logger.warning(
            "No %d-bit metric calibration for modality=%s (run scripts/calibrate_biohash_metric_mapping.py) - "
            "falling back to the Hamming-similarity rule.", template_bits, modality,
        )

    threshold = _hamming_rule_threshold(modality, settings)
    return ModalityDecision(
        metric=HAMMING,
        value=hamming_similarity,
        threshold=threshold,
        higher_is_better=True,
        matched=hamming_similarity >= threshold,
        uncertainty=0.0,
        fusion_score=curve.estimate_cosine(hamming_similarity) if curve else hamming_similarity,
        fusion_threshold=curve.estimate_cosine(threshold) if curve else threshold,
        hamming_threshold=threshold,
    )


def fusion_threshold(modality: str, template_bits: int, settings: Settings) -> float:
    """This modality's threshold on the fusion scale (used when nothing is enrolled to compare)."""
    return decide(modality, 0.0, template_bits, settings).fusion_threshold
