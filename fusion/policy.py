"""Fusion policy evaluation: decides `authenticated` from per-modality results.

Pure function over already-computed per-modality scores/decisions - like
`fusion/score_fusion.py`, this has no biometric inference, no database
access, and no knowledge of HTTP. `backend/api/fusion.py` is the only
caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from fusion.config import DEFAULT_WEIGHTED_FLOOR, FusionPolicy
from fusion.score_fusion import fuse_scores


@dataclass(frozen=True)
class FusionDecision:
    authenticated: bool
    fused_score: float
    matched_modalities: list[str]
    failed_modalities: list[str]


def evaluate_fusion_policy(
    scores: dict[str, float],
    individually_authenticated: dict[str, bool],
    policy: FusionPolicy,
    fusion_threshold: float,
    weighted_floor: float = DEFAULT_WEIGHTED_FLOOR,
) -> FusionDecision:
    """Apply `policy` to decide the final `authenticated` boolean.

    `scores`/`individually_authenticated` must only contain modalities the
    caller actually submitted this request - a modality that was never
    provided must never appear here (see backend/api/fusion.py, which only
    populates these from `provided`), so every policy below automatically
    inherits the "omitted modalities never count" rule from its caller
    rather than needing to re-implement it.

    - `ALL_REQUIRED`: every submitted modality's own `authenticated` must be
      True. The new default - fixes the audit finding that a strong match
      could compensate for another modality's individual failure.
    - `AT_LEAST_TWO`: at least two submitted modalities, and at least two of
      them individually authenticated (two of two, or two of three).
    - `WEIGHTED`: the original weighted-average behavior, but any submitted
      modality scoring below `weighted_floor` now vetoes authentication
      outright, regardless of the average.
    """
    if not scores:
        raise ValueError("evaluate_fusion_policy requires at least one modality score.")

    matched = sorted(modality for modality, ok in individually_authenticated.items() if ok)
    failed = sorted(modality for modality, ok in individually_authenticated.items() if not ok)
    fused_score = fuse_scores(scores)

    if policy == FusionPolicy.ALL_REQUIRED:
        authenticated = len(failed) == 0

    elif policy == FusionPolicy.AT_LEAST_TWO:
        if len(scores) < 2:
            raise ValueError(
                f"Fusion policy AT_LEAST_TWO requires at least two submitted modalities, got {sorted(scores)}."
            )
        authenticated = len(matched) >= 2

    elif policy == FusionPolicy.WEIGHTED:
        vetoed_by_floor = any(score < weighted_floor for score in scores.values())
        authenticated = (not vetoed_by_floor) and (fused_score >= fusion_threshold)

    else:
        raise ValueError(f"Unsupported fusion policy: {policy!r}")

    return FusionDecision(
        authenticated=authenticated,
        fused_score=fused_score,
        matched_modalities=matched,
        failed_modalities=failed,
    )
