"""Read-only fusion diagnostics: reshapes values ALREADY computed by `fusion/score_fusion.py`
and `fusion/policy.py` into one object for local-development observability (the DEBUG_SCORES-gated
`fusion_diagnostics` field on the authentication response - see
`backend/services/authentication.py::to_public_response`).

Computes nothing new. Every score, weight, threshold, and decision surfaced here is read from
what the real fusion engine already produced for this exact request - this module never
recalculates a score, re-derives a threshold, or makes its own access decision. Like the rest of
`fusion/`, it has no biometric inference, no database access, and no knowledge of HTTP.

Status values, and what they mean given how `backend/services/authentication.py::authenticate_samples`
is actually structured:

- `"verified"`: the modality was submitted and its own score met its own threshold.
- `"failed_below_threshold"`: the modality was submitted, produced a real score, and that score
  did not meet its own threshold.
- `"not_presented"`: the modality was not part of this request at all - never a fabricated score
  of `0.0` (`fusion/score_fusion.py::fuse_scores` never treats an absent modality as a zero
  either; this module must not silently imply otherwise for the same reason).

A submitted modality without a score is a `"capture_error"`: its sample could not be processed (e.g.
no face detected, undecodable or silent audio - backend/services/failure_policy.py). That attempt is
denied without fusion (fail closed); an unenrolled modality is a 409 and a security-validation
failure a 500, so neither ever reaches this module.
"""

from __future__ import annotations

from fusion.score_fusion import resolve_normalized_weights

#: The modalities this project's fusion engine ever combines (backend/services/authentication.py).
ALL_MODALITIES: tuple[str, ...] = ("face", "voice", "hand")

STATUS_VERIFIED = "verified"
STATUS_FAILED_BELOW_THRESHOLD = "failed_below_threshold"
STATUS_NOT_PRESENTED = "not_presented"
STATUS_CAPTURE_ERROR = "capture_error"


def build_fusion_diagnostics(
    *,
    submitted_modalities: list[str],
    scores: dict[str, float],
    thresholds: dict[str, float],
    individually_authenticated: dict[str, bool],
    fused_score: float,
    fusion_threshold: float,
    policy: str,
    access_granted: bool,
) -> dict:
    """Assemble the `fusion_diagnostics` object for one authentication response.

    `scores`, `thresholds`, and `individually_authenticated` must already be the exact
    per-modality values `fusion/policy.py::evaluate_fusion_policy` used to reach `access_granted`
    for this request - this function only labels and reshapes them. `fused_score`,
    `fusion_threshold`, and `policy` are likewise the real values that decision was made with,
    not recomputed here.
    """
    weights = resolve_normalized_weights(scores) if scores else {}

    modalities: dict[str, dict] = {}
    for modality in ALL_MODALITIES:
        if modality not in submitted_modalities:
            modalities[modality] = {"score": None, "threshold": None, "verified": None, "status": STATUS_NOT_PRESENTED}
            continue
        if modality not in scores:
            modalities[modality] = {"score": None, "threshold": None, "verified": False, "status": STATUS_CAPTURE_ERROR}
            continue
        verified = individually_authenticated.get(modality, False)
        modalities[modality] = {
            "score": scores.get(modality),
            "threshold": thresholds.get(modality),
            "verified": verified,
            "status": STATUS_VERIFIED if verified else STATUS_FAILED_BELOW_THRESHOLD,
        }

    return {
        **modalities,
        "weights": weights,
        "fused_score": fused_score,
        "threshold": fusion_threshold,
        "policy": policy,
        "access_granted": access_granted,
    }
