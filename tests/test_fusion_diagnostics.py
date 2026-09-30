"""Offline tests for fusion/diagnostics.py::build_fusion_diagnostics.

Pure function tests, no HTTP/database - mirrors tests/test_fusion.py and tests/test_fusion_policies.py's
style. `test_all_required_denies_despite_a_fused_score_above_threshold` chains the real
`evaluate_fusion_policy` (fusion/policy.py) into `build_fusion_diagnostics` to prove the
diagnostics object reports the SAME override this project's actual fusion engine already applies
- the exact audit-finding scenario `tests/test_fusion_policies.py::test_all_required_blocks_the_compensatory_averaging_bug`
pins at the policy level, reproduced here at the diagnostics-reporting level.
"""

from __future__ import annotations

from fusion.diagnostics import (
    STATUS_FAILED_BELOW_THRESHOLD,
    STATUS_NOT_PRESENTED,
    STATUS_VERIFIED,
    build_fusion_diagnostics,
)
from fusion.policy import evaluate_fusion_policy
from fusion.config import FusionPolicy


def test_verified_modality_reports_its_real_score_and_threshold():
    diagnostics = build_fusion_diagnostics(
        submitted_modalities=["face"],
        scores={"face": 0.95},
        thresholds={"face": 0.8},
        individually_authenticated={"face": True},
        fused_score=0.95,
        fusion_threshold=0.8,
        policy="ALL_REQUIRED",
        access_granted=True,
    )
    assert diagnostics["face"] == {"score": 0.95, "threshold": 0.8, "verified": True, "status": STATUS_VERIFIED}


def test_failed_modality_reports_below_threshold_status():
    diagnostics = build_fusion_diagnostics(
        submitted_modalities=["voice"],
        scores={"voice": 0.6},
        thresholds={"voice": 0.8},
        individually_authenticated={"voice": False},
        fused_score=0.6,
        fusion_threshold=0.8,
        policy="ALL_REQUIRED",
        access_granted=False,
    )
    assert diagnostics["voice"]["status"] == STATUS_FAILED_BELOW_THRESHOLD
    assert diagnostics["voice"]["verified"] is False
    assert diagnostics["voice"]["score"] == 0.6


def test_not_presented_modality_has_no_fabricated_score():
    """A modality never submitted must never get an invented 0.0 - matches
    fusion/score_fusion.py::fuse_scores's own "never a zero score" rule."""
    diagnostics = build_fusion_diagnostics(
        submitted_modalities=["face"],
        scores={"face": 0.95},
        thresholds={"face": 0.8},
        individually_authenticated={"face": True},
        fused_score=0.95,
        fusion_threshold=0.8,
        policy="ALL_REQUIRED",
        access_granted=True,
    )
    for absent in ("voice",):
        assert diagnostics[absent] == {"score": None, "threshold": None, "verified": None, "status": STATUS_NOT_PRESENTED}


def test_weights_only_cover_submitted_modalities_and_sum_to_one():
    diagnostics = build_fusion_diagnostics(
        submitted_modalities=["face", "voice"],
        scores={"face": 0.9, "voice": 0.8},
        thresholds={"face": 0.8, "voice": 0.8},
        individually_authenticated={"face": True, "voice": True},
        fused_score=0.85,
        fusion_threshold=0.8,
        policy="ALL_REQUIRED",
        access_granted=True,
    )
    assert diagnostics["weights"] == {"face": 0.5, "voice": 0.5}
    assert sum(diagnostics["weights"].values()) == 1.0


def test_top_level_fields_pass_through_the_real_fusion_values_unmodified():
    diagnostics = build_fusion_diagnostics(
        submitted_modalities=["face", "voice"],
        scores={"face": 1.0, "voice": 0.82},
        thresholds={"face": 0.9, "voice": 0.9},
        individually_authenticated={"face": True, "voice": False},
        fused_score=0.91,
        fusion_threshold=0.9,
        policy="ALL_REQUIRED",
        access_granted=False,
    )
    assert diagnostics["fused_score"] == 0.91
    assert diagnostics["threshold"] == 0.9
    assert diagnostics["policy"] == "ALL_REQUIRED"
    assert diagnostics["access_granted"] is False


def test_all_required_denies_despite_a_fused_score_above_threshold():
    """The exact audit-finding scenario, run through the REAL fusion engine
    (fusion/policy.py::evaluate_fusion_policy, no mocking), then reported
    through build_fusion_diagnostics: Face passes, Voice individually
    fails at its own threshold; the plain average (0.91) clears the fusion
    threshold (0.9), but ALL_REQUIRED must still deny - and the diagnostics
    object must show exactly that, not the average's verdict."""
    scores = {"face": 1.0, "voice": 0.82}
    individually_authenticated = {"face": True, "voice": False}
    thresholds = {"face": 0.9, "voice": 0.9}
    fusion_threshold = 0.9

    decision = evaluate_fusion_policy(scores, individually_authenticated, FusionPolicy.ALL_REQUIRED, fusion_threshold)
    assert decision.fused_score > fusion_threshold  # sanity: the average alone would have passed
    assert decision.authenticated is False  # ALL_REQUIRED overrides it

    diagnostics = build_fusion_diagnostics(
        submitted_modalities=["face", "voice"],
        scores=scores,
        thresholds=thresholds,
        individually_authenticated=individually_authenticated,
        fused_score=decision.fused_score,
        fusion_threshold=fusion_threshold,
        policy=FusionPolicy.ALL_REQUIRED.value,
        access_granted=decision.authenticated,
    )

    assert diagnostics["fused_score"] > diagnostics["threshold"]  # the fused number alone looks like a pass
    assert diagnostics["access_granted"] is False  # but the reported decision must still be the real, denied one
    assert diagnostics["face"]["status"] == STATUS_VERIFIED
    assert diagnostics["voice"]["status"] == STATUS_FAILED_BELOW_THRESHOLD
