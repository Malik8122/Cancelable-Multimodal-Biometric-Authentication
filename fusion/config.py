"""Fusion decision policies.

The audit finding this fixes: `fuse_scores`'s plain weighted average let one
strong modality compensate for another that individually failed its own
threshold - e.g. Face=1.0 (pass) + Voice=0.82 (individually a fail at
threshold=0.9) averaged to 0.91, which cleared the fusion threshold even
though Voice's own result said "no match". `fusion/policy.py` is where
that decision actually gets made now; this module only defines the available
policies and their tunable parameters.
"""

from __future__ import annotations

from enum import Enum


class FusionPolicy(str, Enum):
    #: Every submitted modality must individually pass its own threshold.
    #: The new default - the secure, "AND" behavior a multi-factor system is
    #: generally expected to have.
    ALL_REQUIRED = "ALL_REQUIRED"

    #: At least two of the submitted modalities must individually pass their own acceptance rule, and at least two
    #: modalities must be submitted (fewer -> rejected as a request error). Face + Voice -> both must pass; Face +
    #: Voice + Hand -> any two. Like every policy, it only runs when every submitted modality was actually compared: a
    #: capture error or failed quality gate in any submitted modality still denies the attempt without fusion (fail
    #: closed, backend/services/authentication.py). Reintroduced with the third modality (hand gesture).
    AT_LEAST_TWO = "AT_LEAST_TWO"

    #: The original weighted-average behavior (`fusion/score_fusion.py::fuse_scores`),
    #: but no longer unconditionally: any submitted modality scoring below
    #: `weighted_floor` vetoes authentication regardless of the average, so a
    #: strong modality can no longer fully compensate for a very weak one.
    WEIGHTED = "WEIGHTED"


#: Policy used when a request doesn't specify one. Deliberately the
#: strictest option - the bug this sprint fixes was `WEIGHTED` being the
#: *only* option, so the safe default changes rather than staying opt-in.
DEFAULT_FUSION_POLICY = FusionPolicy.ALL_REQUIRED

#: WEIGHTED mode's per-modality minimum: any present modality scoring below
#: this vetoes authentication before the average is even considered. Fusion
#: scores are on the estimated-cosine scale (backend/services/modality_metrics.py),
#: where 0.0 means "no better than chance": unrelated embeddings have cosine ~0,
#: which is the Hamming-similarity midpoint ~0.5 this floor used to be expressed
#: in. A conservative floor a real deployment should tighten once per-modality
#: genuine/impostor calibration data exists.
DEFAULT_WEIGHTED_FLOOR = 0.0

ALL_POLICIES: tuple[str, ...] = tuple(policy.value for policy in FusionPolicy)
