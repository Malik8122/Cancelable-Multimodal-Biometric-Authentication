"""Runtime security invariant checks, run on every authentication attempt.

Distinct from a normal "no match" (which is an expected, everyday outcome
with `authenticated=False`): everything checked here is a *should-never-
happen* structural invariant about the cryptographic/template-protection
pipeline itself - if one of these is ever false, something is wrong with the
backend's own state (a corrupted row, a config drift, a code regression),
not with the caller's biometric sample. `ModalityService.authenticate`
(backend/services/base_service.py) runs this before trusting a comparison
result; a failure here always fails closed (never authenticates), and is
mapped to a 500 by `backend/main.py`'s exception handler rather than a
normal 200/`authenticated: false` - the caller should be able to tell "you
weren't recognized" apart from "the backend's security invariants broke".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.config import Settings
from backend.database.models import STATUS_ACTIVE, STATUS_REVOKED, STATUS_STANDBY, TEMPLATE_STATUSES, ProtectedTemplate
from template_protection.biohash import TEMPLATE_FORMAT_VERSION
from template_protection.hkdf_keys import KeyMaterial, SEED_LENGTH_BYTES


class SecurityValidationError(RuntimeError):
    """A required security invariant did not hold; the caller must not be authenticated."""


@dataclass(frozen=True)
class SecurityValidationReport:
    """One row per check, so a failure is diagnosable rather than a bare bool."""

    checks: dict[str, bool]

    @property
    def ok(self) -> bool:
        return all(self.checks.values())

    def failed_checks(self) -> list[str]:
        return [name for name, passed in self.checks.items() if not passed]


def validate_master_secret(settings: Settings) -> bool:
    """A non-empty secret was actually loaded - `Settings.master_secret` has
    no default (backend/config.py), so this mainly guards against it somehow
    being set to an empty string via `.env`."""
    return bool(settings.master_secret) and len(settings.master_secret) > 0


def validate_key_material(key: KeyMaterial) -> bool:
    """HKDF actually ran and produced three independent, full-length seeds
    (see template_protection/hkdf_keys.py::derive_key) - not all-zero, not
    truncated, not accidentally identical to each other."""
    seeds = (key.projection_seed, key.permutation_seed, key.threshold_seed)
    if any(len(seed) != SEED_LENGTH_BYTES for seed in seeds):
        return False
    if any(seed == b"\x00" * SEED_LENGTH_BYTES for seed in seeds):
        return False
    return len({seeds[0], seeds[1], seeds[2]}) == 3


def validate_protected_template_bits(template_bits: np.ndarray, expected_output_bits: int) -> bool:
    """BioHash actually ran and produced a bit vector of the expected length
    (see template_protection/biohash.py::generate_template) - catches a
    silently-mismatched `output_bits` before it corrupts a comparison."""
    return template_bits.ndim == 1 and template_bits.shape[0] == expected_output_bits


def validate_stored_template(stored: ProtectedTemplate) -> bool:
    """The row `crud.get_active_template` returned is actually usable:
    active, a template-format version this codebase knows how to unpack, a
    positive key_version, and a protected_template blob whose packed byte
    length is consistent with its own declared `output_bits` (see
    template_protection/utils.py::pack_bits - `ceil(output_bits / 8)` bytes)."""
    if not stored.is_active or stored.template_status != STATUS_ACTIVE:
        return False
    if stored.template_version != TEMPLATE_FORMAT_VERSION:
        return False
    if stored.key_version < 1:
        return False
    expected_bytes = (stored.output_bits + 7) // 8
    return len(stored.protected_template) == expected_bytes


def validate_template_pool(
    pool: list[ProtectedTemplate], *, stored: ProtectedTemplate, key: KeyMaterial, pool_size: int
) -> dict[str, bool]:
    """Template-set invariants for one (user, application).

    `pool` is every row of that context (all sets, all modalities, any
    status); `stored` is the ACTIVE row being authenticated against. Checks:

    - `exactly_one_active_set`: every `is_active` row belongs to one single
      set, that set is ACTIVE, `stored` is one of them, each modality has at
      most one active row, and no ACTIVE-status row sits outside it. This is
      what guarantees authentication never mixes templates from different sets.
    - `revoked_never_active`: no REVOKED row - and no row of a REVOKED set - is active.
    - `standby_unique`: live templates have pairwise distinct bytes, and
      distinct key versions per modality.
    - `pool_size_correct`: at most `pool_size` live (ACTIVE + STANDBY) sets.
    - `set_status_consistent`: every live row's status follows its set's
      status (ACTIVE set -> ACTIVE rows, STANDBY set -> STANDBY rows).
    - `key_version_matches_template`: the key derived for this attempt is the
      stored row's key version, and its set version is valid.
    """
    live = [row for row in pool if row.template_status != STATUS_REVOKED]
    active_rows = [row for row in pool if row.is_active]
    active_set_versions = {row.template_set_version for row in active_rows}
    active_status_versions = {row.template_set_version for row in pool if row.template_set_status == STATUS_ACTIVE}
    live_set_versions = {row.template_set_version for row in pool if row.template_set_status != STATUS_REVOKED}
    modality_counts: dict[str, int] = {}
    for row in active_rows:
        modality_counts[row.modality] = modality_counts.get(row.modality, 0) + 1

    return {
        "exactly_one_active_set": (
            len(active_set_versions) == 1
            and active_set_versions == active_status_versions
            and stored.template_set_version in active_set_versions
            and stored.template_id in {row.template_id for row in active_rows}
            and all(count == 1 for count in modality_counts.values())
            and all(row.template_status in TEMPLATE_STATUSES for row in pool)
            and all((row.template_status == STATUS_ACTIVE) == bool(row.is_active) for row in pool)
        ),
        "revoked_never_active": not any(
            row.is_active and (row.template_status == STATUS_REVOKED or row.template_set_status == STATUS_REVOKED)
            for row in pool
        ),
        "standby_unique": (
            len({bytes(row.protected_template) for row in live}) == len(live)
            and len({(row.modality, row.key_version) for row in live}) == len(live)
        ),
        "pool_size_correct": len(live_set_versions) <= max(pool_size, 1),
        "set_status_consistent": all(
            row.template_set_status == STATUS_REVOKED
            or row.template_status == row.template_set_status
            for row in live
        ),
        "key_version_matches_template": key.key_version == stored.key_version and (stored.template_set_version or 0) >= 1,
    }


def validate_authentication(
    *,
    settings: Settings,
    stored: ProtectedTemplate,
    key: KeyMaterial,
    candidate_template_bits: np.ndarray,
    pool: list[ProtectedTemplate] | None = None,
) -> SecurityValidationReport:
    """Run every check for one authentication attempt against `stored`.

    Called from `ModalityService.authenticate` after the candidate template
    has been generated but before its score is trusted - see
    backend/services/base_service.py.
    """
    checks = {
        "master_secret_loaded": validate_master_secret(settings),
        "hkdf_key_material_valid": validate_key_material(key),
        "biohash_template_valid": validate_protected_template_bits(candidate_template_bits, stored.output_bits),
        "stored_template_valid": validate_stored_template(stored),
    }
    if pool is not None:
        checks.update(validate_template_pool(pool, stored=stored, key=key, pool_size=settings.template_pool_size))
    return SecurityValidationReport(checks=checks)


def validate_stored_hand_template(stored: ProtectedTemplate) -> bool:
    """The ACTIVE hand-gesture row is usable: active, the hand template format/version this codebase compares, a
    positive key version, and a blob that deserializes to (attempts, RESAMPLED_LENGTH, FEATURE_DIM) finite values."""
    from preprocessing.hand_gesture import FEATURE_DIM, HAND_TEMPLATE_FORMAT, HAND_TEMPLATE_VERSION, RESAMPLED_LENGTH
    from template_protection.sequence_transform import deserialize

    if not stored.is_active or stored.template_status != STATUS_ACTIVE:
        return False
    if stored.template_format != HAND_TEMPLATE_FORMAT or stored.template_version != HAND_TEMPLATE_VERSION:
        return False
    if stored.key_version < 1:
        return False
    try:
        version, sequences = deserialize(stored.protected_template)
    except ValueError:
        return False
    return version == HAND_TEMPLATE_VERSION and all(
        seq.shape == (RESAMPLED_LENGTH, FEATURE_DIM) and np.all(np.isfinite(seq)) for seq in sequences
    )


def validate_hand_authentication(
    *,
    settings: Settings,
    stored: ProtectedTemplate,
    key: KeyMaterial,
    candidate: np.ndarray,
    pool: list[ProtectedTemplate] | None = None,
) -> SecurityValidationReport:
    """The hand-gesture counterpart of `validate_authentication` (backend/services/hand_service.py): the same secret,
    key and template-pool invariants; the candidate must be a finite (RESAMPLED_LENGTH, FEATURE_DIM) sequence."""
    from preprocessing.hand_gesture import FEATURE_DIM, RESAMPLED_LENGTH

    checks = {
        "master_secret_loaded": validate_master_secret(settings),
        "hkdf_key_material_valid": validate_key_material(key),
        "hand_candidate_valid": candidate.shape == (RESAMPLED_LENGTH, FEATURE_DIM) and bool(np.all(np.isfinite(candidate))),
        "stored_template_valid": validate_stored_hand_template(stored),
    }
    if pool is not None:
        checks.update(validate_template_pool(pool, stored=stored, key=key, pool_size=settings.template_pool_size))
    return SecurityValidationReport(checks=checks)


def assert_valid(report: SecurityValidationReport, *, modality: str, user_id: str) -> None:
    """Raise `SecurityValidationError` if any check failed - the fail-closed entry point."""
    if not report.ok:
        raise SecurityValidationError(
            f"Security validation failed for modality={modality!r} user_id={user_id!r}: "
            f"{', '.join(report.failed_checks())}"
        )
