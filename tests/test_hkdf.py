"""Offline tests for template_protection/hkdf_keys.py.

No network, no filesystem, no GPU - HKDF derivation is pure computation over
caller-supplied bytes/strings.
"""

from __future__ import annotations

from template_protection.hkdf_keys import SEED_LENGTH_BYTES, derive_key

MASTER_SECRET = "unit-test-master-secret-not-for-production"


def _derive(**overrides):
    kwargs = dict(application_id="capstone-demo", user_id="U001", modality="face", key_version=1)
    kwargs.update(overrides)
    return derive_key(MASTER_SECRET, **kwargs)


def test_derive_key_is_deterministic():
    first = _derive()
    second = _derive()
    assert first.projection_seed == second.projection_seed
    assert first.permutation_seed == second.permutation_seed
    assert first.threshold_seed == second.threshold_seed


def test_seeds_are_fixed_length():
    key = _derive()
    assert len(key.projection_seed) == SEED_LENGTH_BYTES
    assert len(key.permutation_seed) == SEED_LENGTH_BYTES
    assert len(key.threshold_seed) == SEED_LENGTH_BYTES


def test_the_three_seeds_are_mutually_independent():
    key = _derive()
    assert key.projection_seed != key.permutation_seed
    assert key.projection_seed != key.threshold_seed
    assert key.permutation_seed != key.threshold_seed


def test_different_user_id_changes_key_material():
    baseline = _derive()
    other_user = _derive(user_id="U002")
    assert other_user.projection_seed != baseline.projection_seed
    assert other_user.permutation_seed != baseline.permutation_seed
    assert other_user.threshold_seed != baseline.threshold_seed


def test_different_modality_changes_key_material():
    baseline = _derive()
    other_modality = _derive(modality="voice")
    assert other_modality.projection_seed != baseline.projection_seed


def test_different_application_id_changes_key_material_diversity():
    """Diversity property: same user+modality, different application -> different key."""
    baseline = _derive()
    other_app = _derive(application_id="other-app")
    assert other_app.projection_seed != baseline.projection_seed
    assert other_app.permutation_seed != baseline.permutation_seed


def test_different_key_version_changes_key_material_revocation():
    """Revocation property: bumping key_version alone must change the key."""
    baseline = _derive(key_version=1)
    rotated = _derive(key_version=2)
    assert rotated.projection_seed != baseline.projection_seed
    assert rotated.permutation_seed != baseline.permutation_seed
    assert rotated.threshold_seed != baseline.threshold_seed
    assert rotated.key_version == 2


def test_different_master_secret_changes_key_material():
    baseline = _derive()
    different_secret = derive_key(
        "a-completely-different-master-secret",
        application_id="capstone-demo",
        user_id="U001",
        modality="face",
        key_version=1,
    )
    assert different_secret.projection_seed != baseline.projection_seed


def test_accepts_master_secret_as_bytes_or_str_identically():
    from_str = _derive()
    from_bytes = derive_key(
        MASTER_SECRET.encode("utf-8"),
        application_id="capstone-demo",
        user_id="U001",
        modality="face",
        key_version=1,
    )
    assert from_str.projection_seed == from_bytes.projection_seed
    assert from_str.permutation_seed == from_bytes.permutation_seed
    assert from_str.threshold_seed == from_bytes.threshold_seed
