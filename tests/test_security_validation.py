"""Offline tests for backend/security_validation.py.

Each check is tested both for the real, correct case (passes) and a
deliberately-corrupted input (fails) - a validation layer that always
reports "ok" isn't actually validating anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.config import Settings
from backend.database.models import ProtectedTemplate
from backend.security_validation import (
    SecurityValidationError,
    assert_valid,
    validate_authentication,
    validate_key_material,
    validate_master_secret,
    validate_protected_template_bits,
    validate_stored_template,
)
from template_protection.biohash import TEMPLATE_FORMAT_VERSION, generate_template
from template_protection.hkdf_keys import KeyMaterial, derive_key
from template_protection.utils import pack_bits

MASTER_SECRET = "unit-test-master-secret-not-for-production"


def _settings(**overrides) -> Settings:
    return Settings(master_secret=MASTER_SECRET, **overrides)


def _real_key() -> KeyMaterial:
    return derive_key(MASTER_SECRET, application_id="app", user_id="U001", modality="face", key_version=1)


def _stored_template(output_bits: int = 128, **overrides) -> ProtectedTemplate:
    key = _real_key()
    embedding = np.random.default_rng(0).standard_normal(512).astype(np.float32)
    bits = generate_template(embedding, key, output_bits=output_bits)
    defaults = dict(
        user_id="U001",
        modality="face",
        application_id="app",
        template_version=TEMPLATE_FORMAT_VERSION,
        key_version=1,
        output_bits=output_bits,
        protected_template=pack_bits(bits),
        is_active=True,
        template_status="ACTIVE",
        template_index=1,
    )
    defaults.update(overrides)
    return ProtectedTemplate(**defaults)


def test_validate_master_secret_accepts_a_real_secret():
    assert validate_master_secret(_settings()) is True


def test_validate_key_material_accepts_real_hkdf_output():
    assert validate_key_material(_real_key()) is True


def test_validate_key_material_rejects_all_zero_seeds():
    corrupted = KeyMaterial(
        projection_seed=b"\x00" * 32, permutation_seed=b"\x00" * 32, threshold_seed=b"\x00" * 32, key_version=1
    )
    assert validate_key_material(corrupted) is False


def test_validate_key_material_rejects_identical_seeds():
    seed = b"\x01" * 32
    corrupted = KeyMaterial(projection_seed=seed, permutation_seed=seed, threshold_seed=seed, key_version=1)
    assert validate_key_material(corrupted) is False


def test_validate_key_material_rejects_truncated_seeds():
    corrupted = KeyMaterial(projection_seed=b"\x01" * 8, permutation_seed=b"\x02" * 32, threshold_seed=b"\x03" * 32, key_version=1)
    assert validate_key_material(corrupted) is False


def test_validate_protected_template_bits_accepts_correct_length():
    key = _real_key()
    embedding = np.random.default_rng(1).standard_normal(512).astype(np.float32)
    bits = generate_template(embedding, key, output_bits=128)
    assert validate_protected_template_bits(bits, expected_output_bits=128) is True


def test_validate_protected_template_bits_rejects_wrong_length():
    bits = np.zeros(64, dtype=np.uint8)
    assert validate_protected_template_bits(bits, expected_output_bits=128) is False


def test_validate_stored_template_accepts_a_well_formed_row():
    assert validate_stored_template(_stored_template()) is True


def test_validate_stored_template_rejects_an_inactive_row():
    assert validate_stored_template(_stored_template(is_active=False)) is False


def test_validate_stored_template_rejects_an_unknown_template_version():
    assert validate_stored_template(_stored_template(template_version=TEMPLATE_FORMAT_VERSION + 1)) is False


def test_validate_stored_template_rejects_a_non_positive_key_version():
    assert validate_stored_template(_stored_template(key_version=0)) is False


def test_validate_stored_template_rejects_a_truncated_blob():
    stored = _stored_template()
    stored.protected_template = stored.protected_template[:-1]
    assert validate_stored_template(stored) is False


def test_validate_authentication_all_checks_pass_for_real_inputs():
    stored = _stored_template()
    key = _real_key()
    embedding = np.random.default_rng(2).standard_normal(512).astype(np.float32)
    candidate_bits = generate_template(embedding, key, output_bits=stored.output_bits)

    report = validate_authentication(settings=_settings(), stored=stored, key=key, candidate_template_bits=candidate_bits)

    assert report.ok is True
    assert report.failed_checks() == []


def test_assert_valid_raises_on_a_failing_report():
    stored = _stored_template(is_active=False)
    key = _real_key()
    candidate_bits = generate_template(
        np.random.default_rng(3).standard_normal(512).astype(np.float32), key, output_bits=stored.output_bits
    )
    report = validate_authentication(settings=_settings(), stored=stored, key=key, candidate_template_bits=candidate_bits)

    with pytest.raises(SecurityValidationError):
        assert_valid(report, modality="face", user_id="U001")


def test_assert_valid_does_not_raise_on_a_passing_report():
    stored = _stored_template()
    key = _real_key()
    candidate_bits = generate_template(
        np.random.default_rng(4).standard_normal(512).astype(np.float32), key, output_bits=stored.output_bits
    )
    report = validate_authentication(settings=_settings(), stored=stored, key=key, candidate_template_bits=candidate_bits)

    assert_valid(report, modality="face", user_id="U001")  # must not raise
