"""Offline tests for template_protection/revoke.py."""

from __future__ import annotations

import numpy as np

from template_protection.biohash import generate_template
from template_protection.hkdf_keys import derive_key
from template_protection.revoke import revoke_template
from template_protection.utils import l2_normalize

MASTER_SECRET = "unit-test-master-secret-not-for-production"


def _embedding(dim=512, seed=0):
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.standard_normal(dim).astype(np.float32))


def test_revoke_template_matches_deriving_the_new_version_directly():
    embedding = _embedding()
    revoked = revoke_template(
        embedding,
        MASTER_SECRET,
        application_id="capstone-demo",
        user_id="U001",
        modality="face",
        new_key_version=2,
        output_bits=256,
    )
    key_v2 = derive_key(MASTER_SECRET, application_id="capstone-demo", user_id="U001", modality="face", key_version=2)
    expected = generate_template(embedding, key_v2, output_bits=256)
    assert np.array_equal(revoked, expected)


def test_revoke_template_invalidates_the_old_template():
    embedding = _embedding()
    key_v1 = derive_key(MASTER_SECRET, application_id="capstone-demo", user_id="U001", modality="face", key_version=1)
    old_template = generate_template(embedding, key_v1, output_bits=256)

    new_template = revoke_template(
        embedding,
        MASTER_SECRET,
        application_id="capstone-demo",
        user_id="U001",
        modality="face",
        new_key_version=2,
        output_bits=256,
    )

    assert not np.array_equal(old_template, new_template)
    hamming_fraction = float(np.mean(old_template != new_template))
    assert 0.3 < hamming_fraction < 0.7
