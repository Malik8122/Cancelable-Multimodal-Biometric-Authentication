"""Offline tests for template_protection/biohash.py.

Uses synthetic embeddings (random unit vectors) - no models, no images, no
GPU - matching this repo's "offline test" convention (see tests/conftest.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from template_protection.biohash import DEFAULT_OUTPUT_BITS, TEMPLATE_FORMAT_VERSION, generate_template
from template_protection.hkdf_keys import derive_key
from template_protection.utils import l2_normalize

MASTER_SECRET = "unit-test-master-secret-not-for-production"


def _random_embedding(dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.standard_normal(dim).astype(np.float32))


def _key(**overrides):
    kwargs = dict(application_id="capstone-demo", user_id="U001", modality="face", key_version=1)
    kwargs.update(overrides)
    return derive_key(MASTER_SECRET, **kwargs)


def test_template_format_version_is_a_positive_int():
    assert isinstance(TEMPLATE_FORMAT_VERSION, int)
    assert TEMPLATE_FORMAT_VERSION >= 1


@pytest.mark.parametrize("output_bits,embedding_dim", [(128, 512), (256, 256), (512, 256)])
def test_generate_template_has_requested_length(output_bits, embedding_dim):
    embedding = _random_embedding(embedding_dim)
    template = generate_template(embedding, _key(), output_bits=output_bits)
    assert template.shape == (output_bits,)
    assert set(np.unique(template)).issubset({0, 1})


def test_default_output_bits_matches_module_constant():
    embedding = _random_embedding(512)
    template = generate_template(embedding, _key())
    assert template.shape == (DEFAULT_OUTPUT_BITS,)


def test_same_embedding_and_key_reproduce_the_same_template():
    """Required for authentication: enrollment and a later genuine login must
    independently produce the same protected template from the same embedding."""
    embedding = _random_embedding(512)
    key = _key()
    first = generate_template(embedding, key)
    second = generate_template(embedding, key)
    assert np.array_equal(first, second)


def test_different_embeddings_produce_different_templates():
    key = _key()
    template_a = generate_template(_random_embedding(512, seed=1), key)
    template_b = generate_template(_random_embedding(512, seed=2), key)
    assert not np.array_equal(template_a, template_b)


def test_revocation_key_version_changes_template_substantially():
    """Same embedding, only key_version differs -> near-random (~50%) Hamming distance."""
    embedding = _random_embedding(512)
    template_v1 = generate_template(embedding, _key(key_version=1), output_bits=256)
    template_v2 = generate_template(embedding, _key(key_version=2), output_bits=256)
    hamming_fraction = float(np.mean(template_v1 != template_v2))
    assert not np.array_equal(template_v1, template_v2)
    assert 0.3 < hamming_fraction < 0.7


def test_diversity_different_application_ids_change_template_substantially():
    """Same embedding and user/modality, only application_id differs -> unlinkable templates."""
    embedding = _random_embedding(512)
    template_app1 = generate_template(embedding, _key(application_id="app-one"), output_bits=256)
    template_app2 = generate_template(embedding, _key(application_id="app-two"), output_bits=256)
    hamming_fraction = float(np.mean(template_app1 != template_app2))
    assert not np.array_equal(template_app1, template_app2)
    assert 0.3 < hamming_fraction < 0.7


def test_generate_template_re_normalizes_non_unit_input():
    """Defensive re-normalization: an un-normalized embedding must still yield
    a valid template equal to normalizing it first."""
    unit_embedding = _random_embedding(256)
    scaled_embedding = unit_embedding * 5.0
    key = _key(modality="voice")
    assert np.array_equal(
        generate_template(unit_embedding, key, output_bits=128),
        generate_template(scaled_embedding, key, output_bits=128),
    )
