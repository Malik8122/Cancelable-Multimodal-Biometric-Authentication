"""Offline tests for template_protection/transform.py and utils.py.

Pure numpy math - no keys, no models, no GPU. Hand-built seeds and vectors
only.
"""

from __future__ import annotations

import numpy as np
import pytest

from template_protection.transform import (
    apply_permutation,
    build_orthonormal_projection,
    project,
    quantize,
)
from template_protection.utils import constant_time_equals, l2_normalize, pack_bits, seed_to_uint64, unpack_bits

SEED_A = b"\x01" * 32
SEED_B = b"\x02" * 32


def test_l2_normalize_produces_unit_vector():
    vector = np.array([3.0, 4.0], dtype=np.float32)
    normalized = l2_normalize(vector)
    assert np.isclose(np.linalg.norm(normalized), 1.0)


def test_l2_normalize_handles_zero_vector():
    zero = np.zeros(4, dtype=np.float32)
    assert np.array_equal(l2_normalize(zero), zero)


def test_seed_to_uint64_is_deterministic_and_in_range():
    value = seed_to_uint64(SEED_A)
    assert value == seed_to_uint64(SEED_A)
    assert 0 <= value < 2**64
    assert seed_to_uint64(SEED_A) != seed_to_uint64(SEED_B)


def test_pack_unpack_bits_roundtrip():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
    packed = pack_bits(bits)
    assert isinstance(packed, bytes)
    restored = unpack_bits(packed, num_bits=len(bits))
    assert np.array_equal(restored, bits)


def test_constant_time_equals():
    assert constant_time_equals(b"abc", b"abc")
    assert not constant_time_equals(b"abc", b"abd")
    assert not constant_time_equals(b"abc", b"abcd")


@pytest.mark.parametrize("output_bits,embedding_dim", [(128, 512), (256, 256), (512, 256)])
def test_build_orthonormal_projection_shape(output_bits, embedding_dim):
    matrix = build_orthonormal_projection(SEED_A, output_bits, embedding_dim)
    assert matrix.shape == (output_bits, embedding_dim)


def test_build_orthonormal_projection_is_deterministic():
    first = build_orthonormal_projection(SEED_A, 128, 256)
    second = build_orthonormal_projection(SEED_A, 128, 256)
    assert np.array_equal(first, second)


def test_build_orthonormal_projection_differs_per_seed():
    first = build_orthonormal_projection(SEED_A, 128, 256)
    second = build_orthonormal_projection(SEED_B, 128, 256)
    assert not np.array_equal(first, second)


def test_projection_rows_are_orthonormal_within_a_block():
    """output_bits <= embedding_dim: the whole matrix is one block, fully orthonormal."""
    matrix = build_orthonormal_projection(SEED_A, output_bits=128, embedding_dim=256)
    gram = matrix @ matrix.T
    assert np.allclose(gram, np.eye(128), atol=1e-8)


def test_projection_rows_are_unit_norm_even_when_output_bits_exceeds_embedding_dim():
    """output_bits > embedding_dim: cross-block orthogonality isn't guaranteed, but
    every row must still be a unit vector (see build_orthonormal_projection's
    documented block-composition limitation)."""
    matrix = build_orthonormal_projection(SEED_A, output_bits=512, embedding_dim=256)
    row_norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(row_norms, 1.0, atol=1e-8)


def test_project_bounded_by_cauchy_schwarz():
    rng = np.random.default_rng(0)
    embedding = l2_normalize(rng.standard_normal(256).astype(np.float32))
    matrix = build_orthonormal_projection(SEED_A, output_bits=128, embedding_dim=256)
    projected = project(embedding, matrix)
    assert np.all(np.abs(projected) <= 1.0 + 1e-6)


def test_quantize_is_deterministic_and_binary():
    projected = np.array([0.5, -0.3, 0.9, -0.9, 0.0])
    bits = quantize(projected, SEED_A)
    assert set(np.unique(bits)).issubset({0, 1})
    assert np.array_equal(bits, quantize(projected, SEED_A))


def test_quantize_differs_per_seed():
    projected = np.array([0.5, -0.3, 0.9, -0.9, 0.0])
    bits_a = quantize(projected, SEED_A)
    bits_b = quantize(projected, SEED_B)
    assert not np.array_equal(bits_a, bits_b)


def test_apply_permutation_is_a_reordering():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    permuted = apply_permutation(bits, SEED_A)
    assert sorted(permuted.tolist()) == sorted(bits.tolist())


def test_apply_permutation_is_deterministic_and_seed_dependent():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    assert np.array_equal(apply_permutation(bits, SEED_A), apply_permutation(bits, SEED_A))
    assert not np.array_equal(apply_permutation(bits, SEED_A), apply_permutation(bits, SEED_B))
