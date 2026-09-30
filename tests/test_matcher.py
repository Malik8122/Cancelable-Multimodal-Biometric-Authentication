"""Offline tests for template_protection/matcher.py.

Hand-built bit vectors only - no keys, no models needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from template_protection.matcher import accept, compare

IDENTICAL = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
ONE_BIT_FLIPPED = np.array([1, 0, 1, 1, 0, 0, 1, 1], dtype=np.uint8)
FULLY_OPPOSITE = np.array([0, 1, 0, 0, 1, 1, 0, 1], dtype=np.uint8)


def test_compare_identical_templates_is_perfect_match():
    assert compare(IDENTICAL, IDENTICAL.copy(), metric="hamming") == 1.0


def test_compare_fully_opposite_templates_is_zero_hamming_similarity():
    assert compare(IDENTICAL, FULLY_OPPOSITE, metric="hamming") == 0.0


def test_compare_hamming_partial_match():
    score = compare(IDENTICAL, ONE_BIT_FLIPPED, metric="hamming")
    assert score == pytest.approx(7 / 8)


def test_compare_cosine_identical_templates_is_one():
    assert compare(IDENTICAL, IDENTICAL.copy(), metric="cosine") == pytest.approx(1.0)


def test_compare_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compare(IDENTICAL, IDENTICAL[:-1], metric="hamming")


def test_compare_rejects_unsupported_metric():
    with pytest.raises(ValueError):
        compare(IDENTICAL, IDENTICAL.copy(), metric="euclidean")


def test_accept_above_threshold():
    assert accept(0.95, threshold=0.9)


def test_accept_below_threshold_is_rejected():
    assert not accept(0.85, threshold=0.9)


def test_accept_at_threshold_boundary_is_accepted():
    assert accept(0.9, threshold=0.9)


def test_accept_rejects_unsupported_metric():
    with pytest.raises(ValueError):
        accept(0.9, threshold=0.9, metric="euclidean")
