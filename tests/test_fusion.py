"""Offline tests for fusion/score_fusion.py::fuse_scores. Pure function, no I/O."""

from __future__ import annotations

import pytest

from fusion.score_fusion import fuse_scores


def test_equal_weighted_average_of_both_modalities():
    result = fuse_scores({"face": 0.90, "voice": 0.95})
    assert result == pytest.approx((0.90 + 0.95) / 2)


def test_single_modality_returns_that_modality_unchanged():
    assert fuse_scores({"face": 0.87}) == pytest.approx(0.87)


def test_two_modalities_equal_weighted():
    result = fuse_scores({"face": 0.80, "voice": 1.0})
    assert result == pytest.approx(0.90)


def test_missing_modality_is_excluded_not_treated_as_zero():
    """The whole point of 'only present modalities count': a single-modality
    fusion must not be dragged down by an unsupplied second modality."""
    with_one = fuse_scores({"face": 0.80})
    with_two_but_a_zero = fuse_scores({"face": 0.80, "voice": 0.0})
    assert with_one != pytest.approx(with_two_but_a_zero)
    assert with_one == pytest.approx(0.80)


def test_custom_weights_are_applied():
    # face weighted 3x voice: (0.6*3 + 0.9*1) / 4 = 0.675
    result = fuse_scores({"face": 0.6, "voice": 0.9}, weights={"face": 3.0, "voice": 1.0})
    assert result == pytest.approx(0.675)


def test_weights_for_modalities_not_present_are_ignored():
    result_without_extra = fuse_scores({"face": 0.6}, weights={"face": 3.0})
    result_with_extra_weight = fuse_scores({"face": 0.6}, weights={"face": 3.0, "voice": 100.0})
    assert result_without_extra == pytest.approx(result_with_extra_weight)


def test_missing_weight_entries_default_to_1_before_renormalization():
    # Only voice given an explicit weight (2.0); face defaults to 1.0.
    # (0.5*1 + 1.0*2) / 3 = 0.8333...
    result = fuse_scores({"face": 0.5, "voice": 1.0}, weights={"voice": 2.0})
    assert result == pytest.approx(2.5 / 3)


def test_is_deterministic():
    scores = {"face": 0.91, "voice": 0.97}
    assert fuse_scores(scores) == fuse_scores(scores)


def test_raises_on_empty_scores():
    with pytest.raises(ValueError):
        fuse_scores({})


def test_raises_when_total_weight_is_not_positive():
    with pytest.raises(ValueError):
        fuse_scores({"face": 0.9, "voice": 0.8}, weights={"face": -1.0, "voice": 1.0})
