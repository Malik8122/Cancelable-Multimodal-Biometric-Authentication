"""DTW matcher and the keyed sequence transform (template_protection/dtw.py, sequence_transform.py)."""

from __future__ import annotations

import numpy as np
import pytest

from template_protection.dtw import aggregate, dtw_distance
from template_protection.hkdf_keys import derive_key
from template_protection.sequence_transform import deserialize, key_matrix, protect, serialize, unprotect


def _curve(n: int, phase: float = 0.0, warp: float = 0.0) -> np.ndarray:
    t = np.linspace(0, 1, n)
    t = t + warp * np.sin(2 * np.pi * t) / (2 * np.pi)  # non-uniform speed
    return np.column_stack([np.sin(2 * np.pi * t + phase), np.sin(4 * np.pi * t) / 2, np.cos(2 * np.pi * t)])


def test_identical_sequences_have_zero_distance():
    a = _curve(64)
    assert dtw_distance(a, a) == 0.0


def test_near_identical_sequences_are_close():
    a = _curve(64)
    b = a + np.random.default_rng(0).normal(0, 0.01, a.shape)
    assert dtw_distance(a, b) < 0.03


def test_different_trajectories_are_far():
    assert dtw_distance(_curve(64), _curve(64, phase=np.pi)) > 0.5


def test_length_and_speed_variation_are_absorbed():
    """The same shape drawn with a different number of frames and a non-uniform speed stays close."""
    a, b = _curve(64), _curve(90, warp=0.5)
    assert dtw_distance(a, b) < 0.1
    assert dtw_distance(a, b) < dtw_distance(a, _curve(64, phase=np.pi))


def test_dtw_is_symmetric_and_non_negative():
    a, b = _curve(50), _curve(70, warp=0.3)
    assert dtw_distance(a, b) == pytest.approx(dtw_distance(b, a))
    assert dtw_distance(a, b) >= 0


@pytest.mark.parametrize("a, b", [
    (np.zeros((0, 3)), np.zeros((5, 3))),
    (np.zeros((5, 3)), np.zeros((5, 4))),
    (np.zeros(5), np.zeros((5, 1))),
    (np.full((5, 3), np.nan), np.zeros((5, 3))),
])
def test_malformed_sequences_are_rejected(a, b):
    with pytest.raises(ValueError):
        dtw_distance(a, b)


def test_aggregation():
    assert aggregate([0.1, 0.9, 0.2, 0.3, 0.25]) == 0.25  # median: robust to one outlier
    assert aggregate([0.1, 0.3], "mean") == pytest.approx(0.2)
    assert aggregate([0.4, 0.1], "min") == 0.1
    with pytest.raises(ValueError):
        aggregate([])


def _key(version: int):
    return derive_key("unit-test-master-secret-not-for-production", application_id="app", user_id="u", modality="hand",
                      key_version=version)


def test_keyed_transform_is_orthonormal_and_preserves_every_dtw_distance():
    q = key_matrix(_key(1), 12)
    np.testing.assert_allclose(q @ q.T, np.eye(12), atol=1e-10)
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(64, 12)), rng.normal(size=(64, 12))
    pa, pb = protect([a, b], _key(1))
    assert dtw_distance(pa, pb) == pytest.approx(dtw_distance(a, b), rel=1e-5)


def test_different_key_versions_give_unrelated_stored_sequences_and_the_right_key_inverts():
    a = np.random.default_rng(2).normal(size=(64, 12)).astype(np.float32)
    s1, s2 = protect([a], _key(1))[0], protect([a], _key(2))[0]
    assert not np.allclose(s1, s2) and not np.allclose(s1, a)
    np.testing.assert_allclose(unprotect([s1], _key(1))[0], a, atol=1e-5)
    assert not np.allclose(unprotect([s1], _key(2))[0], a, atol=1e-2)


def test_serialization_roundtrip_and_malformed_blobs():
    seqs = [np.random.default_rng(k).normal(size=(64, 12)).astype(np.float32) for k in range(5)]
    version, back = deserialize(serialize(seqs, 1))
    assert version == 1 and all(np.array_equal(x, y) for x, y in zip(seqs, back))
    blob = serialize(seqs, 1)
    for bad in (b"", b"XXXX" + blob[4:], blob[:-4]):
        with pytest.raises(ValueError):
            deserialize(bad)
