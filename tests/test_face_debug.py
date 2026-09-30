"""Unit tests for the TEMPORARY backend/services/face_debug.py diagnostics (alignment
investigation). Pure function tests, synthetic vectors, no HTTP/database - mirrors
tests/test_fusion_diagnostics.py's style.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from backend.services.face_debug import log_authentication_diagnostics, log_enrollment_pose_diagnostics

FLOAT_TOKEN = re.compile(r"-?\d+\.\d+")


def _unit(rng, dim=512):
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def test_enrollment_diagnostics_log_matches_manual_pairwise_and_centroid_cosine(caplog):
    rng = np.random.default_rng(1)
    poses = [_unit(rng) for _ in range(5)]
    centroid = poses[0] * 0.6 + poses[1] * 0.4  # any vector to measure against, doesn't need to be a real centroid here
    centroid = centroid / np.linalg.norm(centroid)

    expected_pairs = [float(np.dot(poses[i], poses[j])) for i in range(5) for j in range(i + 1, 5)]
    expected_centroid_sims = [float(np.dot(p, centroid)) for p in poses]

    with caplog.at_level("INFO", logger="backend.face_debug"):
        log_enrollment_pose_diagnostics(poses, centroid)

    message = caplog.records[-1].getMessage()
    assert "[FACE DEBUG]" in message and "stage=enrollment" in message
    assert f"mean={np.mean(expected_pairs):.4f}" in message
    assert f"min={np.min(expected_pairs):.4f}" in message
    assert f"max={np.max(expected_pairs):.4f}" in message
    assert f"mean={np.mean(expected_centroid_sims):.4f}" in message
    assert "enrollment_pose_similarity" in message
    assert "enrollment_centroid_similarity" in message


def test_enrollment_diagnostics_handles_a_single_pose_without_crashing(caplog):
    rng = np.random.default_rng(2)
    pose = _unit(rng)
    with caplog.at_level("INFO", logger="backend.face_debug"):
        log_enrollment_pose_diagnostics([pose], pose)
    message = caplog.records[-1].getMessage()
    assert "enrollment_pose_similarity: mean=nan" in message  # no pairs to compare
    assert "enrollment_centroid_similarity" in message


def test_authentication_diagnostics_reports_the_real_hamming_score_and_marks_cosine_unavailable(caplog):
    with caplog.at_level("INFO", logger="backend.face_debug"):
        log_authentication_diagnostics(0.7422)

    message = caplog.records[-1].getMessage()
    assert "[FACE DEBUG]" in message and "stage=authentication" in message
    assert "protected_hamming_similarity: 0.7422" in message
    assert "live_vs_centroid_cosine: unavailable" in message
    # Must not silently fabricate a number for the unavailable field.
    assert not re.search(r"live_vs_centroid_cosine:\s*-?\d", message)


def test_no_log_ever_contains_more_than_a_handful_of_scalar_values():
    """Sanity check against the failure mode this whole feature must avoid: a raw 512-d embedding
    dumped into a log line would show up as hundreds of float tokens - a scalar diagnostic line
    never should."""
    import logging

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("backend.face_debug")
    handler = _Capture()
    logger.addHandler(handler)
    try:
        rng = np.random.default_rng(3)
        poses = [_unit(rng) for _ in range(5)]
        centroid = _unit(rng)
        log_enrollment_pose_diagnostics(poses, centroid)
        log_authentication_diagnostics(0.81)
    finally:
        logger.removeHandler(handler)

    for message in records:
        float_count = len(FLOAT_TOKEN.findall(message))
        assert float_count <= 10, f"log message looks like it may contain vector data ({float_count} float tokens): {message!r}"


def test_no_log_contains_a_test_secret_value():
    """Never logs anything resembling MASTER_SECRET, even incidentally."""
    import logging

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("backend.face_debug")
    handler = _Capture()
    logger.addHandler(handler)
    try:
        rng = np.random.default_rng(4)
        poses = [_unit(rng) for _ in range(5)]
        log_enrollment_pose_diagnostics(poses, _unit(rng))
        log_authentication_diagnostics(0.75)
    finally:
        logger.removeHandler(handler)

    forbidden = "unit-test-master-secret-not-for-production"
    for message in records:
        assert forbidden not in message
