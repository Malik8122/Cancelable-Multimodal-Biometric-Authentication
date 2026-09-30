"""Tempo-robustness representation experiment (evaluation/hand_representation_experiment.py) and the tempo diagnostics.
SYNTHETIC landmark captures - software tests of the experiment machinery, not evidence for choosing a representation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from evaluation.hand_representation_experiment import VARIANTS, _features, run, synthetic_tempo_dataset
from preprocessing.hand_gesture import motion_sample, parse_capture, process_capture, segment_time_fractions
from tests.hand_signals import Person, polyline_capture, Z_WAYPOINTS


def _motion(payload):
    return motion_sample(parse_capture(json.dumps(payload)))


def test_the_current_variant_is_exactly_the_production_representation():
    for p in range(4):
        capture = parse_capture(json.dumps(Person(p).capture(p, fps=(12, 30)[p % 2])))
        production = process_capture(capture).features.astype(np.float64)
        motion = motion_sample(capture)
        np.testing.assert_allclose(_features(motion.frames, motion.times_ms, "time")[:, :13], production, atol=1e-5)


def test_a_uniformly_slower_z_is_comparable_in_both_resamplings():
    """The same spatial path at constant speed in 1.5 s and 3.5 s: a pure tempo change is absorbed by either
    resampling (residual: the time-based smoothing rounds a fast Z's corners slightly more)."""
    fast, slow = _motion(polyline_capture(Z_WAYPOINTS, duration_s=1.5)), _motion(polyline_capture(Z_WAYPOINTS, duration_s=3.5))
    for resampling in ("time", "arc"):
        a, b = _features(fast.frames, fast.times_ms, resampling), _features(slow.frames, slow.times_ms, resampling)
        assert np.abs(a[:, :2] - b[:, :2]).mean() < 0.05
    a, b = _features(fast.frames, fast.times_ms, "arc"), _features(slow.frames, slow.times_ms, "arc")
    assert np.abs(a[:, 13] - b[:, 13]).mean() < 0.03                 # same relative timing, whatever the duration


def test_arc_points_are_equally_spaced_along_the_path_and_timing_is_monotonic():
    motion = _motion(Person(3).capture(7, profile_jitter=0.3))
    f = _features(motion.frames, motion.times_ms, "arc")
    steps = np.linalg.norm(np.diff(f[:, :2], axis=0), axis=1)
    assert steps.std() / steps.mean() < 0.35                      # ~uniform progress (smoothing / resampling slack)
    assert np.all(np.diff(f[:, 13]) >= -1e-9) and f[0, 13] == pytest.approx(0) and f[-1, 13] == pytest.approx(1)


def test_a_changed_rhythm_moves_the_time_trajectory_but_not_the_arc_trajectory():
    """Same spatial Z, different rhythm (re-timed frames): time resampling puts the same points at different indices
    (large trajectory difference); arc resampling keeps the trajectory and moves the difference into the separate
    timing feature - the mechanism the tempo experiment tests."""
    even = polyline_capture(Z_WAYPOINTS, duration_s=2.0)
    uneven = polyline_capture(Z_WAYPOINTS, duration_s=2.0)
    n = len(uneven["frames"])
    for k, frame in enumerate(uneven["frames"]):
        frame["t"] = round(1000 * 2.0 * (k / (n - 1)) ** 0.6, 3)      # same positions, re-timed
    a, b = _motion(even), _motion(uneven)
    time_diff = np.abs(_features(a.frames, a.times_ms, "time")[:, :2] - _features(b.frames, b.times_ms, "time")[:, :2]).mean()
    fa, fb = _features(a.frames, a.times_ms, "arc"), _features(b.frames, b.times_ms, "arc")
    assert np.abs(fa[:, :2] - fb[:, :2]).mean() < 0.3 * time_diff
    assert np.abs(fa[:, 13] - fb[:, 13]).mean() > 0.05


def test_segment_time_fractions():
    trajectory = np.column_stack([np.linspace(0, 3, 64), np.zeros(64)])   # constant speed along a line
    top, diagonal, bottom = segment_time_fractions(trajectory, (0.3, 0.7))
    assert (top, diagonal, bottom) == pytest.approx((0.3, 0.4, 0.3), abs=0.02)
    meta = process_capture(parse_capture(json.dumps(Person(1).capture(1)))).metadata
    assert meta["segment_time_top"] + meta["segment_time_diagonal"] + meta["segment_time_bottom"] == pytest.approx(1, abs=0.01)
    assert meta["path_length_palms"] > meta["trajectory_extent_palms"] and meta["mean_speed_palms_per_s"] > 0


def test_experiment_runs_end_to_end_with_a_development_and_held_out_split(tmp_path, monkeypatch):
    import evaluation.hand_representation_experiment as experiment

    monkeypatch.setattr(experiment, "RESULTS", tmp_path / "out")
    synthetic_tempo_dataset(tmp_path / "data", participants=4)
    result = run(tmp_path / "data", "SYNTHETIC CALIBRATION", "t", variants=[v for v in VARIANTS if v.name in ("A_current", "G_arc_shape_timing")], workers=1)
    assert [s["variant"] for s in result["summary"]] == ["A_current", "G_arc_shape_timing"]
    for s in result["summary"]:
        assert s["dev_participants"] == 2 and s["test_participants"] == 2
        assert 0 <= s["test_FAR"] <= 1 and 0 <= s["test_FRR"] <= 1 and 0 <= s["test_ROC_AUC"] <= 1
        assert s["test_genuine_n"] == 2 * 10 and s["test_impostor_n"] == 2 * 10   # 10 probes each, 1 other target
    assert {r["by"] for r in result["by_duration"]} == {"duration_s", "normalized_duration"}
    assert {r["session"] for r in result["by_session"]} == {"session_1", "session_2"}
    assert (tmp_path / "out" / "t_summary.csv").read_text(encoding="utf-8").startswith("evidence_label,variant")


def test_authentication_reports_tempo_diagnostics(db_session):
    from backend.config import get_settings
    from backend.services.hand_service import HandGestureService

    service = HandGestureService(get_settings())
    person = Person(21)
    rows, summary = service.enroll_attempts(
        db_session, [parse_capture(json.dumps(person.capture(s, duration_s=2.0))) for s in range(3)], "u", "app")
    assert len(summary["attempt_segment_times"]) == 3 and len(summary["attempt_path_lengths_palms"]) == 3
    result = service.authenticate(db_session, parse_capture(json.dumps(person.capture(9, duration_s=3.0))), "u", "app")
    d = result.diagnostics
    assert d["normalized_duration"] > 1.2                         # drawn ~1.5x slower than the enrolled median
    assert {"path_length_palms", "mean_speed_palms_per_s", "segment_time_top"} <= set(d)


def test_finger_pose_change_diagnostics_and_shadow_distances(db_session):
    """Same Z movement, enrolled with a pointing hand, verified with an open hand: the finger share rises and the
    production distance grows, while the no-finger shadow distance stays put. The shadow 'A_current' distance equals
    the production decision distance (diagnostics never change a decision)."""
    from backend.config import Settings
    from backend.services.hand_service import HandGestureService
    from tests.hand_signals import OPEN_HAND, POINTING_HAND

    service = HandGestureService(Settings(master_secret="unit-test-master-secret-not-for-production", debug_scores=True))
    person = Person(701)
    service.enroll_attempts(db_session, [parse_capture(json.dumps(person.capture(s, finger_extension=POINTING_HAND)))
                                         for s in range(3)], "u", "app")
    same = service.authenticate(db_session, parse_capture(json.dumps(person.capture(50, finger_extension=POINTING_HAND))), "u", "app")
    opened = service.authenticate(db_session, parse_capture(json.dumps(person.capture(50, finger_extension=OPEN_HAND))), "u", "app")
    for result in (same, opened):
        d = result.diagnostics
        assert d["shadow_distances"]["A_current"]["median"] == pytest.approx(result.metric_value, abs=1e-3)
        assert set(d["finger_extension_live"]) == {"thumb", "index", "middle", "ring", "pinky"} and len(d["finger_shares"]) == 3
    share = lambda r: np.mean([sum(s.values()) for s in r.diagnostics["finger_shares"]])
    assert share(opened) > 0.2 > share(same)
    assert opened.metric_value > same.metric_value
    no_fingers = lambda r: r.diagnostics["shadow_distances"]["A_no_fingers"]["median"]
    assert no_fingers(opened) == pytest.approx(no_fingers(same), abs=0.01)
    assert opened.diagnostics["finger_extension_live"]["ring"] > same.diagnostics["finger_extension_live"]["ring"] + 0.3
