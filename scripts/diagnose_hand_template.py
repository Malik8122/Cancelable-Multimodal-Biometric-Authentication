"""DEVELOPMENT-ONLY diagnosis of one user's enrolled Z hand-gesture template (read-only; never writes the database).

Answers "why does a genuine user fail?" with the shipped matching code:

- the enrollment-to-enrollment distances D(S1,S2), D(S1,S3), D(S2,S3) with exactly the authentication path
  (rotation search + DTW on the stored, keyed sequences) and the outlier / medoid analysis;
- the best rotation per pair;
- per feature group, its share of the squared frame differences along the optimal DTW warping path (which groups
  dominate the distance);
- the per-sample capture metadata stored with the template, and the audited distances of the user's recent hand
  authentications;
- the preprocessing signature (template format, feature names / weights, resampled length, smoothing, band, rotation
  search) of the running code next to what the template was made with.

The stored sequences are keyed; they are un-keyed IN MEMORY with the set's key for the group analysis (a
development-only operation - the same one template re-keying performs). Only distances, angles, shares and metadata
are printed - never features or landmarks.

Run: python -m scripts.diagnose_hand_template --user USER-XXXX [--application-id ...]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing import hand_gesture as hg  # noqa: E402

GROUPS = {
    "trajectory": (0, 1), "direction": (2, 3), "speed": (4,), "acceleration": (5,), "orientation": (6, 7),
    "fingers": (8, 9, 10, 11, 12),
}


def dtw_path(a: np.ndarray, b: np.ndarray, band: float) -> tuple[float, list[tuple[int, int]]]:
    """The same recurrence, band and tie-breaking as template_protection/dtw.py, plus the optimal path."""
    n, m = len(a), len(b)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    window = max(int(np.ceil(band * max(n, m))), abs(n - m))
    acc = np.full((n + 1, m + 1), math.inf)
    steps = np.zeros((n + 1, m + 1), dtype=int)
    move = np.zeros((n + 1, m + 1), dtype=int)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        centre = int(round(i * m / n))
        for j in range(max(1, centre - window), min(m, centre + window) + 1):
            options = (acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
            k = int(np.argmin(options))
            prev = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[k]
            acc[i, j] = cost[i - 1, j - 1] + options[k]
            steps[i, j] = steps[prev] + 1
            move[i, j] = k
    path, i, j = [], n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[move[i, j]]
    return float(acc[n, m] / steps[n, m]), path[::-1]


def group_shares(a: np.ndarray, b: np.ndarray, band: float) -> dict[str, float]:
    """Share of each feature group in the squared weighted frame differences along the optimal path."""
    _, path = dtw_path(a, b, band)
    diff2 = np.array([(a[i] - b[j]) ** 2 for i, j in path]).sum(axis=0)
    total = float(diff2.sum()) or 1.0
    return {g: float(diff2[list(cols)].sum()) / total for g, cols in GROUPS.items()}


def best_rotation(candidate: np.ndarray, reference: np.ndarray, band: float, angles) -> tuple[float, float, dict]:
    from template_protection.dtw import dtw_distance

    per_angle = {a: dtw_distance(hg.rotate_features(candidate, a), reference, band) for a in angles}
    angle = min(per_angle, key=per_angle.get)
    return per_angle[angle], angle, per_angle


def signature(settings) -> dict:
    return {
        "template_format": hg.HAND_TEMPLATE_FORMAT, "feature_dim": hg.FEATURE_DIM, "feature_names": list(hg.FEATURE_NAMES),
        "feature_weights": [float(w) for w in hg.FEATURE_WEIGHTS], "resampled_length": hg.RESAMPLED_LENGTH,
        "smoothing_sigma_s": hg.SMOOTHING_SIGMA_S, "dtw_band": settings.hand_gesture_dtw_band,
        "rotation_tolerance_deg": settings.hand_gesture_rotation_tolerance_deg,
        "aggregation": settings.hand_gesture_aggregation, "threshold": settings.hand_gesture_dtw_threshold,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", required=True)
    ap.add_argument("--application-id", default=None)
    a = ap.parse_args()

    from backend.config import get_settings
    from backend.database import crud
    from backend.database.models import AuditLog
    from backend.database.session import get_session_factory
    from backend.services.hand_service import enrollment_analysis, rotation_angles
    from template_protection.hkdf_keys import derive_key
    from template_protection.sequence_transform import deserialize, unprotect

    settings = get_settings()
    app = a.application_id or settings.application_id
    db = get_session_factory()()
    row = next((r for r in crud.get_set_rows(db, a.user, app) if r.modality == "hand" and r.is_active), None)
    if row is None:
        raise SystemExit(f"no active hand template for {a.user} / {app}")
    meta = dict(row.template_metadata or {})
    print(f"user {a.user}  set {row.template_set_version}  key v{row.key_version}  format {row.template_format}")

    running = signature(settings)
    stored = {k: meta.get(k) for k in ("template_format", "feature_dim", "feature_weights", "resampled_length", "dtw_band",
                                       "rotation_tolerance_deg", "aggregation")}
    print("\n== preprocessing signature (running code vs template) ==")
    for k, v in stored.items():
        print(f"  {k:24s} template={v!s:60.60s} running={running[k]!s:.60s}  {'OK' if v == running[k] else 'DIFF'}")
    print(f"  {'smoothing_sigma_s':24s} running={running['smoothing_sigma_s']}  (not stored in the template)")

    print("\n== enrollment samples (stored capture metadata) ==")
    for i in range(meta.get("attempts", 0)):
        print(f"  S{i + 1}: recorded {meta['attempt_durations_s'][i]:.2f} s, movement {meta['attempt_motion_durations_s'][i]:.2f} s, "
              f"frames {meta['attempt_frames_detected'][i]}, tracking {meta['attempt_tracking_fps'][i]} fps")

    key = derive_key(settings.master_secret, application_id=app, user_id=a.user, modality="hand", key_version=row.key_version)
    _, protected = deserialize(row.protected_template)
    band, angles = meta.get("dtw_band", settings.hand_gesture_dtw_band), rotation_angles(settings.hand_gesture_rotation_tolerance_deg)
    plain = unprotect(protected, key)
    weights = hg.FEATURE_WEIGHTS

    print("\n== enrollment-to-enrollment (authentication path: rotation search + DTW, keyed sequences) ==")
    analysis = enrollment_analysis(protected, band, angles, settings.hand_gesture_dtw_threshold)
    for i, j in itertools.combinations(range(len(plain)), 2):
        d_ij, ang_ij, _ = best_rotation(plain[i], plain[j], band, angles)
        d_ji, ang_ji, _ = best_rotation(plain[j], plain[i], band, angles)
        d, ang = (d_ij, ang_ij) if d_ij <= d_ji else (d_ji, -ang_ji)
        d0, _, _ = best_rotation(plain[i], plain[j], band, [0.0])
        shares = group_shares(hg.rotate_features(plain[i], ang), plain[j], band)
        print(f"  S{i + 1}-S{j + 1}: {analysis['matrix'][i, j]:.4f}  (plaintext {d:.4f}; best rotation {ang:+.0f} deg; at 0 deg {d0:.4f})")
        print("         squared-cost share: " + "  ".join(f"{g} {s:5.1%}" for g, s in shares.items()))
    print(f"  medoid S{analysis['medoid'] + 1}; outlier: {analysis['outlier']}")

    print("\n== per-sample shape / rhythm summaries (unweighted, no coordinates) ==")
    for i, s in enumerate(plain):
        f = s.astype(np.float64) / weights
        traj = f[:, :2]
        span = traj.max(axis=0) - traj.min(axis=0)
        orient = np.degrees(np.arctan2(f[:, 7], f[:, 6]))
        shape = hg.z_structure(traj)
        print(f"  S{i + 1}: trajectory box {span[0]:.2f} x {span[1]:.2f} (RMS units, aspect h/w {span[1] / max(span[0], 1e-9):.2f}); "
              f"Z corners at {shape.corners}; speed CV {f[:, 4].std() / max(f[:, 4].mean(), 1e-9):.2f}; "
              f"hand orientation {np.median(orient):+.0f} deg (spread {np.percentile(orient, 90) - np.percentile(orient, 10):.0f})")

    print("\n== audited hand authentications (latest first) ==")
    logs = (db.query(AuditLog).filter(AuditLog.user_id == a.user).order_by(AuditLog.timestamp.desc()).limit(10).all())
    for log in logs:
        h = (log.modality_diagnostics or {}).get("hand")
        if h and h.get("dtw_distances"):
            print(f"  {log.timestamp:%H:%M:%S}  set {log.template_set_version}  per-sample {h['dtw_distances']}  "
                  f"{h.get('aggregation')} {h.get('dtw_distance')}  thr {h.get('threshold')}  {h.get('decision')}  "
                  f"fps {h.get('tracking_fps')}  movement {h.get('motion_duration_s')} s  idle {h.get('idle_trimmed_s')} s")
    print("\n(development-only: distances, angles, shares and metadata only)")
    db.close()
    json.dumps(running)


if __name__ == "__main__":
    main()
