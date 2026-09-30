"""Z hand gesture - tempo-robustness representation experiment (EXPERIMENTAL; does not change the production pipeline).

Question: is the production representation (uniform TIME resampling + relative speed / acceleration features) too
sensitive to how fast the Z is drawn, and does a tempo-robust representation separate genuine from impostor Z's better?

Every variant is computed from exactly the same gated input as production (`preprocessing.hand_gesture.motion_sample`
plus the production Z check), and matched exactly like production: rotation search (0, +/-6, +/-12 deg), DTW (band 0.2,
path-length normalized), median over the three enrolled samples. The keyed orthonormal transform is omitted because it
leaves every DTW distance unchanged (tests/test_hand_dtw.py).

Resampling:
  time  64 points uniform in TIME (production). A slow and a fast Z with a different speed profile put the same
        spatial point at different indices; DTW must absorb that, and the speed features differ everywhere.
  arc   64 points uniform in PATH LENGTH (normalized trajectory progress 0..1). A slow and a fast Z with the same
        spatial movement give the same trajectory; timing is kept as a separate, secondary feature "timing" =
        elapsed time / total time at each point, which encodes the relative stroke timing (e.g. 0.30 / 0.40 / 0.30 of
        the time per stroke) independent of the absolute duration. Pauses become a jump in "timing", not extra points.

Feature groups (columns): trajectory (2, centroid-centred / RMS-radius), direction (2), speed (1, / the sample's own
mean speed), acceleration (1, change of that relative speed, scaled / clipped as in production), orientation (2),
fingers (5), timing (1, arc variants only; constant-slope in time variants so it never contributes there).

Variants (predefined - never tuned continuously on anyone's data):
  A_current            time  trajectory 1, direction .6, orientation .3, speed .3, acceleration .15, fingers .1
  A_no_fingers         time  A without fingertip extension
  A_fingers_0.02       time  A with fingertip extension weight .02
  B_no_speed           time  A without speed
  C_traj_direction     time  trajectory 1, direction .6
  D_traj_only          time  trajectory 1
  E1_tempo_robust      time  1 / .6 / orientation .2 / speed .10 / acceleration .05 / fingers .05
  E2_shape_dominant    time  1 / .5 / orientation .1 / speed .05 / acceleration .02 / fingers .02
  F_arc_current        arc   A's weights + timing .3
  G_arc_shape_timing   arc   trajectory 1, direction .6, orientation .2, timing .3, fingers .05
  H_arc_traj_direction arc   trajectory 1, direction .6
  I_arc_tempo_robust   arc   E1's weights + timing .15

Protocol: participants are split into DEVELOPMENT (first half, sorted IDs) and HELD-OUT TEST (second half). Per variant
the threshold is the EER threshold on development genuine / impostor distances; FAR / FRR are reported once on the
held-out test participants at that threshold, together with test EER and ROC-AUC, genuine / impostor distributions,
and breakdowns by gesture duration, normalized duration (probe / enrolled median), participant and session. The
production threshold 0.216 is reported as a reference only (it belongs to A_current's distance scale).

Data:
  --root DATA --consent-confirmed    REAL recordings in the evaluation/hand_gesture_evaluation.py layout (enroll/,
                                     session_N/ genuine probes, impostor_N/ probes compared only with others)
  --synthetic N                      SYNTHETIC tempo-stress users (tests/hand_signals.py with non-uniform tempo
                                     variation: profile_jitter) - software evidence only, never real-world FAR/FRR

Outputs (evidence label REAL DATA / SYNTHETIC CALIBRATION): evaluation/results/hand_representation[_synthetic]_
{summary,by_duration,by_participant,by_session}.csv
"""

from __future__ import annotations

import argparse
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation.ieee.common import REAL, RESULTS, SYNTHETIC, describe, full_report, rates, write_csv

GROUPS = {
    "trajectory": (0, 1), "direction": (2, 3), "speed": (4,), "acceleration": (5,), "orientation": (6, 7),
    "fingers": (8, 9, 10, 11, 12), "timing": (13,),
}
DIM = 14
REFERENCE_THRESHOLD = 0.216
BAND = 0.2
ANGLES = (0.0, 6.0, -6.0, 12.0, -12.0)
LENGTH = 64


@dataclass(frozen=True)
class Variant:
    name: str
    resampling: str          # "time" | "arc"
    weights: dict            # group -> weight (missing = 0)

    def vector(self) -> np.ndarray:
        w = np.zeros(DIM)
        for group, weight in self.weights.items():
            w[list(GROUPS[group])] = weight
        return w


_CURRENT = {"trajectory": 1.0, "direction": 0.6, "orientation": 0.3, "speed": 0.3, "acceleration": 0.15, "fingers": 0.1}
_TEMPO_ROBUST = {"trajectory": 1.0, "direction": 0.6, "orientation": 0.2, "speed": 0.10, "acceleration": 0.05, "fingers": 0.05}
_SHAPE_DOMINANT = {"trajectory": 1.0, "direction": 0.5, "orientation": 0.1, "speed": 0.05, "acceleration": 0.02, "fingers": 0.02}
VARIANTS = [
    Variant("A_current", "time", _CURRENT),
    Variant("A_no_fingers", "time", {k: v for k, v in _CURRENT.items() if k != "fingers"}),
    Variant("A_fingers_0.02", "time", {**_CURRENT, "fingers": 0.02}),
    Variant("B_no_speed", "time", {k: v for k, v in _CURRENT.items() if k != "speed"}),
    Variant("C_traj_direction", "time", {"trajectory": 1.0, "direction": 0.6}),
    Variant("D_traj_only", "time", {"trajectory": 1.0}),
    Variant("E1_tempo_robust", "time", _TEMPO_ROBUST),
    Variant("E2_shape_dominant", "time", _SHAPE_DOMINANT),
    Variant("F_arc_current", "arc", {**_CURRENT, "timing": 0.3}),
    Variant("G_arc_shape_timing", "arc", {"trajectory": 1.0, "direction": 0.6, "orientation": 0.2, "timing": 0.3, "fingers": 0.05}),
    Variant("H_arc_traj_direction", "arc", {"trajectory": 1.0, "direction": 0.6}),
    Variant("I_arc_tempo_robust", "arc", {**_TEMPO_ROBUST, "timing": 0.15}),
]


# ----------------------------------------------------------------------------- representations


def _features(frames: np.ndarray, times_ms: np.ndarray, resampling: str) -> np.ndarray:
    """(LENGTH, DIM) unweighted features of one gated motion sample."""
    from preprocessing import hand_gesture as hg

    t = (times_ms - times_ms[0]) / max(float(times_ms[-1] - times_ms[0]), 1e-9)       # normalized elapsed time 0..1
    palm_centre = frames[:, hg.PALM_POINTS, :2].mean(axis=1)
    raw_speed = np.linalg.norm(np.gradient(palm_centre, t, axis=0), axis=1)              # per normalized time
    if resampling == "time":
        axis = t
    elif resampling == "arc":
        steps = np.linalg.norm(np.diff(palm_centre, axis=0), axis=1)
        axis = np.concatenate([[0.0], np.cumsum(steps)]) / max(float(steps.sum()), 1e-12)   # path progress 0..1
    else:
        raise ValueError(resampling)
    axis = np.maximum.accumulate(axis)
    target = np.linspace(0.0, 1.0, LENGTH)

    def along(values: np.ndarray) -> np.ndarray:
        flat = values.reshape(len(values), -1)
        out = np.stack([np.interp(target, axis, flat[:, c]) for c in range(flat.shape[1])], axis=1)
        return out.reshape((LENGTH,) + values.shape[1:])

    frames_r = along(frames)
    centre = frames_r[:, hg.PALM_POINTS, :2].mean(axis=1)
    centred = centre - centre.mean(axis=0)
    trajectory = centred / max(float(np.sqrt(np.mean(np.sum(centred ** 2, axis=1)))), 1e-12)
    velocity = np.gradient(trajectory, axis=0)
    step = np.linalg.norm(velocity, axis=1)
    direction = velocity / np.maximum(step, hg.DIRECTION_SPEED_FLOOR * max(float(step.mean()), 1e-9))[:, None]
    # Relative speed, normalized per sequence (/ its own mean): in time resampling the step length of the resampled
    # trajectory (exactly production); in arc resampling steps are equal by construction, so the real speed over time
    # is carried along the path instead.
    speed = step if resampling == "time" else along(raw_speed)
    relative_speed = speed / max(float(speed.mean()), 1e-9)
    acceleration = np.clip(np.gradient(relative_speed) * hg.ACCEL_SCALE, -hg.ACCEL_CLIP, hg.ACCEL_CLIP)
    shapes = np.stack([hg.normalize_shape(f) for f in frames_r])
    tips = np.linalg.norm(shapes[:, hg.FINGERTIPS, :], axis=2)
    orientation = shapes[:, hg.MIDDLE_MCP, :2]
    orientation = orientation / np.maximum(np.linalg.norm(orientation, axis=1), 1e-9)[:, None]
    timing = along(t)
    return np.column_stack([trajectory, direction, relative_speed, acceleration, orientation, tips, timing])


@dataclass
class Sample:
    participant: str
    folder: str                 # enroll | session_N | impostor_N
    name: str
    duration_s: float
    features: dict              # resampling -> (LENGTH, DIM)


def load_samples(root: Path) -> tuple[list[Sample], list[dict]]:
    """Gate every file with the production pipeline (capture checks + Z check); build both resamplings."""
    from evaluation.hand_gesture_evaluation import _probe_folders
    from evaluation.real_user_evaluation import discover
    from preprocessing.hand_gesture import HandCaptureRejected, motion_sample, parse_capture, process_capture

    samples, failures = [], []
    for pid in discover(root):
        folders = ([root / pid / "enroll"] if (root / pid / "enroll").is_dir() else []) + _probe_folders(root / pid)
        for folder in folders:
            for f in sorted(folder.glob("attempt_*.json"), key=lambda p: int(p.stem.split("_")[1])):
                try:
                    capture = parse_capture(f.read_bytes())
                    process_capture(capture)                      # production gate incl. Z check
                    motion = motion_sample(capture)
                except HandCaptureRejected as rejected:
                    failures.append({"participant_id": pid, "folder": folder.name, "probe": f.stem, "failure": rejected.code})
                    continue
                samples.append(Sample(pid, folder.name, f.stem, motion.metadata["motion_duration_s"],
                                      {r: _features(motion.frames, motion.times_ms, r) for r in ("time", "arc")}))
    return samples, failures


# ----------------------------------------------------------------------------- matching (as production)


def _distance(probe: np.ndarray, enrolled: list[np.ndarray]) -> tuple[float, list[float]]:
    from preprocessing.hand_gesture import rotate_features
    from template_protection.dtw import dtw_distance

    rotated = [rotate_features(probe, a) for a in ANGLES]
    per = [min(dtw_distance(r, e, BAND) for r in rotated) for e in enrolled]
    return float(np.median(per)), per


def _score_variant(args) -> list[dict]:
    variant, enrollments, probes = args
    w = variant.vector()
    enrolled_w = {pid: [s.features[variant.resampling] * w for s in samples] for pid, samples in enrollments.items()}
    rows = []
    for probe in probes:
        feats = probe.features[variant.resampling] * w
        impostor_folder = probe.folder.startswith("impostor_")
        median_enrolled = {pid: float(np.median([s.duration_s for s in samples])) for pid, samples in enrollments.items()}
        for target, enrolled in enrolled_w.items():
            if impostor_folder and target == probe.participant:
                continue
            d, per = _distance(feats, enrolled)
            rows.append({"variant": variant.name, "claimant_id": probe.participant, "participant_id": target,
                         "session": probe.folder, "probe": probe.name, "genuine": target == probe.participant,
                         "duration_s": probe.duration_s, "normalized_duration": probe.duration_s / median_enrolled[target],
                         "distance": d, "per_sample": per})
    return rows


# ----------------------------------------------------------------------------- reporting


def _auc(g: np.ndarray, i: np.ndarray) -> float:
    """P(genuine distance < impostor distance) + 0.5 P(tie) (Mann-Whitney)."""
    if not len(g) or not len(i):
        return float("nan")
    order = np.argsort(np.concatenate([g, i]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    r_imp = ranks[len(g):].sum()
    return float((r_imp - len(i) * (len(i) + 1) / 2) / (len(g) * len(i)))


def summarize(rows: list[dict], participants: list[str]) -> tuple[dict, list[dict], list[dict], list[dict]]:
    dev = set(participants[: len(participants) // 2])
    test = set(participants) - dev

    def pick(people, genuine):
        return np.array([r["distance"] for r in rows if r["genuine"] == genuine and r["claimant_id"] in people
                         and r["participant_id"] in people], float)

    gd, idv, gt, it = pick(dev, True), pick(dev, False), pick(test, True), pick(test, False)
    variant = rows[0]["variant"]
    dev_report = full_report(gd, idv, higher_is_better=False, operating_threshold=None)
    thr = float(dev_report["EER_threshold"])
    at = rates(gt, it, thr, higher_is_better=False)
    ref = rates(gt, it, REFERENCE_THRESHOLD, higher_is_better=False)
    g, i = describe(gt), describe(it)
    summary = {
        "variant": variant, "dev_participants": len(dev), "test_participants": len(test),
        "dev_EER": dev_report["EER"], "dev_threshold": thr,
        "test_genuine_n": len(gt), "test_impostor_n": len(it),
        "genuine_mean": g["mean"], "genuine_median": g["median"], "genuine_p95": float(np.percentile(gt, 95)),
        "genuine_max": g["max"], "impostor_mean": i["mean"], "impostor_median": i["median"],
        "impostor_p5": float(np.percentile(it, 5)), "impostor_min": i["min"],
        "separation_d_prime": float((i["mean"] - g["mean"]) / np.sqrt(0.5 * (i["sd"] ** 2 + g["sd"] ** 2))),
        "test_FAR": at["FAR"], "test_FRR": at["FRR"],
        "test_EER": full_report(gt, it, higher_is_better=False, operating_threshold=None)["EER"],
        "test_ROC_AUC": _auc(gt, it),
        "reference_0.216_FAR": ref["FAR"], "reference_0.216_FRR": ref["FRR"],
    }
    test_genuine = [r for r in rows if r["genuine"] and r["claimant_id"] in test]
    by_duration = []
    for label, key, edges in (("duration_s", "duration_s", (0, 2.0, 3.0, 99)),
                              ("normalized_duration", "normalized_duration", (0, 0.8, 1.25, 99))):
        for lo, hi in zip(edges[:-1], edges[1:]):
            d = np.array([r["distance"] for r in test_genuine if lo <= r[key] < hi], float)
            if len(d):
                by_duration.append({"variant": variant, "by": label, "bin": f"[{lo}, {hi})", "genuine_n": len(d),
                                    "genuine_mean": float(d.mean()), "FRR": float(np.mean(d > thr))})
    by_participant = []
    for pid in sorted(test):
        d = np.array([r["distance"] for r in test_genuine if r["claimant_id"] == pid], float)
        imp = np.array([r["distance"] for r in rows if not r["genuine"] and r["participant_id"] == pid
                        and r["claimant_id"] in test], float)
        by_participant.append({"variant": variant, "participant_id": pid, "genuine_n": len(d),
                               "genuine_mean": float(d.mean()) if len(d) else None,
                               "FRR": float(np.mean(d > thr)) if len(d) else None,
                               "FAR_as_target": float(np.mean(imp <= thr)) if len(imp) else None})
    by_session = []
    for session in sorted({r["session"] for r in test_genuine}):
        d = np.array([r["distance"] for r in test_genuine if r["session"] == session], float)
        by_session.append({"variant": variant, "session": session, "genuine_n": len(d),
                           "genuine_mean": float(d.mean()), "FRR": float(np.mean(d > thr))})
    return summary, by_duration, by_participant, by_session


def run(root: Path, evidence_label: str, prefix: str, variants=VARIANTS, workers: int | None = None) -> dict:
    samples, failures = load_samples(root)
    enrollments: dict[str, list[Sample]] = {}
    for s in samples:
        if s.folder == "enroll":
            enrollments.setdefault(s.participant, []).append(s)
    enrollments = {pid: v for pid, v in enrollments.items() if len(v) == 3}
    probes = [s for s in samples if s.folder != "enroll"]
    participants = sorted(enrollments)
    if len(participants) < 4:
        raise ValueError(f"need at least 4 enrolled participants for a development / held-out split, got {len(participants)}")
    jobs = [(v, enrollments, probes) for v in variants]
    if workers == 1:
        results = [_score_variant(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_score_variant, jobs))
    summaries, durations, per_participant, sessions = [], [], [], []
    for rows in results:
        s, d, p, se = summarize(rows, participants)
        summaries.append(s)
        durations += d
        per_participant += p
        sessions += se
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary", summaries), ("by_duration", durations), ("by_participant", per_participant),
                       ("by_session", sessions)):
        write_csv(RESULTS / f"{prefix}_{name}.csv", data, evidence_label)
    return {"summary": summaries, "by_duration": durations, "by_participant": per_participant, "by_session": sessions,
            "failures": failures, "participants": participants, "probes": len(probes)}


# ----------------------------------------------------------------------------- synthetic tempo-stress data


def synthetic_tempo_dataset(root: Path, participants: int, seed: int = 5000, jitter: float = 0.25,
                            pose_variation: bool = False) -> None:
    """Consented synthetic participants with REAL-LIKE tempo variation: every execution (enrollment and probe) drawn
    at an independent pace (1.4-3.8 s) with a non-uniformly changed speed profile, stroke timing and corner pauses
    (`profile_jitter`), tracking at 12-30 fps. session_1 and session_2 (another 'day': more jitter, a few degrees of
    rotation, a different Z size) each hold 5 genuine probes; other participants' probes are the impostors.

    `pose_variation`: every participant has a HABITUAL hand pose somewhere between pointing (index out) and open,
    varied per execution (pose_jitter 0.2); in session 2, 2 of the 5 probes are drawn with the hand held noticeably
    differently (a natural pose change with the same Z movement) - the nuisance seen in a real attempt."""
    from tests.hand_signals import OPEN_HAND, POINTING_HAND, Person, transformed

    def write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    lines = ["participant_id,consent,withdrawn"]
    for n in range(participants):
        pid, person = f"P{n + 1:03d}", Person(seed + n)
        rng = np.random.default_rng(seed * 7 + n)
        lines.append(f"{pid},yes,no")
        # Drawn only with pose variation, so the default dataset is generated exactly as before this option existed.
        mix = float(rng.uniform(0, 1)) if pose_variation else 0.0
        habitual = tuple(mix * np.asarray(OPEN_HAND) + (1 - mix) * np.asarray(POINTING_HAND))
        pose = {"finger_extension": habitual, "pose_jitter": 0.2} if pose_variation else {}
        for k in range(3):
            write(root / pid / "enroll" / f"attempt_{k + 1}.json",
                  person.capture(10 * n + k, duration_s=float(rng.uniform(1.4, 3.8)), fps=float(rng.choice([12, 15, 20, 30])),
                                 profile_jitter=jitter, **pose))
        for session, extra in ((1, 0.0), (2, 0.1)):
            for k in range(5):
                probe_pose = dict(pose)
                if pose_variation and session == 2 and k < 2:
                    other = OPEN_HAND if mix < 0.5 else POINTING_HAND
                    probe_pose["finger_extension"] = tuple(0.5 * np.asarray(habitual) + 0.5 * np.asarray(other))
                capture = person.capture(100_000 + 100 * n + 10 * session + k, duration_s=float(rng.uniform(1.4, 3.8)),
                                         fps=float(rng.choice([12, 15, 20, 30])), profile_jitter=jitter + extra, **probe_pose)
                if session == 2:
                    capture = transformed(capture, angle_deg=float(rng.uniform(-8, 8)), scale=float(rng.uniform(0.8, 1.2)))
                write(root / pid / f"session_{session}" / f"attempt_{k + 1}.json", capture)
    (root / "consent.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print(result: dict) -> None:
    print(f"participants {len(result['participants'])} (dev {result['summary'][0]['dev_participants']}, held-out test "
          f"{result['summary'][0]['test_participants']}), probes {result['probes']}, capture failures {len(result['failures'])}")
    head = (f"{'variant':22s} {'devEER':>6s} {'thr':>6s} | {'gen mean':>8s} {'med':>6s} {'p95':>6s} | {'imp mean':>8s} "
            f"{'p5':>6s} | {'FAR':>6s} {'FRR':>6s} {'EER':>6s} {'AUC':>6s} {'dprime':>6s} | ref0.216 FAR/FRR")
    print(head)
    for s in result["summary"]:
        print(f"{s['variant']:22s} {s['dev_EER']:6.3f} {s['dev_threshold']:6.3f} | {s['genuine_mean']:8.3f} "
              f"{s['genuine_median']:6.3f} {s['genuine_p95']:6.3f} | {s['impostor_mean']:8.3f} {s['impostor_p5']:6.3f} | "
              f"{s['test_FAR']:6.3f} {s['test_FRR']:6.3f} {s['test_EER']:6.3f} {s['test_ROC_AUC']:6.4f} "
              f"{s['separation_d_prime']:6.2f} | {s['reference_0.216_FAR']:.3f}/{s['reference_0.216_FRR']:.3f}")
    print("\ngenuine (held-out) by normalized duration  [<0.8 | 0.8-1.25 | >=1.25]  mean distance / FRR")
    for v in dict.fromkeys(r["variant"] for r in result["by_duration"]):
        cells = [f"{r['genuine_mean']:.3f}/{r['FRR']:.2f} (n={r['genuine_n']})" for r in result["by_duration"]
                 if r["variant"] == v and r["by"] == "normalized_duration"]
        print(f"  {v:22s} " + "  ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--consent-confirmed", action="store_true")
    ap.add_argument("--synthetic", type=int, default=None, metavar="N")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--variants", default=None, help="comma-separated variant names (default: all)")
    ap.add_argument("--pose-variation", action="store_true", help="synthetic: habitual hand poses + natural pose changes")
    a = ap.parse_args()
    variants = [v for v in VARIANTS if a.variants is None or v.name in a.variants.split(",")]
    if a.synthetic:
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_tempo_dataset(Path(tmp), a.synthetic, pose_variation=a.pose_variation)
            prefix = "hand_representation_synthetic" + ("_pose" if a.pose_variation else "")
            if a.variants:
                prefix += "_subset"   # never overwrite the full comparison with a partial run
            result = run(Path(tmp), SYNTHETIC, prefix, variants=variants, workers=a.workers)
    else:
        if a.root is None:
            raise SystemExit("--root DATA --consent-confirmed (real data) or --synthetic N")
        if not a.consent_confirmed:
            raise SystemExit("Refusing to run without --consent-confirmed (informed consent for every participant).")
        result = run(a.root, REAL, "hand_representation", variants=variants, workers=a.workers)
    _print(result)


if __name__ == "__main__":
    main()
