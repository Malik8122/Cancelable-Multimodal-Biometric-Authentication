"""Z hand gesture evaluation harness: genuine / impostor DTW distances -> FAR, FRR, EER, ROC-AUC, threshold calibration.

It runs the SHIPPED code (backend/services/hand_service.py: capture gate + Z check, features, keyed template sets,
rotation search, DTW, aggregation) and writes only scores and non-biometric metadata - never landmarks.

Folder layout (same consent rules as evaluation/real_user_evaluation.py; anonymous IDs like P001 only):

    <root>/consent.csv                      participant_id,consent,withdrawn
    <root>/P001/enroll/attempt_1.json ... attempt_3.json          the three enrollment Z samples
    <root>/P001/session_1/attempt_1.json ...                       genuine authentication probes (any number)
    <root>/P001/session_2/...
    <root>/P001/impostor_1/attempt_1.json ...                      impostor probes: P001's own natural Z, compared
                                                                   ONLY with the other participants' enrollments
    <root>/P009/impostor_1/...                                     an impostor-only participant (no enroll folder)

Each JSON file is one gesture execution in the wire format the frontend sends (preprocessing/hand_gesture.py
::parse_capture); the Testing page's "Hand gesture data collection" saves files in exactly this format, plus a
`collection` block (participant, sample type, tracking rate, duration, pipeline version) that the harness reports and
checks against the running pipeline (HAND_TEMPLATE_FORMAT) - samples recorded for another version are flagged.

Protocol: every probe of every participant is compared with EVERY participant's enrollment under the TARGET's key
(genuine when claimant == target, impostor otherwise) - the pairing the live system performs, i.e. zero-effort
impostors who draw the same Z. Probes that fail the capture gate (including the Z-structure check, e.g. a different
movement) are counted as failures to acquire by code, never as matches or mismatches.

Calibration: distances are aggregated over the enrolled samples with each candidate rule (min / median / mean); the
threshold is the EER threshold on a DEV half of the participants and FAR / FRR are then reported on the held-out TEST
half at that threshold (with fewer than 4 participants, only the pooled figures are reported). The operating threshold
of the config is reported alongside. Nothing is tuned on a single user or a single successful attempt.

Files saved by the Testing page are named `P001__enroll__attempt_1.json`; file them into the layout above with
      python -m evaluation.hand_gesture_evaluation --root <folder> --import-downloads <downloads folder>

Run (real data):  python -m evaluation.hand_gesture_evaluation --root <folder> --consent-confirmed
      -> evidence label REAL DATA, evaluation/results/hand_gesture_*.csv
Run (synthetic):  python -m evaluation.hand_gesture_evaluation --synthetic 30
      -> evidence label SYNTHETIC CALIBRATION, evaluation/results/hand_gesture_synthetic_*.csv. Synthetic users
         (tests/hand_signals.py) only show the software separates the modelled variation; they are NOT evidence of
         real-world FAR / FRR.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from evaluation.ieee.common import EVAL_SECRET, REAL, RESULTS, SYNTHETIC, describe, full_report, rates, write_csv
from evaluation.real_user_evaluation import discover

APP = "hand-gesture-eval"
OUTPUT_NAMES = ("scores", "enrollment", "metrics", "threshold_sweep", "calibration")
OUTPUT_FILES = tuple(f"hand_gesture_{n}.csv" for n in OUTPUT_NAMES)
SWEEP = np.round(np.arange(0.02, 1.2001, 0.02), 3)
AGGREGATIONS = {"min": np.min, "median": np.median, "mean": np.mean}
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def _captures(folder: Path):
    from preprocessing.hand_gesture import parse_capture

    files = sorted(folder.glob("attempt_*.json"), key=lambda p: int(p.stem.split("_")[1])) if folder.is_dir() else []
    return [(f.stem, parse_capture(f.read_bytes())) for f in files]


def _collection(folder: Path, name: str) -> dict:
    """The Testing page's `collection` metadata of one file (empty for files without it)."""
    try:
        meta = json.loads((folder / f"{name}.json").read_text(encoding="utf-8")).get("collection")
    except (OSError, ValueError, AttributeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _probe_folders(participant_folder: Path) -> list[Path]:
    """session_N (genuine probes) and impostor_N (impostor-only probes) folders, in order."""
    folders = [p for p in participant_folder.iterdir() if p.is_dir() and (p.name.startswith("session_") or p.name.startswith("impostor_"))]
    return sorted(folders, key=lambda p: (p.name.split("_")[0], int(p.name.split("_")[1])))


def import_downloads(downloads: Path, root: Path) -> list[Path]:
    """Move `<participant>__<folder>__attempt_<n>.json` files (Testing page) into `<root>/<participant>/<folder>/`."""
    import re
    import shutil

    pattern = re.compile(r"^(P\d{3,})__(enroll|session_\d+|impostor_\d+)__(attempt_\d+)\.json$")
    moved = []
    for f in sorted(downloads.glob("*.json")):
        match = pattern.match(f.name)
        if match:
            target = root / match.group(1) / match.group(2) / f"{match.group(3)}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), target)
            moved.append(target)
    return moved


def _distribution(scope: str, values: np.ndarray) -> dict:
    row = {"scope": scope, **describe(values)}
    if len(values):
        row.update({f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES})
    return row


def calibrate(rows: list[dict], participants: list[str], operating_threshold: float) -> list[dict]:
    """Per aggregation rule: EER threshold on the DEV half of the participants, FAR/FRR at it on the TEST half; plus
    the pooled EER / AUC and the rates at the configured operating threshold."""
    dev = set(participants[: len(participants) // 2])
    test = set(participants) - dev
    out = []
    for name, fn in AGGREGATIONS.items():
        def values(genuine: bool, people: set[str] | None):
            return np.array([fn(r["_per_sample"]) for r in rows if r["genuine"] == genuine
                             and (people is None or (r["claimant_id"] in people and r["participant_id"] in people))], float)

        g, i = values(True, None), values(False, None)
        if not (len(g) and len(i)):
            continue
        pooled = full_report(g, i, higher_is_better=False, operating_threshold=operating_threshold)
        row = {"aggregation": name, "n_genuine": len(g), "n_impostor": len(i), "pooled_EER": pooled["EER"],
               "pooled_ROC_AUC": pooled["ROC_AUC"], "pooled_EER_threshold": pooled["EER_threshold"],
               "operating_threshold": operating_threshold, "operating_FAR": pooled["at_operating_FAR"],
               "operating_FRR": pooled["at_operating_FRR"]}
        gd, idv, gt, it = values(True, dev), values(False, dev), values(True, test), values(False, test)
        if len(participants) >= 4 and len(gd) and len(idv) and len(gt) and len(it):
            dev_report = full_report(gd, idv, higher_is_better=False, operating_threshold=None)
            held_out = rates(gt, it, dev_report["EER_threshold"], higher_is_better=False)
            row.update(dev_participants=len(dev), test_participants=len(test), dev_EER=dev_report["EER"],
                       calibrated_threshold=dev_report["EER_threshold"], test_FAR=held_out["FAR"], test_FRR=held_out["FRR"],
                       test_EER=full_report(gt, it, higher_is_better=False, operating_threshold=None)["EER"])
        out.append(row)
    return out


def run(root: Path, out_dir: Path = RESULTS, evidence_label: str = REAL, threshold: float | None = None,
        prefix: str = "hand_gesture") -> dict:
    """Enroll every consented participant, score every probe against every enrollment, write the CSVs."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from backend.config import Settings
    from backend.database.models import Base
    from backend.services.hand_service import HandGestureService
    from preprocessing.hand_gesture import HAND_TEMPLATE_FORMAT, HandCaptureRejected, weighted

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    settings = Settings(master_secret=EVAL_SECRET, template_pool_size=4)
    service = HandGestureService(settings)
    threshold = settings.hand_gesture_dtw_threshold if threshold is None else threshold
    participants = discover(root)

    enrolled, enrollment_rows = [], []
    for pid, info in participants.items():
        if not info["enroll"].is_dir():
            enrollment_rows.append({"participant_id": pid, "outcome": "IMPOSTOR_ONLY"})
            continue
        try:
            _, summary = service.enroll_attempts(db, [c for _, c in _captures(info["enroll"])], pid, APP)
            enrolled.append(pid)
            enrollment_rows.append({"participant_id": pid, "outcome": "ENROLLED", **{
                k: summary[k] for k in ("gesture_type", "attempts", "intra_distance_median", "intra_distance_max",
                                        "medoid_sample")}})
        except (HandCaptureRejected, ValueError) as error:
            enrollment_rows.append({"participant_id": pid, "outcome": getattr(error, "code", type(error).__name__),
                                    "sample": getattr(error, "attempt", None), "detail": str(error)})

    score_rows = []
    for claimant, info in participants.items():
        for session in _probe_folders(root / claimant):
            impostor_folder = session.name.startswith("impostor_")
            s_id = session.name
            for name, capture in _captures(session):
                meta = _collection(session, name)
                sample = {"sample_type": "impostor" if impostor_folder else "genuine",
                          "pipeline_version": meta.get("pipeline_version", ""),
                          "pipeline_mismatch": bool(meta.get("pipeline_version")) and meta.get("pipeline_version") != HAND_TEMPLATE_FORMAT}
                started = time.perf_counter()
                try:
                    processed = service.process(capture)
                except HandCaptureRejected as rejected:
                    score_rows.append({"claimant_id": claimant, "session": s_id, "probe": name,
                                       "failure_to_acquire": rejected.code, **sample})
                    continue
                features = weighted(processed.features)
                feature_ms = (time.perf_counter() - started) * 1000
                for target in enrolled:
                    if impostor_folder and target == claimant:
                        continue  # an impostor sample is never compared with its own enrollment
                    r = service.authenticate_embedding(db, features, target, APP, capture_metadata=processed.metadata,
                                                       feature_ms=feature_ms)
                    d = r.diagnostics or {}
                    per_sample = d.get("dtw_distances", [])
                    score_rows.append({
                        "participant_id": target, "claimant_id": claimant, "session": s_id, "probe": name, **sample,
                        "genuine": claimant == target, "dtw_distance": r.metric_value, "accepted": r.metric_value <= threshold,
                        **{f"dtw_sample_{k}": v for k, v in enumerate(per_sample, start=1)},
                        "_per_sample": per_sample,
                        "frames_detected": d.get("frames_detected"), "duration_s": d.get("duration_s"),
                        "motion_duration_s": d.get("motion_duration_s"), "tracking_fps": d.get("tracking_fps"),
                        "feature_ms": d.get("timing_ms", {}).get("feature_extraction"),
                        "dtw_ms": d.get("timing_ms", {}).get("dtw_matching"),
                        "best_rotation_deg": d.get("dtw_best_rotation_deg"),
                        "mediapipe_ms_mean": d.get("timing_ms", {}).get("mediapipe_ms_mean"),
                    })
    out_dir.mkdir(parents=True, exist_ok=True)
    public_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in score_rows]
    write_csv(out_dir / f"{prefix}_scores.csv", public_rows, evidence_label)
    write_csv(out_dir / f"{prefix}_enrollment.csv", enrollment_rows, evidence_label)

    valid = [r for r in score_rows if "genuine" in r]
    genuine = np.array([r["dtw_distance"] for r in valid if r["genuine"]], float)
    impostor = np.array([r["dtw_distance"] for r in valid if not r["genuine"]], float)
    fta = sum(1 for r in score_rows if r.get("failure_to_acquire"))
    mismatched = sorted({(r["claimant_id"], r["session"], r["probe"]) for r in score_rows if r.get("pipeline_mismatch")})
    if mismatched:
        print(f"WARNING: {len(mismatched)} probe(s) were recorded for another pipeline version than {HAND_TEMPLATE_FORMAT}")
    probes = len({(r["claimant_id"], r["session"], r["probe"]) for r in score_rows})
    metrics, sweep, calibration = [], [], []
    if len(genuine) and len(impostor):
        metrics.append({"scope": "hand_gesture", "aggregation": settings.hand_gesture_aggregation,
                        "operating_threshold": threshold, "rule": "dtw_distance <=", "failure_to_acquire": fta,
                        "probes": probes,
                        **full_report(genuine, impostor, higher_is_better=False, operating_threshold=threshold)})
        metrics.append(_distribution("genuine_distance", genuine))
        metrics.append(_distribution("impostor_distance", impostor))
        for t in SWEEP:
            r = rates(genuine, impostor, float(t), higher_is_better=False)
            sweep.append({"threshold": float(t), **{k: r[k] for k in ("FAR", "FRR", "TAR", "accuracy")}})
        calibration = calibrate(valid, enrolled, threshold)
    else:
        metrics.append({"scope": "hand_gesture", "note": "not enough genuine AND impostor comparisons to compute metrics",
                        "failure_to_acquire": fta, "probes": probes})
    write_csv(out_dir / f"{prefix}_metrics.csv", metrics, evidence_label)
    write_csv(out_dir / f"{prefix}_threshold_sweep.csv", sweep or [{"note": "no metrics"}], evidence_label)
    write_csv(out_dir / f"{prefix}_calibration.csv", calibration or [{"note": "no metrics"}], evidence_label)
    return {"enrolled": enrolled, "scores": public_rows, "metrics": metrics, "sweep": sweep, "calibration": calibration}


# ----------------------------------------------------------------------------- synthetic dataset


#: Probe conditions of a synthetic participant: natural variation plus the stresses the pipeline must absorb.
SYNTHETIC_PROBES = ("normal", "slow_3.1s", "fast_1.5s", "idle_padding", "tracked_12fps", "moved_scaled_rotated_dropouts")


def synthetic_dataset(root: Path, participants: int, seed: int = 3000) -> None:
    """Write a consented synthetic participant folder (tests/hand_signals.py) in the harness layout: 3 enrollment Z
    samples and one probe per SYNTHETIC_PROBES condition per participant. The default seeds (3000+) are disjoint from
    the synthetic people the representation was designed on (1000+), so the run is an independent confirmation."""
    from tests.hand_signals import Person, transformed, with_idle

    def write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    lines = ["participant_id,consent,withdrawn"]
    for n in range(participants):
        pid, person = f"P{n + 1:03d}", Person(seed + n)
        lines.append(f"{pid},yes,no")
        for k in range(3):
            write(root / pid / "enroll" / f"attempt_{k + 1}.json", person.capture(10 * n + k))
        base = 100_000 + 10 * n
        probes = [
            person.capture(base),
            person.capture(base + 1, duration_s=3.1),
            person.capture(base + 2, duration_s=1.5),
            with_idle(person.capture(base + 3), 1.2, 0.8, seed=n),
            person.capture(base + 4, fps=12),
            transformed(person.capture(base + 5, dropout=0.1), dx=0.05, scale=0.8, angle_deg=10),
        ]
        for k, payload in enumerate(probes, start=1):
            write(root / pid / "session_1" / f"attempt_{k}.json", payload)
    (root / "consent.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--consent-confirmed", action="store_true", help="confirm every participant gave informed consent")
    ap.add_argument("--threshold", type=float, default=None, help="operating threshold to report (default: configured)")
    ap.add_argument("--import-downloads", type=Path, default=None, help="file Testing-page downloads into --root and stop")
    ap.add_argument("--synthetic", type=int, default=None, metavar="N",
                    help="evaluate N synthetic participants instead of --root (SYNTHETIC CALIBRATION label)")
    a = ap.parse_args()
    if a.synthetic:
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_dataset(Path(tmp), a.synthetic)
            result = run(Path(tmp), evidence_label=SYNTHETIC, threshold=a.threshold, prefix="hand_gesture_synthetic")
        _print(result)
        return
    if a.root is None:
        raise SystemExit("--root is required (or --synthetic N)")
    if a.import_downloads:
        moved = import_downloads(a.import_downloads, a.root)
        print(f"{len(moved)} files filed under {a.root}")
        return
    if not a.consent_confirmed:
        raise SystemExit("Refusing to run without --consent-confirmed (informed consent for every participant).")
    _print(run(a.root, threshold=a.threshold))


def _print(result: dict) -> None:
    print(f"{len(result['enrolled'])} enrolled, {len(result['scores'])} comparison rows -> {RESULTS}")
    for m in result["metrics"]:
        if m["scope"] == "hand_gesture" and "EER" in m:
            print(f"pooled ({m['aggregation']}): EER {m['EER']:.4f}  ROC-AUC {m['ROC_AUC']:.4f}  at threshold "
                  f"{m['operating_threshold']}: FAR {m['at_operating_FAR']:.4f} FRR {m['at_operating_FRR']:.4f}  "
                  f"FTA {m['failure_to_acquire']}")
        elif "mean" in m:
            print(f"{m['scope']:18s} n={m['n']:5d} mean {m['mean']:.3f} sd {m['sd']:.3f} median {m['median']:.3f} "
                  f"p5 {m['p5']:.3f} p95 {m['p95']:.3f} min {m['min']:.3f} max {m['max']:.3f}")
    for c in result["calibration"]:
        line = f"{c['aggregation']:6s} pooled EER {c['pooled_EER']:.4f} AUC {c['pooled_ROC_AUC']:.4f}"
        if "calibrated_threshold" in c:
            line += (f" | dev EER {c['dev_EER']:.4f} -> threshold {c['calibrated_threshold']:.3f} | held-out test "
                     f"FAR {c['test_FAR']:.4f} FRR {c['test_FRR']:.4f}")
        print(line)


if __name__ == "__main__":
    main()
