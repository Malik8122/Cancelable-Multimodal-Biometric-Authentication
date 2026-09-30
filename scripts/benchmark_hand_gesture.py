"""Backend latency of the hand gesture modality (landmarks -> decision), on this machine.

Measures, over `--runs` repetitions: capture parsing + validation, feature extraction (normalization, resampling,
features), one DTW comparison, and a full one-gesture authentication (feature extraction + keyed transform + rotation search x 3
enrolled samples DTW comparisons + aggregation, against a real template set in an in-memory database).

The INPUT is synthetic landmark sequences (tests/hand_signals.py) - timing depends on the sequence length and the
64-frame resampling, not on who performed the gesture, so synthetic input is adequate for latency (never for
accuracy). MediaPipe Hands runs in the browser and is NOT measured here: the frontend reports its per-frame
inference time with every capture (`client_metrics.mediapipe_ms_mean`, audited in `modality_diagnostics`).

Run: python -m scripts.benchmark_hand_gesture [--runs 50] [--out evaluation/results/hand_gesture_latency.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.hand_gesture import parse_capture, process_capture, weighted  # noqa: E402
from template_protection.dtw import dtw_distance  # noqa: E402
from tests.hand_signals import Person  # noqa: E402


def _stats(name: str, values: list[float]) -> dict:
    v = np.asarray(values)
    return {"stage": name, "runs": len(v), "mean_ms": round(float(v.mean()), 3), "median_ms": round(float(np.median(v)), 3),
            "p95_ms": round(float(np.percentile(v, 95)), 3), "max_ms": round(float(v.max()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from backend.config import Settings
    from backend.database.models import Base
    from backend.services.hand_service import HandGestureService

    person = Person(1)
    payloads = [json.dumps(person.capture(100 + k)) for k in range(a.runs)]
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    service = HandGestureService(Settings(master_secret="benchmark-secret"))
    service.enroll_attempts(db, [parse_capture(json.dumps(person.capture(k))) for k in range(3)], "bench", "bench-app")

    parse_ms, feature_ms, dtw_ms, auth_ms = [], [], [], []
    reference = weighted(process_capture(parse_capture(payloads[0])).features)
    for payload in payloads:
        t0 = time.perf_counter()
        capture = parse_capture(payload)
        t1 = time.perf_counter()
        features = weighted(process_capture(capture).features)
        t2 = time.perf_counter()
        dtw_distance(features, reference)
        t3 = time.perf_counter()
        service.authenticate(db, capture, "bench", "bench-app")
        t4 = time.perf_counter()
        parse_ms.append((t1 - t0) * 1000)
        feature_ms.append((t2 - t1) * 1000)
        dtw_ms.append((t3 - t2) * 1000)
        auth_ms.append((t4 - t3) * 1000)
    rows = [_stats("parse_and_validate_json", parse_ms), _stats("feature_extraction", feature_ms),
            _stats("dtw_single_comparison", dtw_ms), _stats("authentication_total_3_samples", auth_ms)]
    frames = [len(json.loads(p)["frames"]) for p in payloads]
    print(f"input: synthetic landmark sequences, {min(frames)}-{max(frames)} frames, resampled to 64")
    for r in rows:
        print(f"{r['stage']:<38} mean {r['mean_ms']:8.2f} ms   median {r['median_ms']:8.2f}   p95 {r['p95_ms']:8.2f}")
    if a.out:
        from evaluation.ieee.common import SOFTWARE, write_csv

        write_csv(a.out, [dict(r, note="backend only; synthetic landmark input; MediaPipe runs in the browser") for r in rows],
                  SOFTWARE)
        print("written:", a.out)


if __name__ == "__main__":
    main()
