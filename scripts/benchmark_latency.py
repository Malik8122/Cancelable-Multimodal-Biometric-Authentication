"""Latency benchmark of every stage of the shipped pipeline on real samples (Part 10 of the IEEE evaluation).

Warm: `--warm` repetitions per stage in one process after a warm-up call.
Cold: `--cold` fresh subprocesses per modality, each timing import + model load + first embedding.
API: FastAPI TestClient against a temporary SQLite database with real models (DEBUG_SCORES off, as in production).
Samples: one real LFW image and one VoxCeleb-subset test utterance (paths from data/eval_cache).

Run: python -m scripts.benchmark_latency [--warm 100] [--cold 10] [--api-fusion-runs 30]
Outputs: evaluation/results/latency_benchmark.csv, evaluation/reports/latency_report.md, figure fig17.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.ieee.common import COLORS, DBL_W, EVAL_SECRET, REAL, REPORTS, RESULTS, hardware, ieee_style, save_fig, save_json, write_csv  # noqa: E402


def _samples():
    cache = REPO / "data" / "eval_cache"
    face = str(np.load(cache / "face_embeddings.npz", allow_pickle=True)["paths"][0])
    voice = str(np.load(cache / "voice_embeddings.npz", allow_pickle=True)["paths"][0])
    return face, voice


def _rgb(path):
    import cv2

    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def _time(fn, n):
    fn()  # warm-up
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return out


def stats(ms):
    a = np.asarray(ms)
    return {"runs": len(a), "mean_ms": a.mean(), "median_ms": np.median(a), "sd_ms": a.std(ddof=1) if len(a) > 1 else 0.0,
            "p95_ms": np.percentile(a, 95), "p99_ms": np.percentile(a, 99), "min_ms": a.min(), "max_ms": a.max()}


def warm(n: int, api_fusion_runs: int) -> list[dict]:
    import torch

    from backend.config import Settings
    from backend.services.modality_metrics import decide
    from embeddings.pipelines import FacePipeline, VoicePipeline
    from fusion.config import FusionPolicy
    from fusion.policy import evaluate_fusion_policy
    from preprocessing.voice import load_wav_file
    from template_protection.biohash import generate_template
    from template_protection.hkdf_keys import derive_key
    from template_protection.matcher import compare

    face_path, voice_path = _samples()
    img = _rgb(face_path)
    wav, sr = load_wav_file(voice_path)
    fpipe, vpipe = FacePipeline(), VoicePipeline()
    rows = []

    def add(stage, ms, **extra):
        rows.append({"stage": stage, "mode": "warm", **stats(ms), **extra})
        print(stage, round(float(np.median(ms)), 2), "ms", flush=True)

    face_pre = fpipe._preprocessor.preprocess(img)
    add("face_preprocessing_mtcnn", _time(lambda: fpipe._preprocessor.preprocess(img), n))
    add("face_embedding_inceptionresnetv1", _time(lambda: fpipe._embedder.extract_embedding(face_pre), n))
    add("face_pipeline_total", _time(lambda: fpipe.embed(img), n))

    def voice_pre():
        m = vpipe._preprocessor.preprocess(wav, sample_rate=sr, training=False)
        return np.stack([m] * 3, axis=-1) if m.ndim == 2 else m

    vp = voice_pre()
    add("voice_preprocessing_logmel", _time(voice_pre, n))
    add("voice_embedding_ecapa", _time(lambda: vpipe._embedder.extract_embedding(vp), n))
    add("voice_pipeline_total", _time(lambda: vpipe.embed(wav, sample_rate=sr), n))

    e_face, e_voice = fpipe.embed(img), vpipe.embed(wav, sample_rate=sr)
    key = derive_key(EVAL_SECRET, application_id="bench", user_id="u", modality="face", key_version=1)
    add("hkdf_key_derivation", _time(lambda: derive_key(EVAL_SECRET, application_id="bench", user_id="u", modality="face", key_version=1), n))
    add("biohash_face_512d_256bit", _time(lambda: generate_template(e_face, key, output_bits=256), n))
    add("biohash_voice_192d_256bit", _time(lambda: generate_template(e_voice, key, output_bits=256), n))
    ta, tb = generate_template(e_face, key, output_bits=256), generate_template(e_face * 0.9 + 0.1, key, output_bits=256)
    add("hamming_comparison", _time(lambda: compare(ta, tb, metric="hamming"), n))
    s = Settings(master_secret="x")
    add("decision_estimate", _time(lambda: decide("face", 0.85, 256, s), n))
    add("fusion_policy", _time(lambda: evaluate_fusion_policy({"face": 0.9, "voice": 0.8}, {"face": True, "voice": True},
                                                              FusionPolicy.ALL_REQUIRED, 0.76), n))
    rows += _db_and_api(n, api_fusion_runs, img, wav, sr, e_face, e_voice)
    rows.append({"stage": "torch_threads", "mode": "info", "runs": torch.get_num_threads()})
    return rows


def _db_and_api(n, api_fusion_runs, img, wav, sr, e_face, e_voice):
    import io

    import cv2
    from scipy.io import wavfile

    tmp = Path(tempfile.mkdtemp(prefix="bench_"))
    os.environ.update({"DATABASE_URL": f"sqlite:///{tmp / 'bench.db'}", "MASTER_SECRET": EVAL_SECRET, "DEBUG_SCORES": "false"})
    from backend.config import get_settings
    from backend.database.session import get_engine, get_session_factory

    for fn in (get_settings, get_engine, get_session_factory):
        fn.cache_clear()
    from fastapi.testclient import TestClient

    from backend.database import crud
    from backend.main import app
    from backend.services.face_service import get_face_service
    from backend.services.voice_service import get_voice_service

    rows = []
    with TestClient(app) as client:
        db = get_session_factory()()
        settings = get_settings()
        face_svc, voice_svc = get_face_service(), get_voice_service()
        rng = np.random.default_rng(0)
        for u in range(1000):  # a realistic table: 1000 users x 2 modalities x 4 template sets
            for svc, dim in ((face_svc, 512), (voice_svc, 192)):
                v = rng.standard_normal(dim)
                svc._store(db, v / np.linalg.norm(v), f"u{u}", settings.application_id)
        face_svc._store(db, e_face, "target", settings.application_id)
        voice_svc._store(db, e_voice, "target", settings.application_id)
        rows.append({"stage": "db_lookup_active_template_2000_users_x4_sets", "mode": "warm",
                     **stats(_time(lambda: crud.get_active_template(db, "u500", "face", settings.application_id), n))})
        png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()
        buf = io.BytesIO()
        wavfile.write(buf, sr, (np.asarray(wav) * 32767).astype(np.int16))
        wav_bytes = buf.getvalue()

        def call(files):
            r = client.post("/authenticate/fusion", data={"user_id": "target", "application_id": settings.application_id}, files=files)
            assert r.status_code == 200, r.text

        for name, files, runs in (
            ("api_authenticate_face", lambda: {"face_image": ("f.png", png, "image/png")}, n),
            ("api_authenticate_voice", lambda: {"voice_audio": ("v.wav", wav_bytes, "audio/wav")}, n),
            ("api_authenticate_face_plus_voice", lambda: {"face_image": ("f.png", png, "image/png"), "voice_audio": ("v.wav", wav_bytes, "audio/wav")}, api_fusion_runs),
        ):
            rows.append({"stage": name, "mode": "warm", **stats(_time(lambda f=files: call(f()), runs)),
                         "note": "multi-modality requests release model caches after use (512 MB host design) - models reload per request" if "plus" in name else ""})
            print(name, round(rows[-1]["median_ms"], 1), "ms", flush=True)
    return rows


def cold_worker(modality: str) -> None:
    t0 = time.perf_counter()
    from embeddings.pipelines import FacePipeline, VoicePipeline

    face_path, voice_path = _samples()
    t_import = time.perf_counter()
    if modality == "face":
        p = FacePipeline()
        t_load = time.perf_counter()
        p.embed(_rgb(face_path))
    else:
        from preprocessing.voice import load_wav_file

        p = VoicePipeline()
        t_load = time.perf_counter()
        w, sr = load_wav_file(voice_path)
        p.embed(w, sample_rate=sr)
    t_end = time.perf_counter()
    print(json.dumps({"import_ms": (t_import - t0) * 1000, "model_load_ms": (t_load - t_import) * 1000,
                      "first_inference_ms": (t_end - t_load) * 1000, "total_ms": (t_end - t0) * 1000}))


def cold(runs: int) -> list[dict]:
    rows = []
    for modality in ("face", "voice"):
        res = []
        for _ in range(runs):
            out = subprocess.run([sys.executable, "-m", "scripts.benchmark_latency", "--cold-worker", modality],
                                 cwd=REPO, capture_output=True, text=True, timeout=600)
            res.append(json.loads(out.stdout.strip().splitlines()[-1]))
        for part in ("import_ms", "model_load_ms", "first_inference_ms", "total_ms"):
            rows.append({"stage": f"cold_{modality}_{part[:-3]}", "mode": "cold", **stats([r[part] for r in res])})
        print("cold", modality, round(float(np.median([r["total_ms"] for r in res]))), "ms", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=100)
    ap.add_argument("--cold", type=int, default=10)
    ap.add_argument("--api-fusion-runs", type=int, default=30)
    ap.add_argument("--cold-worker")
    a = ap.parse_args()
    if a.cold_worker:
        cold_worker(a.cold_worker)
        return
    hw = hardware()
    rows = warm(a.warm, a.api_fusion_runs) + cold(a.cold)
    write_csv(RESULTS / "latency_benchmark.csv", rows, REAL)
    save_json(REPORTS / "hardware.json", hw)
    lines = ["# Latency benchmark", "", f"Evidence label: {REAL} (measured on this machine; real samples).", "",
             "Hardware: " + ", ".join(f"{k}={v}" for k, v in hw.items()), "",
             "| stage | mode | runs | mean ms | median ms | SD | p95 | p99 | min | max |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["mode"] == "info":
            continue
        lines.append(f"| {r['stage']} | {r['mode']} | {r['runs']} | " + " | ".join(f"{r[k]:.2f}" for k in
                     ("mean_ms", "median_ms", "sd_ms", "p95_ms", "p99_ms", "min_ms", "max_ms")) + " |")
    (REPORTS / "latency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt = ieee_style()
    wr = [r for r in rows if r["mode"] == "warm"]
    fig, ax = plt.subplots(figsize=(DBL_W * 0.6, 2.8))
    y = np.arange(len(wr))
    ax.barh(y, [r["median_ms"] for r in wr], xerr=[[r["median_ms"] - r["min_ms"] for r in wr], [r["p95_ms"] - r["median_ms"] for r in wr]],
            color=COLORS["face"], height=0.6, error_kw={"lw": 0.6, "capsize": 1.5})
    ax.set_yticks(y, [r["stage"] for r in wr], fontsize=5.5)
    ax.invert_yaxis()
    ax.set(xscale="log", xlabel="latency (ms, log scale): median, whisker min..p95", title="Warm per-stage latency (CPU)")
    ax.grid(True, axis="x", which="both")
    save_fig(fig, "fig17_latency_distribution", REAL, "Warm latency per pipeline stage on the evaluation machine (median; whiskers min to p95).")


if __name__ == "__main__":
    main()
