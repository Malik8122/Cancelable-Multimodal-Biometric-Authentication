"""Memory benchmark (Part 11): RSS and peak working set per scenario, each in a FRESH process (3 repeats).

`peak_wset` (Windows) / `ru_maxrss` (POSIX) is the process's true peak; RSS is sampled before and after the
scenario. Template generation additionally reports Python-heap peak via tracemalloc.
Run: python -m scripts.benchmark_memory
Outputs: evaluation/results/memory_benchmark.csv, figure fig28.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SCENARIOS = [
    "baseline_python", "import_backend", "face_model_loaded", "voice_model_loaded",
    "face_embedding", "voice_embedding", "template_generation",
    "api_enroll_voice", "api_authenticate_face", "api_authenticate_face_plus_voice",
]


def _peak_mb():
    import psutil

    mi = psutil.Process().memory_info()
    return (getattr(mi, "peak_wset", None) or mi.rss) / 1e6


def _rss_mb():
    import psutil

    return psutil.Process().memory_info().rss / 1e6


def worker(name: str):
    rss_start = _rss_mb()
    extra = {}
    if name == "baseline_python":
        pass
    elif name == "import_backend":
        import backend.main  # noqa: F401
    elif name.endswith("_model_loaded") or name.endswith("_embedding"):
        from scripts.benchmark_latency import _rgb, _samples
        from embeddings.pipelines import FacePipeline, VoicePipeline

        face, voice = _samples()
        m = name.split("_")[0]
        p = {"face": FacePipeline, "voice": VoicePipeline}[m]()
        if name.endswith("_embedding"):
            if m == "voice":
                from preprocessing.voice import load_wav_file

                w, sr = load_wav_file(voice)
                p.embed(w, sample_rate=sr)
            else:
                p.embed(_rgb(face))
    elif name == "template_generation":
        import tracemalloc

        from template_protection.biohash import generate_template
        from template_protection.hkdf_keys import derive_key

        key = derive_key("x", application_id="a", user_id="u", modality="face", key_version=1)
        v = np.random.default_rng(0).standard_normal(512)
        tracemalloc.start()
        generate_template(v, key, output_bits=256)
        extra["python_heap_peak_mb"] = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
    elif name.startswith("api_"):
        extra.update(_api(name))
    print(json.dumps({"rss_start_mb": rss_start, "rss_end_mb": _rss_mb(), "peak_mb": _peak_mb(), **extra}))


def _api(name):
    import io

    import cv2
    from scipy.io import wavfile

    from evaluation.ieee.common import EVAL_SECRET
    from scripts.benchmark_latency import _rgb, _samples

    tmp = Path(tempfile.mkdtemp(prefix="mem_"))
    os.environ.update({"DATABASE_URL": f"sqlite:///{tmp / 'm.db'}", "MASTER_SECRET": EVAL_SECRET, "DEBUG_SCORES": "false"})
    from fastapi.testclient import TestClient

    from backend.main import app
    from preprocessing.voice import load_wav_file

    face, voice = _samples()
    png = cv2.imencode(".png", cv2.cvtColor(_rgb(face), cv2.COLOR_RGB2BGR))[1].tobytes()
    w, sr = load_wav_file(voice)
    buf = io.BytesIO()
    wavfile.write(buf, sr, (np.asarray(w) * 32767).astype(np.int16))
    wav = buf.getvalue()
    with TestClient(app) as c:
        from backend.config import get_settings
        from backend.database.session import get_session_factory
        from backend.services.face_service import get_face_service
        from backend.services.voice_service import get_voice_service
        from embeddings.pipelines import FacePipeline, VoicePipeline

        s = get_settings()
        db = get_session_factory()()
        if name == "api_enroll_voice":
            r = c.post("/enroll", data={"user_id": "u", "modality": "voice", "application_id": s.application_id, "accept_low_quality": "true"},
                       files={"image": ("a.wav", wav, "audio/wav"), "confirm_image": ("b.wav", wav, "audio/wav")})
            return {"status": r.status_code}
        get_face_service()._store(db, FacePipeline().embed(_rgb(face)), "u", s.application_id)
        files = {"face_image": ("f.png", png, "image/png")}
        if name == "api_authenticate_face_plus_voice":
            get_voice_service()._store(db, VoicePipeline().embed(w, sample_rate=sr), "u", s.application_id)
            files["voice_audio"] = ("v.wav", wav, "audio/wav")
        r = c.post("/authenticate/fusion", data={"user_id": "u", "application_id": s.application_id}, files=files)
        return {"status": r.status_code}


def main():
    from evaluation.ieee.common import COLORS, COL_W, REAL, RESULTS, ieee_style, save_fig, write_csv

    rows = []
    for name in SCENARIOS:
        res = []
        for _ in range(3):
            out = subprocess.run([sys.executable, "-m", "scripts.benchmark_memory", "--worker", name], cwd=REPO,
                                 capture_output=True, text=True, timeout=900)
            res.append(json.loads(out.stdout.strip().splitlines()[-1]))
        row = {"scenario": name, "repeats": 3}
        for k in res[0]:
            vals = [r[k] for r in res if isinstance(r.get(k), (int, float))]
            if vals:
                row[f"{k}_mean"] = float(np.mean(vals))
                row[f"{k}_max"] = float(np.max(vals))
        rows.append(row)
        print(name, round(row["peak_mb_mean"]), "MB peak", flush=True)
    write_csv(RESULTS / "memory_benchmark.csv", rows, REAL)
    plt = ieee_style()
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    y = np.arange(len(rows))
    ax.barh(y, [r["peak_mb_mean"] for r in rows], color=COLORS["voice"], height=0.6)
    ax.set_yticks(y, [r["scenario"] for r in rows], fontsize=6)
    ax.invert_yaxis()
    ax.set(xlabel="peak working set (MB), fresh process, mean of 3", title="Memory per scenario")
    ax.grid(True, axis="x")
    save_fig(fig, "fig28_memory_benchmark", REAL, "Peak process memory per scenario (fresh process each, mean of 3 runs).")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()
