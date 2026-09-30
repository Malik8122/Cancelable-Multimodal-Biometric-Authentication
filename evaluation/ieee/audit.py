"""Parts 1 and 17: repository audit (configuration, API, schema, datasets) extracted automatically from the code,
and software validation (backend tests, frontend typecheck / lint / build).

Run: python -m evaluation.ieee.audit [audit|software]
Outputs: evaluation/REPOSITORY_AUDIT.md, evaluation/results/{configuration,dataset_audit,api_endpoints,db_schema}.csv,
         evaluation/results/software_validation.csv, evaluation/reports/software_validation.md
"""

from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from evaluation.ieee.common import CACHE, CONFIG, EVAL, EVAL_SECRET, REAL, REPO, REPORTS, RESULTS, SOFTWARE, timed, write_csv


def _where(obj_or_path, pattern: str) -> str:
    """'file:line' of the first line matching `pattern` in a module/file (so every value cites its source)."""
    path = Path(inspect.getsourcefile(obj_or_path)) if not isinstance(obj_or_path, (str, Path)) else REPO / obj_or_path
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if re.search(pattern, line):
            return f"{path.relative_to(REPO).as_posix()}:{n}"
    return path.relative_to(REPO).as_posix()


def configuration() -> list[dict]:
    os.environ.setdefault("MASTER_SECRET", EVAL_SECRET)
    import backend.services.recording_quality as rq
    import fusion.config as fc
    import models.face.inference as fi
    import models.voice.inference as vi
    import preprocessing.face as pf
    import preprocessing.voice as pv
    import template_protection.biohash as bh
    import template_protection.hkdf_keys as hk
    from backend.config import Settings
    from backend.services import face_enrollment as fe

    s = Settings(master_secret="x")
    cfg = "backend/config.py"
    rows = [
        ("Face embedding dimension", fi.FACE_EMBEDDING_DIM, _where(fi, r"FACE_EMBEDDING_DIM =")),
        ("Voice embedding dimension", vi.VOICE_EMBEDDING_DIM, _where(vi, r"VOICE_EMBEDDING_DIM =")),
        ("Embedding normalization", "L2 (unit norm)", _where("models/common/base_embedder.py", r"def _l2_normalize")),
        ("Template bits (runtime)", s.template_bits, _where(cfg, r"template_bits: int")),
        ("Template bits (library default, unused at runtime)", bh.DEFAULT_OUTPUT_BITS, _where(bh, r"DEFAULT_OUTPUT_BITS =")),
        ("Template format version", bh.TEMPLATE_FORMAT_VERSION, _where(bh, r"TEMPLATE_FORMAT_VERSION =")),
        ("Template sets per user (pool)", s.template_pool_size, _where(cfg, r"template_pool_size")),
        ("Face threshold (estimated cosine, accept >=)", s.face_cosine_threshold, _where(cfg, r"face_cosine_threshold:")),
        ("Voice threshold (estimated Euclidean distance, accept <=)", s.voice_euclidean_threshold, _where(cfg, r"voice_euclidean_threshold:")),
        ("Fallback Hamming threshold (no metric calibration)", s.match_threshold, _where(cfg, r"match_threshold: float")),
        ("HKDF hash", "SHA-256", _where(hk, r"hashes.SHA256")),
        ("HKDF seed length (bytes)", hk.SEED_LENGTH_BYTES, _where(hk, r"SEED_LENGTH_BYTES =")),
        ("HKDF salt", "SHA-256(application_id|user_id|modality|key_version)", _where(hk, r"hashlib.sha256\(context\)")),
        ("Quantization threshold sigma", "0.5 x std(projected)", _where("template_protection/transform.py", r"scale=0\.5 \* spread")),
        ("Projection", "Haar-random orthonormal rows (QR of seeded Gaussian), blocks of <= D rows", _where("template_protection/transform.py", r"np.linalg.qr")),
        ("Default fusion policy", fc.DEFAULT_FUSION_POLICY.value, _where(fc, r"DEFAULT_FUSION_POLICY =")),
        ("Fusion weights", "equal (1.0 each, renormalized)", _where("fusion/score_fusion.py", r"weights.get\(modality, 1.0\)")),
        ("WEIGHTED policy veto floor (fusion scale)", fc.DEFAULT_WEIGHTED_FLOOR, _where(fc, r"DEFAULT_WEIGHTED_FLOOR =")),
        ("Max upload size (bytes)", s.max_upload_size_bytes, _where(cfg, r"max_upload_size_bytes")),
        ("Face input size (px)", pf.FACE_INPUT_SIZE, _where(pf, r"FACE_INPUT_SIZE =")),
        ("Face enrollment poses", "front, left, right, up, down", _where(fe, r"FACE_POSES =")),
        ("Face minimum valid poses", fe.MIN_VALID_POSES, _where(fe, r"MIN_VALID_POSES =")),
        ("Face blur gate (Laplacian variance >=)", pf.BLUR_MIN_SHARPNESS, _where(pf, r"BLUR_MIN_SHARPNESS =")),
        ("Face detection confidence gate (>=)", pf.MIN_DETECTION_CONFIDENCE, _where(pf, r"MIN_DETECTION_CONFIDENCE =")),
        ("Face size ratio gate (>=)", pf.MIN_FACE_SIZE_RATIO, _where(pf, r"MIN_FACE_SIZE_RATIO =")),
        ("Face centre offset gate (<=)", pf.MAX_CENTER_OFFSET, _where(pf, r"MAX_CENTER_OFFSET =")),
        ("Face roll gate (degrees <=)", pf.MAX_ROLL_DEGREES, _where(pf, r"MAX_ROLL_DEGREES =")),
        ("Face yaw ratio gate (<=)", pf.MAX_YAW_RATIO, _where(pf, r"MAX_YAW_RATIO =")),
        ("Voice sample rate (Hz)", pv.TARGET_SAMPLE_RATE, _where(pv, r"TARGET_SAMPLE_RATE =")),
        ("Voice clip length (s)", pv.CLIP_SECONDS, _where(pv, r"CLIP_SECONDS =")),
        ("Voice mel bins", pv.N_MELS, _where(pv, r"N_MELS =")),
        ("Voice n_fft / hop", f"{pv.N_FFT} / {pv.HOP_LENGTH}", _where(pv, r"N_FFT =")),
        ("Voice enrollment consistency: EXCELLENT / GOOD / FAIR (cosine >=)", f"{rq.EXCELLENT_MIN} / {rq.GOOD_MIN} / {rq.FAIR_MIN}", _where(rq, r"EXCELLENT_MIN =")),
    ]
    out = [{"parameter": p, "value": v, "source": src, "evidence_label": CONFIG} for p, v, src in rows]
    write_csv(RESULTS / "configuration.csv", out, CONFIG)
    return out


def api_and_schema():
    os.environ.setdefault("MASTER_SECRET", EVAL_SECRET)
    from backend.database.models import Base
    from backend.main import app

    routes = []
    for r in app.routes:
        methods = sorted(getattr(r, "methods", []) - {"HEAD", "OPTIONS"}) if getattr(r, "methods", None) else []
        ep = getattr(r, "endpoint", None)
        if not methods or ep is None or r.path.startswith(("/docs", "/redoc", "/openapi")):
            continue
        routes.append({"method": ",".join(methods), "path": r.path, "handler": f"{ep.__module__}.{ep.__name__}",
                       "source": _where(inspect.getmodule(ep), rf"def {ep.__name__}\(")})
    write_csv(RESULTS / "api_endpoints.csv", routes, CONFIG)
    schema = []
    for t in Base.metadata.sorted_tables:
        for c in t.columns:
            schema.append({"table": t.name, "column": c.name, "type": str(c.type), "nullable": c.nullable, "primary_key": c.primary_key})
    write_csv(RESULTS / "db_schema.csv", schema, CONFIG)
    return routes, schema


def dataset_audit() -> list[dict]:
    rows = []
    for m in ("face", "voice"):
        path = CACHE / f"{m}_embeddings.npz"
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        labels = z["labels"]
        rows.append({"modality": m, "attempted": int(z["attempted"]), "embedded": len(labels),
                     "failures_to_acquire": int(len(z["failures"])), "identities": len(set(labels.tolist())),
                     "extraction_seconds": float(z["seconds"])})
    meta = {
        "face": ("LFW (funneled)", "public; UMass non-commercial research use", "identities with 2-19 images (disjoint from the >=20-image identities used for fine-tuning)",
                 "sklearn LFW download (figshare mirror)", "MTCNN 160x160 -> InceptionResnetV1"),
        "voice": ("VoxCeleb1 subset, Indian celebrities (Kaggle gaurav41/voxceleb1-audio-wav-files-for-india-celebrity)", "DbCL-1.0 as declared by the mirror",
                  "TEST split of the training code (per-speaker 70/15/15, seed 42): unseen utterances, speakers overlap training (closed-set)",
                  "Kaggle API", "16 kHz, VAD, RMS, 4 s, 80 log-mel -> ECAPA-TDNN"),
    }
    for r in rows:
        name, lic, split, src, pre = meta[r["modality"]]
        r.update({"dataset": name, "license": lic, "evaluation_split": split, "obtained_via": src, "preprocessing": pre,
                  "demographics": "NOT AVAILABLE", "evidence_label": REAL})
    write_csv(RESULTS / "dataset_audit.csv", rows, REAL)
    return rows


STAGES = [
    ("Registration (name)", "backend/api/user.py", "create_user", "display name (JSON)", "server user ID USER-xxxxxxxxxxxx", "-", "backend/display_names.py rules"),
    ("Face capture", "frontend/src/components/capture/GuidedFaceCapture.tsx", "GuidedFaceCapture", "webcam", "5 PNG poses", "-", "poses front/left/right/up/down"),
    ("Face preprocessing", "preprocessing/face.py", "FacePreprocessor.preprocess / detect_and_align", "RGB image", "160x160x3 crop", "160x160", "MTCNN margin 0, post_process"),
    ("Face enrollment gating", "embeddings/pipelines.py", "FacePipeline.embed_poses", "5 captures", "valid embeddings + report", "512", ">= 3 valid (MIN_VALID_POSES)"),
    ("Face embedding", "models/face/inference.py", "FaceEmbedder.extract_embedding", "160x160x3", "unit vector", "512", "InceptionResnetV1 checkpoint"),
    ("Face centroid", "embeddings/centroid.py", "centroid_embedding", "k unit vectors", "unit vector", "512", "normalize(mean)"),
    ("Voice preprocessing", "preprocessing/voice.py", "VoicePreprocessor.preprocess", "waveform + rate", "80 x frames log-mel", "80x401", "16 kHz, VAD, RMS 0.1, 4 s"),
    ("Voice embedding", "models/voice/inference.py", "VoiceEmbedder.extract_embedding", "log-mel", "unit vector", "192", "ECAPA-TDNN checkpoint"),
    ("Voice enrollment gate", "backend/services/base_service.py", "ModalityService.enroll_confirmed", "2 recordings", "templates or rejection", "192", "cosine >= 0.60 (FAIR opt-in), >= 0.75 GOOD"),
    ("Key derivation", "template_protection/hkdf_keys.py", "derive_key", "MASTER_SECRET, app, user, modality, key_version", "3 x 32-byte seeds", "-", "HKDF-SHA256"),
    ("Template generation", "template_protection/biohash.py", "generate_template", "embedding + key", "bit vector", "256", "project, quantize, permute"),
    ("Template storage", "backend/database/crud.py", "save_modality_templates", "4 templates per modality", "rows (BLOB 32 B)", "256 bits", "set 1 ACTIVE, 2-4 STANDBY"),
    ("Comparison", "template_protection/matcher.py", "compare(metric='hamming')", "two bit vectors", "Hamming similarity", "[0,1]", "constant-time equality shortcut"),
    ("Decision", "backend/services/modality_metrics.py", "decide", "Hamming similarity", "estimate, match, fusion score", "-", "face >= 0.80, voice <= 0.75, fp >= 0.90"),
    ("Fusion", "fusion/policy.py", "evaluate_fusion_policy", "per-modality scores + matches", "decision", "-", "ALL_REQUIRED default"),
    ("Revocation", "backend/database/crud.py", "revoke_active_set_and_promote", "user", "ACTIVE->REVOKED, STANDBY->ACTIVE", "-", "whole multimodal set"),
    ("Authentication orchestration", "backend/services/authentication.py", "authenticate_samples", "uploads", "AuthenticationOutcome + audit row", "-", "ENROLLMENT_REQUIRED if not enrolled"),
]


def write_audit():
    cfg = configuration()
    routes, schema = api_and_schema()
    ds = dataset_audit()
    lines = ["# Repository Audit", "", f"Generated by `python -m evaluation.ieee.audit` at commit `{_git('rev-parse', '--short', 'HEAD')}`.",
             "Every value below is read from the code at generation time; `source` is file:line.", "",
             "## Architecture (stage by stage)", "", "| stage | source file | function | input | output | dimension | configuration |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {a} | `{b}` | `{c}` | {d} | {e} | {f} | {g} |" for a, b, c, d, e, f, g in STAGES]
    lines += ["", "Flows: enrollment = registration -> face poses + voice recordings -> embeddings -> 4 HKDF keys -> 4 BioHash templates -> storage;",
              "authentication = capture -> embedding -> key of the claimed user's ACTIVE set -> BioHash -> Hamming vs stored template -> decision -> fusion.",
              "", f"## Configuration ({len(cfg)} parameters, label {CONFIG})", "", "| parameter | value | source |", "|---|---|---|"]
    lines += [f"| {r['parameter']} | {r['value']} | `{r['source']}` |" for r in cfg]
    lines += ["", f"## API endpoints ({len(routes)})", "", "| method | path | handler | source |", "|---|---|---|---|"]
    lines += [f"| {r['method']} | `{r['path']}` | `{r['handler']}` | `{r['source']}` |" for r in routes]
    tables = sorted({r["table"] for r in schema})
    lines += ["", f"## Database schema ({len(tables)} tables)", ""]
    for t in tables:
        cols = [r for r in schema if r["table"] == t]
        lines.append(f"- **{t}**: " + ", ".join(f"{c['column']} ({c['type']}{', PK' if c['primary_key'] else ''})" for c in cols))
    lines += ["", "No table stores images, audio or embeddings; `protected_templates.protected_template` holds the packed 256-bit template.",
              "", "## Datasets used by this evaluation", "", "| modality | dataset | split | embedded / attempted | identities | failures to acquire | license |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['modality']} | {r['dataset']} | {r['evaluation_split']} | {r['embedded']} / {r['attempted']} | {r['identities']} | {r['failures_to_acquire']} | {r['license']} |" for r in ds]
    lines += ["", "Training data of the shipped checkpoints (documented, not re-run here): face = LFW identities with >= 20 images (62 identities, 3,023 images);",
              "voice = the same VoxCeleb1 subset (train split).", ""]
    (EVAL / "REPOSITORY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def software_validation():
    rows, log = [], []

    def run(name, cmd, cwd=REPO, shell=False):
        t = time.perf_counter()
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=shell, timeout=3600)
        secs = time.perf_counter() - t
        out = (p.stdout or "") + (p.stderr or "")
        shown = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
        shown = shown.replace(sys.executable, "python").replace(str(REPO) + os.sep, "").replace(str(REPO), ".")  # no machine paths
        out = out.replace(str(REPO) + os.sep, "").replace(str(REPO), ".")
        log.append(f"## {name}\n\n`{shown}` -> exit {p.returncode} ({secs:.0f}s)\n\n```\n{out[-2500:]}\n```\n")
        return p.returncode, out, secs

    junit = REPORTS / "pytest_junit.xml"
    code, out, secs = run("backend tests", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"])
    import xml.etree.ElementTree as ET

    suite = ET.parse(junit).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    total, failures, errors, skipped = (int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
    rows.append({"check": "backend pytest", "total": total, "passed": total - failures - errors - skipped, "failed": failures + errors,
                 "skipped": skipped, "exit_code": code, "seconds": secs})
    try:
        import pytest_cov  # noqa: F401
        coverage = "available"
    except ImportError:
        coverage = "NOT AVAILABLE (pytest-cov not installed)"
    rows.append({"check": "coverage", "result": coverage})
    npx = "npx.cmd" if os.name == "nt" else "npx"
    npm = "npm.cmd" if os.name == "nt" else "npm"
    fe = REPO / "frontend"
    code, out, secs = run("frontend typecheck", [npx, "tsc", "-b"], cwd=fe)
    rows.append({"check": "frontend typecheck (tsc -b)", "errors": len(re.findall(r"error TS", out)), "exit_code": code, "seconds": secs})
    code, out, secs = run("frontend lint", [npm, "run", "lint"], cwd=fe)
    m = re.search(r"Found (\d+) warnings? and (\d+) errors?", out)
    rows.append({"check": "frontend lint (oxlint)", "warnings": m.group(1) if m else len(re.findall(r"warning", out)),
                 "errors": m.group(2) if m else len(re.findall(r" error ", out)), "exit_code": code, "seconds": secs})
    code, out, secs = run("frontend build", [npm, "run", "build"], cwd=fe)
    rows.append({"check": "frontend build (vite build)", "result": "success" if code == 0 else "FAILED", "exit_code": code, "seconds": secs})
    rows.append({"check": "frontend unit tests", "result": "NOT AVAILABLE (no frontend test suite in package.json)"})
    write_csv(RESULTS / "software_validation.csv", rows, SOFTWARE)
    (REPORTS / "software_validation.md").write_text("# Software validation\n\nEvidence label: SOFTWARE VALIDATION (not biometric performance).\n\n"
                                                    + "\n".join(log), encoding="utf-8")
    return rows


if __name__ == "__main__":
    what = sys.argv[1:] or ["audit", "software"]
    if "audit" in what:
        with timed("part1_audit"):
            write_audit()
    if "software" in what:
        with timed("part17_software"):
            software_validation()
