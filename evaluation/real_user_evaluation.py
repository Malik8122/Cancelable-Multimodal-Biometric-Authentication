"""Offline real-user evaluation harness (Part 5 of the IEEE evaluation plan).

Runs the SHIPPED enrollment and authentication code (backend/services/base_service.py::ModalityService,
template sets, HKDF keys, BioHash, modality_metrics decisions, fusion/policy.py) on a folder of consented,
anonymously-labelled captures, and writes only scores - never images, audio or embeddings.

Folder layout (participant folders MUST be anonymous IDs like P001; anything else is refused):

    <root>/consent.csv   participant_id,consent,withdrawn   (REQUIRED; only rows with consent=yes and withdrawn!=yes are used)
    <root>/P001/enroll/face_front.png face_left.png face_right.png face_up.png face_down.png
    <root>/P001/enroll/voice_1.wav voice_2.wav
    <root>/P001/session_1/face.png voice.wav
    <root>/P001/session_2/...                             (any number of sessions)

Protocol:
- enrollment exactly as in the product: 5 face poses -> centroid -> 4 template sets; two voice recordings with the
  consistency gate (FAIR accepted only with --accept-fair-voice);
- every session capture of every participant is compared with EVERY participant's enrolled templates under the
  TARGET's key (genuine when claimant == target, impostor otherwise) - the same pairing the live system performs;
- the probe is embedded once and scored with `ModalityService.authenticate_embedding` against each target.

Run:  python -m evaluation.real_user_evaluation --root <folder> --consent-confirmed
      python -m evaluation.real_user_evaluation --root <folder> --withdraw P003 [--delete-captures]
Outputs: evaluation/results/real_user_scores.csv, real_user_enrollment.csv, real_user_metrics.csv and
real_user_protocol_check.csv (evidence label REAL DATA). Protocol and data handling: evaluation/REAL_USER_PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np

from evaluation.ieee.common import EVAL_SECRET, REAL, RESULTS, full_report, rates, write_csv

PARTICIPANT_ID = re.compile(r"^P\d{3,}$")
FACE_POSES = ("front", "left", "right", "up", "down")
APP = "real-user-eval"
CONSENT_FILE = "consent.csv"
#: Protocol minimums (evaluation/REAL_USER_PROTOCOL.md).
MIN_GENUINE_ATTEMPTS = 10
MIN_SESSIONS = 2
OUTPUT_FILES = ("real_user_scores.csv", "real_user_enrollment.csv", "real_user_metrics.csv", "real_user_protocol_check.csv")


def _rgb(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"unreadable image: {path.name}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _wav(path: Path):
    from preprocessing.voice import load_wav_file

    return load_wav_file(path)


def default_pipelines():
    from embeddings.pipelines import FacePipeline, VoicePipeline

    class _Voice:
        def __init__(self):
            self._p = VoicePipeline()
            self.is_mock = self._p.is_mock

        def embed(self, raw):
            waveform, sr = raw
            return self._p.embed(waveform, sample_rate=sr)

    return {"face": FacePipeline(), "voice": _Voice()}


def read_consent(root: Path) -> dict[str, bool]:
    """participant_id -> usable (consent=yes and not withdrawn). The file is mandatory."""
    path = root / CONSENT_FILE
    if not path.exists():
        raise ValueError(f"{CONSENT_FILE} is missing under {root} - a per-participant consent record is required")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        pid = (r.get("participant_id") or "").strip()
        if not PARTICIPANT_ID.match(pid):
            raise ValueError(f"{CONSENT_FILE}: {pid!r} is not an anonymous ID like P001")
        out[pid] = (r.get("consent") or "").strip().lower() == "yes" and (r.get("withdrawn") or "").strip().lower() != "yes"
    return out


def discover(root: Path) -> dict[str, dict]:
    """Participants with a capture folder AND a usable consent record; folders without consent are skipped."""
    participants = {}
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    for folder in folders:
        if not PARTICIPANT_ID.match(folder.name):
            raise ValueError(f"participant folder {folder.name!r} is not an anonymous ID like P001 - refusing to run")
    consent = read_consent(root)
    for folder in folders:
        if not consent.get(folder.name, False):
            continue  # no consent, consent refused, or withdrawn: never read
        enroll = folder / "enroll"
        sessions = sorted((p for p in folder.iterdir() if p.is_dir() and p.name.startswith("session_")), key=lambda p: int(p.name.split("_")[1]))
        participants[folder.name] = {"enroll": enroll, "sessions": sessions}
    if not participants:
        raise ValueError(f"no participant folders under {root}")
    return participants


def enroll_participants(root: Path, pipelines=None, accept_fair_voice: bool = False):
    """Enroll every participant through the shipped services into a fresh in-memory DB.

    Returns (db, services, participants, enrolled, enrollment_log)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from backend.config import Settings
    from backend.database.models import Base
    from backend.services.base_service import ModalityService

    pipelines = pipelines or default_pipelines()
    for m, p in pipelines.items():
        if getattr(p, "is_mock", False):
            raise RuntimeError(f"{m} pipeline is running on a MOCK embedder - install the checkpoints (git lfs pull)")
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    settings = Settings(master_secret=EVAL_SECRET, template_pool_size=4)
    services = {m: ModalityService(m, p, settings) for m, p in pipelines.items()}
    participants = discover(root)

    enrolled: dict[str, set[str]] = {}
    enrollment_log = []
    for pid, info in participants.items():
        e, got = info["enroll"], set()
        poses = [(pose, _rgb(e / f"face_{pose}.png")) for pose in FACE_POSES if (e / f"face_{pose}.png").exists()]
        if poses:
            try:
                services["face"].enroll_poses(db, poses, pid, APP)
                got.add("face")
            except Exception as error:  # noqa: BLE001
                enrollment_log.append({"participant_id": pid, "modality": "face", "outcome": type(error).__name__})
        if (e / "voice_1.wav").exists() and (e / "voice_2.wav").exists():
            try:
                services["voice"].enroll_confirmed(db, _wav(e / "voice_1.wav"), _wav(e / "voice_2.wav"), pid, APP, accept_low_quality=accept_fair_voice)
                got.add("voice")
            except Exception as error:  # noqa: BLE001
                enrollment_log.append({"participant_id": pid, "modality": "voice", "outcome": type(error).__name__})
        enrolled[pid] = got
        for m in got:
            enrollment_log.append({"participant_id": pid, "modality": m, "outcome": "ENROLLED"})
    return db, services, participants, enrolled, enrollment_log


def run(root: Path, pipelines=None, accept_fair_voice: bool = False, out_dir: Path = RESULTS) -> tuple[list[dict], list[dict]]:
    from fusion.config import FusionPolicy
    from fusion.policy import evaluate_fusion_policy

    db, services, participants, enrolled, enrollment_log = enroll_participants(root, pipelines, accept_fair_voice)

    score_rows = []
    loaders = {"face": ("face.png", _rgb), "voice": ("voice.wav", _wav)}
    for claimant, info in participants.items():
        for session in info["sessions"]:
            s_id = int(session.name.split("_")[1])
            for modality, (fname, loader) in loaders.items():
                path = session / fname
                if not path.exists():
                    continue
                try:
                    emb = services[modality].embed(loader(path))
                except Exception as error:  # noqa: BLE001 - failure to acquire
                    score_rows.append({"participant_id": "", "claimant_id": claimant, "modality": modality, "session": s_id,
                                       "genuine": "", "failure_to_acquire": type(error).__name__})
                    continue
                for target, mods in enrolled.items():
                    if modality not in mods:
                        continue
                    r = services[modality].authenticate_embedding(db, emb, target, APP)
                    score_rows.append({
                        "participant_id": target, "claimant_id": claimant, "modality": modality, "session": s_id,
                        "genuine": claimant == target, "hamming_similarity": r.hamming_similarity,
                        "estimated_cosine": r.metric_value if r.metric == "cosine_estimate" else "",
                        "estimated_distance": r.metric_value if r.metric == "euclidean_estimate" else "",
                        "fusion_score": r.score, "fusion_threshold": r.threshold, "accepted": r.authenticated,
                        "template_set": r.template_set_version, "key_version": r.key_version,
                    })
    write_csv(out_dir / "real_user_scores.csv", score_rows, REAL)
    write_csv(out_dir / "real_user_enrollment.csv", enrollment_log, REAL)

    metrics = []
    valid = [r for r in score_rows if r.get("genuine") != ""]
    rules = {"face": ("estimated_cosine", True, 0.80), "voice": ("estimated_distance", False, 0.75)}
    for modality, (col, hib, thr) in rules.items():
        rr = [r for r in valid if r["modality"] == modality]
        g = np.array([r[col] for r in rr if r["genuine"]], float)
        i = np.array([r[col] for r in rr if not r["genuine"]], float)
        if len(g) and len(i):
            metrics.append({"scope": modality, **full_report(g, i, hib, thr),
                            "failure_to_acquire": sum(1 for r in score_rows if r.get("failure_to_acquire") and r["modality"] == modality)})
    attempts: dict[tuple, dict] = {}
    for r in valid:
        attempts.setdefault((r["participant_id"], r["claimant_id"], r["session"]), {})[r["modality"]] = r
    for name, policy in (("ALL_REQUIRED", FusionPolicy.ALL_REQUIRED), ("WEIGHTED", FusionPolicy.WEIGHTED)):
        gen_acc, imp_acc = [], []
        for (target, claimant, _), rs in attempts.items():
            if len(rs) < 2:
                continue
            d = evaluate_fusion_policy({m: r["fusion_score"] for m, r in rs.items()}, {m: r["accepted"] for m, r in rs.items()},
                                       policy, fusion_threshold=float(np.mean([r["fusion_threshold"] for r in rs.values()])))
            (gen_acc if target == claimant else imp_acc).append(d.authenticated)
        if gen_acc and imp_acc:
            g, i = np.array(gen_acc, float), np.array(imp_acc, float)
            metrics.append({"scope": f"fusion_{name}", **{k: v for k, v in rates(g, i, 0.5, True).items() if k != "threshold"},
                            "n_genuine": len(g), "n_impostor": len(i)})
    write_csv(out_dir / "real_user_metrics.csv", metrics, REAL)
    write_csv(out_dir / "real_user_protocol_check.csv", protocol_check(score_rows, enrolled), REAL)
    return score_rows, metrics


def protocol_check(score_rows: list[dict], enrolled: dict[str, set[str]]) -> list[dict]:
    """Per participant x modality: genuine attempts and sessions vs the protocol minimums."""
    out = []
    for pid, mods in sorted(enrolled.items()):
        for modality in ("face", "voice"):
            gen = [r for r in score_rows if r.get("genuine") is True and r["participant_id"] == pid and r["modality"] == modality]
            sessions = sorted({r["session"] for r in gen})
            out.append({"participant_id": pid, "modality": modality, "enrolled": modality in mods, "genuine_attempts": len(gen),
                        "sessions": len(sessions), "meets_protocol": modality in mods and len(gen) >= MIN_GENUINE_ATTEMPTS and len(sessions) >= MIN_SESSIONS})
    return out


def withdraw(participant_id: str, root: Path | None = None, out_dir: Path = RESULTS, delete_captures: bool = False) -> dict[str, int]:
    """Remove every output row of a withdrawn participant (as target or claimant); optionally delete their captures.
    Metrics must be re-run afterwards (they aggregate over participants). Returns rows removed per file."""
    if not PARTICIPANT_ID.match(participant_id):
        raise ValueError(f"{participant_id!r} is not an anonymous ID like P001")
    removed = {}
    for name in OUTPUT_FILES:
        path = out_dir / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames or [], list(reader)
        keep = [r for r in rows if participant_id not in (r.get("participant_id"), r.get("claimant_id"))]
        removed[name] = len(rows) - len(keep)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(keep)
    if delete_captures and root is not None and (root / participant_id).is_dir():
        shutil.rmtree(root / participant_id)
        removed["capture_folder_deleted"] = 1
    return removed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--consent-confirmed", action="store_true", help="confirm every participant gave informed consent")
    ap.add_argument("--accept-fair-voice", action="store_true")
    ap.add_argument("--withdraw", metavar="PID", help="remove this participant's rows from every output file")
    ap.add_argument("--delete-captures", action="store_true", help="with --withdraw: also delete <root>/<PID>")
    a = ap.parse_args()
    if a.withdraw:
        print(withdraw(a.withdraw, a.root, delete_captures=a.delete_captures), "- re-run the evaluation to refresh metrics")
        return
    if not a.consent_confirmed:
        raise SystemExit("Refusing to run without --consent-confirmed (informed consent for every participant).")
    rows, metrics = run(a.root, accept_fair_voice=a.accept_fair_voice)
    print(f"{len(rows)} comparisons, {len(metrics)} metric rows -> {RESULTS}")


if __name__ == "__main__":
    main()
