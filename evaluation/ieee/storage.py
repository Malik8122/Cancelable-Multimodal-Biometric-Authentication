"""Part 12: storage of the shipped schema, measured on fresh SQLite files.

Users are enrolled through `ModalityService._store` (the exact enrollment write path: 4 template sets per user, one
256-bit template per modality per set). Storage size does not depend on embedding values, so random unit vectors are
used; the measurement is of the software artefact, hence SOFTWARE VALIDATION; byte arithmetic is DERIVED.
Run: python -m evaluation.ieee.storage
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from evaluation.ieee.common import COLORS, COL_W, DERIVED, EVAL_SECRET, RESULTS, SOFTWARE, ieee_style, save_fig, timed, write_csv


def _db(path: Path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from backend.database.models import Base

    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)()


def _size(path: Path) -> dict:
    con = sqlite3.connect(path)
    con.execute("VACUUM")
    info = {"file_bytes": path.stat().st_size}
    try:
        for name, size in con.execute("SELECT name, SUM(pgsize) FROM dbstat GROUP BY name"):
            info[f"table_bytes:{name}"] = size
    except sqlite3.OperationalError:
        pass
    for t in ("users", "protected_templates", "audit_logs"):
        info[f"rows:{t}"] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    con.close()
    return info


def run():
    from backend.config import Settings
    from backend.services.base_service import ModalityService
    from backend.utils import record_authentication_audit

    class _P:
        is_mock = False

        def embed(self, x):
            return x

    settings = Settings(master_secret=EVAL_SECRET)
    svc = {m: ModalityService(m, _P(), settings) for m in ("face", "voice")}
    for service in svc.values():
        service.capture_gate = False  # stub input is an embedding/seed, not audio: no capture gate
    dims = {"face": 512, "voice": 192}
    rng = np.random.default_rng(0)
    rows = []
    tmp = Path(tempfile.mkdtemp(prefix="storage_"))
    empty_path = tmp / "empty.db"
    _db(empty_path)
    base = _size(empty_path)["file_bytes"]
    for n in (1, 10, 100, 1000):
        path = tmp / f"users_{n}.db"
        _, db = _db(path)
        for u in range(n):
            for m in ("face", "voice"):
                v = rng.standard_normal(dims[m])
                svc[m]._store(db, v / np.linalg.norm(v), f"USER-{u:06d}", settings.application_id)
        db.close()
        s = _size(path)
        rows.append({"experiment": "enrolled_users_face_voice_4_sets", "users": n, **s,
                     "bytes_per_user_incremental": (s["file_bytes"] - base) / n, "evidence_label": SOFTWARE})
        if n == 1000:
            _, db = _db(path)
            import time

            for k in range(1000):
                record_authentication_audit(
                    db, user_id=f"USER-{k % n:06d}", building_id="defence_research_lab", modality_list=["face", "voice"],
                    similarity_scores={"face": 0.86, "voice": 0.81}, thresholds_used={"face": 0.8, "voice": 0.71875},
                    authenticated=True, started_at=time.perf_counter(), template_versions={"face": 1, "voice": 1},
                    key_versions={"face": 1, "voice": 1}, fusion_score=0.835, fusion_policy="ALL_REQUIRED",
                    fusion_similarity=0.835, template_set_version=1, template_set_status="ACTIVE",
                    authentication_state="ACCESS_GRANTED", submitted_modalities=["face", "voice"],
                    enrolled_modalities=["face", "voice"], authenticated_modalities=["face", "voice"],
                )
            db.close()
            s2 = _size(path)
            rows.append({"experiment": "plus_1000_audit_rows", "users": n, **s2,
                         "bytes_per_audit_row": (s2["file_bytes"] - s["file_bytes"]) / 1000, "evidence_label": SOFTWARE})
    rows += [
        {"experiment": "template_payload_per_modality", "bytes": 256 // 8, "note": "256 bits packed", "evidence_label": DERIVED},
        {"experiment": "template_payload_per_user_face_voice_4_sets", "bytes": 2 * 4 * 32, "evidence_label": DERIVED},
        {"experiment": "raw_embedding_float32_equivalent_face_voice", "bytes": (512 + 192) * 4,
         "note": "what storing embeddings would cost - NOT stored by this system", "evidence_label": DERIVED},
    ]
    write_csv(RESULTS / "storage_analysis.csv", rows, SOFTWARE)
    plt = ieee_style()
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    labels = ["256-bit template\n(1 modality)", "templates per user\n(2 mod. x 4 sets)", "raw float32 embeddings\n(face+voice, not stored)",
              "measured DB bytes\nper user (incl. metadata)"]
    per_user = next(r["bytes_per_user_incremental"] for r in rows if r.get("users") == 1000 and "bytes_per_user_incremental" in r)
    vals = [32, 256, (512 + 192) * 4, per_user]
    ax.bar(range(4), vals, color=[COLORS["protected"], COLORS["protected"], COLORS["raw"], COLORS["face"]])
    ax.set_xticks(range(4), labels, fontsize=5.5)
    ax.set(yscale="log", ylabel="bytes (log)", title="Storage comparison")
    for x, v in enumerate(vals):
        ax.text(x, v * 1.1, f"{v:,.0f}", ha="center", fontsize=6)
    save_fig(fig, "fig20_storage_comparison", SOFTWARE, "Template payload vs raw embedding size vs measured per-user database growth (SQLite, 1000 users).")


if __name__ == "__main__":
    with timed("part12_storage"):
        run()
