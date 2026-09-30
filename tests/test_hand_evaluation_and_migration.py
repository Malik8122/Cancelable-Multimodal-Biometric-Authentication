"""Hand gesture evaluation harness (on a SYNTHETIC participant folder - software test, not results) and the schema
upgrade that adds the hand template columns to an existing database."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.ieee.common import SYNTHETIC
from tests.hand_signals import Person, no_hand_capture


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _dataset(root: Path, consent_all: bool = True) -> None:
    lines = ["participant_id,consent,withdrawn"]
    for n in range(1, 4):
        pid, person = f"P00{n}", Person(n)
        lines.append(f"{pid},{'yes' if consent_all or n < 3 else 'no'},no")
        for k in range(1, 4):
            _write(root / pid / "enroll" / f"attempt_{k}.json", person.capture(100 * n + k))
        for s in (1, 2):
            for k in (1, 2):
                _write(root / pid / f"session_{s}" / f"attempt_{k}.json", person.capture(1000 * n + 10 * s + k))
    _write(root / "P001" / "session_2" / "attempt_3.json", no_hand_capture())
    (root / "consent.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_harness_computes_genuine_and_impostor_statistics(tmp_path):
    from evaluation.hand_gesture_evaluation import OUTPUT_FILES, run

    _dataset(tmp_path / "data")
    result = run(tmp_path / "data", out_dir=tmp_path / "out", evidence_label=SYNTHETIC)
    assert result["enrolled"] == ["P001", "P002", "P003"]
    scored = [r for r in result["scores"] if "genuine" in r]
    assert sum(r["genuine"] for r in scored) == 12 and sum(not r["genuine"] for r in scored) == 24
    assert [r["failure_to_acquire"] for r in result["scores"] if "failure_to_acquire" in r] == ["NO_HAND_DETECTED"]
    summary = result["metrics"][0]
    assert 0.0 <= summary["EER"] <= 1.0 and summary["failure_to_acquire"] == 1 and summary["rule"] == "dtw_distance <="
    assert {m["scope"] for m in result["metrics"]} == {"hand_gesture", "genuine_distance", "impostor_distance"}
    assert result["sweep"] and all(0 <= r["FAR"] <= 1 and 0 <= r["FRR"] <= 1 for r in result["sweep"])
    # every aggregation rule is compared; with < 4 participants there is no dev/test split, only pooled figures
    assert {c["aggregation"] for c in result["calibration"]} == {"min", "median", "mean"}
    assert all("calibrated_threshold" not in c for c in result["calibration"])
    assert all(len([k for k in r if k.startswith("dtw_sample_")]) == 3 for r in scored)
    for name in OUTPUT_FILES:
        text = (tmp_path / "out" / name).read_text(encoding="utf-8")
        assert SYNTHETIC in text and "landmarks" not in text


def test_harness_skips_participants_without_consent(tmp_path):
    from evaluation.hand_gesture_evaluation import run

    _dataset(tmp_path / "data", consent_all=False)
    assert run(tmp_path / "data", out_dir=tmp_path / "out", evidence_label=SYNTHETIC)["enrolled"] == ["P001", "P002"]


def test_harness_refuses_a_folder_without_a_consent_file(tmp_path):
    from evaluation.hand_gesture_evaluation import run

    _dataset(tmp_path / "data")
    (tmp_path / "data" / "consent.csv").unlink()
    with pytest.raises(ValueError, match="consent"):
        run(tmp_path / "data", out_dir=tmp_path / "out", evidence_label=SYNTHETIC)


def test_upgrade_adds_the_hand_columns_and_keeps_existing_templates(tmp_path, monkeypatch):
    import sqlite3

    from sqlalchemy import inspect, text

    from backend.config import get_settings
    from backend.database.migration import upgrade_schema
    from backend.database.session import get_engine, get_session_factory

    db_file = tmp_path / "old.db"
    connection = sqlite3.connect(db_file)
    connection.execute("CREATE TABLE users (id VARCHAR PRIMARY KEY, username VARCHAR, created_at DATETIME)")
    connection.execute(
        "CREATE TABLE protected_templates (template_id VARCHAR PRIMARY KEY, user_id VARCHAR, modality VARCHAR, "
        "application_id VARCHAR, template_version INTEGER, key_version INTEGER, output_bits INTEGER, protected_template BLOB, "
        "is_active BOOLEAN, created_at DATETIME, template_status VARCHAR, activation_time DATETIME, revoked_time DATETIME, "
        "revoked_reason VARCHAR, template_group_id VARCHAR, template_index INTEGER)")
    connection.execute("INSERT INTO users VALUES ('u', NULL, '2026-01-01 00:00:00')")
    for modality in ("face", "voice"):
        connection.execute("INSERT INTO protected_templates VALUES (?, 'u', ?, 'app', 1, 1, 256, ?, 1, '2026-01-01 00:00:00', "
                           "'ACTIVE', '2026-01-01 00:00:00', NULL, NULL, 'g', 1)", (f"{modality}-1", modality, bytes([7]) * 32))
    connection.commit()
    connection.close()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    for fn in (get_settings, get_engine, get_session_factory):
        fn.cache_clear()
    try:
        result = upgrade_schema(get_engine())
        engine = get_engine()
        columns = {c["name"] for c in inspect(engine).get_columns("protected_templates")}
        audit_columns = {c["name"] for c in inspect(engine).get_columns("audit_logs")}
        with engine.connect() as db:
            rows = db.execute(text("SELECT modality, template_format, hex(protected_template) FROM protected_templates")).all()
    finally:
        for fn in (get_settings, get_engine, get_session_factory):
            fn.cache_clear()
    assert {"template_format", "gesture_type", "template_metadata"} <= columns
    assert {"hand_similarity", "modality_diagnostics"} <= audit_columns
    assert {"protected_templates.template_format", "protected_templates.gesture_type"} <= set(result["columns_added"])
    assert sorted(rows) == [("face", None, "07" * 32), ("voice", None, "07" * 32)]  # face/voice data untouched


def test_testing_page_downloads_are_filed_into_the_harness_layout(tmp_path):
    from evaluation.hand_gesture_evaluation import import_downloads

    downloads = tmp_path / "dl"
    downloads.mkdir()
    for name in ("P001__enroll__attempt_1.json", "P002__session_3__attempt_2.json", "notes.json"):
        (downloads / name).write_text("{}", encoding="utf-8")
    moved = import_downloads(downloads, tmp_path / "root")
    assert sorted(p.relative_to(tmp_path / "root").as_posix() for p in moved) == [
        "P001/enroll/attempt_1.json", "P002/session_3/attempt_2.json"]
    assert (downloads / "notes.json").exists()


def test_impostor_folders_and_impostor_only_participants(tmp_path):
    """impostor_N probes are compared only with OTHER participants; a participant without an enroll folder contributes
    impostor probes only; the collection metadata's pipeline version is reported."""
    from evaluation.hand_gesture_evaluation import run

    root = tmp_path / "data"
    _dataset(root)
    _write(root / "P001" / "impostor_1" / "attempt_1.json",
           {**Person(1).capture(4242), "collection": {"sample_type": "impostor", "pipeline_version": "hand_gesture_v2"}})
    _write(root / "P004" / "impostor_1" / "attempt_1.json", Person(4).capture(4343))
    (root / "consent.csv").write_text((root / "consent.csv").read_text(encoding="utf-8") + "P004,yes,no\n", encoding="utf-8")
    result = run(root, out_dir=tmp_path / "out", evidence_label=SYNTHETIC)
    assert result["enrolled"] == ["P001", "P002", "P003"]
    impostor_rows = [r for r in result["scores"] if r.get("sample_type") == "impostor" and "genuine" in r]
    assert impostor_rows and not any(r["genuine"] for r in impostor_rows)
    assert not any(r["claimant_id"] == r["participant_id"] for r in impostor_rows)       # never vs own enrollment
    assert {r["claimant_id"] for r in impostor_rows} == {"P001", "P004"}
    assert all(r["pipeline_version"] == "hand_gesture_v2" and not r["pipeline_mismatch"]
               for r in impostor_rows if r["claimant_id"] == "P001")


def test_impostor_downloads_are_filed(tmp_path):
    from evaluation.hand_gesture_evaluation import import_downloads

    downloads = tmp_path / "dl"
    downloads.mkdir()
    (downloads / "P003__impostor_2__attempt_4.json").write_text("{}", encoding="utf-8")
    moved = import_downloads(downloads, tmp_path / "root")
    assert [p.relative_to(tmp_path / "root").as_posix() for p in moved] == ["P003/impostor_2/attempt_4.json"]
