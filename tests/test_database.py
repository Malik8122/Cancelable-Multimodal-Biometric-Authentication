"""Offline tests for backend/database/{models,crud,migration}.py.

Uses the shared `db_session` fixture (tests/conftest.py) - a fresh in-memory
SQLite session per test, completely independent of
backend/database/session.py's cached, settings-derived engine - so these
tests never touch a real biometric.db file.
"""

from __future__ import annotations

from backend.database import crud
from backend.database.models import Base


def test_get_or_create_user_creates_once_and_reuses(db_session):
    first = crud.get_or_create_user(db_session, "U001", username="alice")
    second = crud.get_or_create_user(db_session, "U001", username="ignored-on-reuse")
    assert first.id == second.id == "U001"
    assert second.username == "alice"


def test_get_user_returns_none_when_missing(db_session):
    assert crud.get_user(db_session, "nobody") is None


def test_next_key_version_starts_at_one(db_session):
    crud.get_or_create_user(db_session, "U001")
    assert crud.next_key_version(db_session, "U001", "face", "capstone-demo") == 1


def test_save_template_then_next_key_version_increments(db_session):
    crud.get_or_create_user(db_session, "U001")
    crud.save_template(
        db_session,
        user_id="U001",
        modality="face",
        application_id="capstone-demo",
        template_version=1,
        key_version=1,
        output_bits=128,
        protected_template=b"\x01\x02",
    )
    assert crud.next_key_version(db_session, "U001", "face", "capstone-demo") == 2


def test_save_template_deactivates_previous_active_template(db_session):
    crud.get_or_create_user(db_session, "U001")
    first = crud.save_template(
        db_session,
        user_id="U001",
        modality="face",
        application_id="capstone-demo",
        template_version=1,
        key_version=1,
        output_bits=128,
        protected_template=b"\x01",
    )
    second = crud.save_template(
        db_session,
        user_id="U001",
        modality="face",
        application_id="capstone-demo",
        template_version=1,
        key_version=2,
        output_bits=128,
        protected_template=b"\x02",
    )

    db_session.refresh(first)
    assert first.is_active is False
    assert second.is_active is True
    active = crud.get_active_template(db_session, "U001", "face", "capstone-demo")
    assert active.template_id == second.template_id


def test_save_template_does_not_affect_other_modalities_or_applications(db_session):
    crud.get_or_create_user(db_session, "U001")
    face_template = crud.save_template(
        db_session, user_id="U001", modality="face", application_id="capstone-demo",
        template_version=1, key_version=1, output_bits=128, protected_template=b"\x01",
    )
    crud.save_template(
        db_session, user_id="U001", modality="voice", application_id="capstone-demo",
        template_version=1, key_version=1, output_bits=128, protected_template=b"\x02",
    )

    db_session.refresh(face_template)
    assert face_template.is_active is True


def test_get_templates_for_user_active_only_by_default(db_session):
    crud.get_or_create_user(db_session, "U001")
    crud.save_template(
        db_session, user_id="U001", modality="face", application_id="capstone-demo",
        template_version=1, key_version=1, output_bits=128, protected_template=b"\x01",
    )
    crud.save_template(
        db_session, user_id="U001", modality="face", application_id="capstone-demo",
        template_version=1, key_version=2, output_bits=128, protected_template=b"\x02",
    )

    active_only = crud.get_templates_for_user(db_session, "U001")
    everything = crud.get_templates_for_user(db_session, "U001", active_only=False)
    assert len(active_only) == 1
    assert len(everything) == 2


def test_delete_user_templates_removes_user_and_templates(db_session):
    crud.get_or_create_user(db_session, "U001")
    crud.save_template(
        db_session, user_id="U001", modality="face", application_id="capstone-demo",
        template_version=1, key_version=1, output_bits=128, protected_template=b"\x01",
    )

    deleted_count = crud.delete_user_templates(db_session, "U001")

    assert deleted_count == 1
    assert crud.get_user(db_session, "U001") is None
    assert crud.get_templates_for_user(db_session, "U001", active_only=False) == []


def test_delete_user_templates_returns_zero_for_missing_user(db_session):
    assert crud.delete_user_templates(db_session, "nobody") == 0


def test_schema_has_no_raw_image_or_embedding_columns():
    """Guardrail: the whole point of Phase 2 is that only protected templates
    are stored - fail loudly if a future edit ever adds a raw-data column."""
    protected_template_columns = {column.name for column in Base.metadata.tables["protected_templates"].columns}
    disallowed_substrings = ("image", "raw_embedding", "embedding_vector")
    for column_name in protected_template_columns:
        assert not any(bad in column_name.lower() for bad in disallowed_substrings), column_name
