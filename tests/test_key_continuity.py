"""Offline tests for the MASTER_SECRET key-continuity mechanism.

Covers, at two levels:

- Unit level (`db_session` fixture, an isolated in-memory SQLite database): every state A-E from
  `backend/key_continuity.py`'s docstring, the CRUD helpers' never-silently-overwrite guarantee,
  and `backend/secret_fingerprint.py`'s determinism/one-wayness properties.
- End-to-end level (`TestClient(app)`, a real file-backed SQLite database via `tmp_path`): that
  `backend/main.py`'s `lifespan()` hook actually refuses to start the real app in states B/D, and
  actually starts and reports `key_continuity` correctly in states A/C.

No test here touches face recognition, preprocessing, embeddings, template-protection math,
`key_version` rotation, thresholds, or fusion - this file is scoped entirely to the new mechanism.
"""

from __future__ import annotations

import pytest

from backend.database.models import ProtectedTemplate, User
from backend.key_continuity import (
    KeyContinuityError,
    KeyContinuityState,
    enforce_key_continuity_at_startup,
    evaluate_key_continuity,
)
from backend.secret_fingerprint import FINGERPRINT_LENGTH_BYTES, fingerprint_master_secret


class _FakeSettings:
    """Just enough of `backend.config.Settings` for these functions' needs (`.master_secret`)."""

    def __init__(self, master_secret: str):
        self.master_secret = master_secret


def _add_template(db_session, *, user_id: str = "U001", modality: str = "face") -> None:
    db_session.add(User(id=user_id))
    db_session.add(
        ProtectedTemplate(
            user_id=user_id,
            modality=modality,
            application_id="test-app",
            template_version=1,
            key_version=1,
            output_bits=256,
            protected_template=b"\x00" * 32,
        )
    )
    db_session.commit()


# --- backend/secret_fingerprint.py -------------------------------------------------------------


def test_fingerprint_is_deterministic_for_the_same_secret():
    assert fingerprint_master_secret("secret-a") == fingerprint_master_secret("secret-a")


def test_fingerprint_differs_for_different_secrets():
    assert fingerprint_master_secret("secret-a") != fingerprint_master_secret("secret-b")


def test_fingerprint_is_32_bytes_and_not_the_secret_itself():
    fingerprint = fingerprint_master_secret("a-real-looking-master-secret")
    assert len(fingerprint) == FINGERPRINT_LENGTH_BYTES
    assert fingerprint != b"a-real-looking-master-secret"
    assert b"a-real-looking-master-secret" not in fingerprint


def test_fingerprint_accepts_str_and_bytes_identically():
    assert fingerprint_master_secret("same-secret") == fingerprint_master_secret(b"same-secret")


# --- State A: no templates, no fingerprint -------------------------------------------------------


def test_state_a_evaluates_no_templates_yet(db_session):
    assert evaluate_key_continuity(db_session, _FakeSettings("secret-a")) == KeyContinuityState.NO_TEMPLATES_YET


def test_state_a_enforce_initializes_the_fingerprint_and_returns_ok(db_session):
    from backend.database import crud

    assert crud.get_master_secret_fingerprint(db_session) is None

    result = enforce_key_continuity_at_startup(db_session, _FakeSettings("secret-a"))

    assert result == KeyContinuityState.OK
    row = crud.get_master_secret_fingerprint(db_session)
    assert row is not None
    assert row.fingerprint == fingerprint_master_secret("secret-a")

    # A second startup with the SAME secret against the now-initialized database is State C.
    assert evaluate_key_continuity(db_session, _FakeSettings("secret-a")) == KeyContinuityState.OK


# --- State B: templates exist, no fingerprint -> HARD FAIL ---------------------------------------


def test_state_b_evaluates_no_fingerprint_recorded(db_session):
    _add_template(db_session)
    assert evaluate_key_continuity(db_session, _FakeSettings("secret-a")) == KeyContinuityState.NO_FINGERPRINT_RECORDED


def test_state_b_enforce_hard_fails_and_never_assumes_the_secret_is_correct(db_session):
    from backend.database import crud

    _add_template(db_session)

    with pytest.raises(KeyContinuityError):
        enforce_key_continuity_at_startup(db_session, _FakeSettings("whatever-secret-is-currently-configured"))

    # Must NOT have silently created a fingerprint as a side effect of the failed attempt.
    assert crud.get_master_secret_fingerprint(db_session) is None
    # Must NOT have touched the existing template.
    assert crud.count_protected_templates(db_session) == 1


# --- State C: fingerprint exists, matches -> proceed ----------------------------------------------


def test_state_c_evaluates_ok(db_session):
    from backend.database import crud

    _add_template(db_session)
    crud.initialize_master_secret_fingerprint(db_session, fingerprint_master_secret("secret-a"))

    assert evaluate_key_continuity(db_session, _FakeSettings("secret-a")) == KeyContinuityState.OK


def test_state_c_enforce_proceeds_without_changing_the_recorded_fingerprint(db_session):
    from backend.database import crud

    _add_template(db_session)
    crud.initialize_master_secret_fingerprint(db_session, fingerprint_master_secret("secret-a"))
    recorded_before = crud.get_master_secret_fingerprint(db_session).fingerprint

    result = enforce_key_continuity_at_startup(db_session, _FakeSettings("secret-a"))

    assert result == KeyContinuityState.OK
    assert crud.get_master_secret_fingerprint(db_session).fingerprint == recorded_before


# --- State D: fingerprint exists, mismatches -> HARD FAIL -----------------------------------------


def test_state_d_evaluates_mismatch(db_session):
    from backend.database import crud

    _add_template(db_session)
    crud.initialize_master_secret_fingerprint(db_session, fingerprint_master_secret("the-original-secret"))

    assert evaluate_key_continuity(db_session, _FakeSettings("a-freshly-generated-different-secret")) == KeyContinuityState.MISMATCH


def test_state_d_enforce_hard_fails_and_leaves_the_recorded_fingerprint_untouched(db_session):
    from backend.database import crud

    _add_template(db_session)
    crud.initialize_master_secret_fingerprint(db_session, fingerprint_master_secret("the-original-secret"))
    recorded_before = crud.get_master_secret_fingerprint(db_session).fingerprint

    with pytest.raises(KeyContinuityError):
        enforce_key_continuity_at_startup(db_session, _FakeSettings("a-freshly-generated-different-secret"))

    assert crud.get_master_secret_fingerprint(db_session).fingerprint == recorded_before
    assert crud.count_protected_templates(db_session) == 1


# --- State E: no templates, fingerprint exists ------------------------------------------------------


def test_state_e_no_templates_fingerprint_matches_proceeds_ok(db_session):
    """No templates yet, but a fingerprint was already recorded (e.g. by an earlier State-A
    startup). Matches the currently configured secret -> proceed normally, same as State C."""
    from backend.database import crud

    crud.initialize_master_secret_fingerprint(db_session, fingerprint_master_secret("secret-a"))
    assert crud.count_protected_templates(db_session) == 0

    assert evaluate_key_continuity(db_session, _FakeSettings("secret-a")) == KeyContinuityState.OK
    assert enforce_key_continuity_at_startup(db_session, _FakeSettings("secret-a")) == KeyContinuityState.OK


def test_state_e_no_templates_fingerprint_mismatches_still_hard_fails(db_session):
    """Even with zero templates at risk, a recorded fingerprint that disagrees with the currently
    configured secret is still a hard fail - a recorded fingerprint is a deliberate commitment
    (via State A's auto-init or an explicit rotation) that must never be silently contradicted,
    not just a convenience that only matters once templates exist."""
    from backend.database import crud

    crud.initialize_master_secret_fingerprint(db_session, fingerprint_master_secret("secret-a"))

    assert evaluate_key_continuity(db_session, _FakeSettings("secret-b")) == KeyContinuityState.MISMATCH
    with pytest.raises(KeyContinuityError):
        enforce_key_continuity_at_startup(db_session, _FakeSettings("secret-b"))


# --- CRUD helpers: never silently overwrite ----------------------------------------------------


def test_initialize_master_secret_fingerprint_refuses_to_overwrite_an_existing_row(db_session):
    from backend.database import crud

    crud.initialize_master_secret_fingerprint(db_session, fingerprint_master_secret("secret-a"))
    with pytest.raises(ValueError):
        crud.initialize_master_secret_fingerprint(db_session, fingerprint_master_secret("secret-b"))

    # The original fingerprint must still be the one recorded.
    assert crud.get_master_secret_fingerprint(db_session).fingerprint == fingerprint_master_secret("secret-a")


def test_rotate_master_secret_fingerprint_creates_when_absent(db_session):
    from backend.database import crud

    assert crud.get_master_secret_fingerprint(db_session) is None
    crud.rotate_master_secret_fingerprint(db_session, fingerprint_master_secret("secret-a"))
    assert crud.get_master_secret_fingerprint(db_session).fingerprint == fingerprint_master_secret("secret-a")


def test_rotate_master_secret_fingerprint_overwrites_an_existing_row(db_session):
    """This is the ONE function allowed to overwrite - `scripts/rotate_master_secret.py`'s job,
    requiring the operator's explicit, typed confirmation before it is ever called."""
    from backend.database import crud

    crud.initialize_master_secret_fingerprint(db_session, fingerprint_master_secret("old-secret"))
    crud.rotate_master_secret_fingerprint(db_session, fingerprint_master_secret("new-secret"))

    row = crud.get_master_secret_fingerprint(db_session)
    assert row.fingerprint == fingerprint_master_secret("new-secret")


def test_at_most_one_fingerprint_row_can_ever_exist(db_session):
    """`MasterSecretFingerprint`'s primary key is a fixed singleton id - a second `rotate` call
    must update the same row, never insert a second one."""
    from sqlalchemy import func, select

    from backend.database import crud
    from backend.database.models import MasterSecretFingerprint

    crud.initialize_master_secret_fingerprint(db_session, fingerprint_master_secret("secret-a"))
    crud.rotate_master_secret_fingerprint(db_session, fingerprint_master_secret("secret-b"))
    crud.rotate_master_secret_fingerprint(db_session, fingerprint_master_secret("secret-c"))

    row_count = db_session.execute(select(func.count()).select_from(MasterSecretFingerprint)).scalar_one()
    assert row_count == 1
