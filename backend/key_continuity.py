"""Key-continuity check: does the currently configured MASTER_SECRET match the one that
protected any biometric templates already in this database?

`template_protection/hkdf_keys.py::derive_key` derives every template's key material from
`Settings.master_secret`. That secret is never stored anywhere - it lives only in the
environment/`.env` of whichever process is running. Nothing previously verified that a *present*
`MASTER_SECRET` is the *correct* one for a database that already holds templates: any
syntactically valid string was accepted, the server started and reported healthy, and
authentication then failed at chance-level similarity for every enrolled user - a real incident,
not a hypothetical (see docs/AUTHENTICATION_RELIABILITY_REPORT.md).

Five states, evaluated from (whether a `MasterSecretFingerprint` row exists, whether it matches
the currently configured secret, whether any `ProtectedTemplate` rows exist):

    A. no templates, no fingerprint   -> safe to auto-initialize (nothing at stake yet).
    B. templates exist, no fingerprint -> HARD FAIL. Never assume the current secret is correct.
    C. fingerprint exists, matches    -> proceed.
    D. fingerprint exists, mismatches -> HARD FAIL, regardless of whether templates exist yet -
                                          a recorded fingerprint is a deliberate commitment
                                          (via State A's auto-init or an explicit rotation) that
                                          must not be silently contradicted.
    E. no templates, fingerprint exists -> the "no templates" sub-case of C/D: proceeds if it
                                          matches, exactly like C (see D above for why a mismatch
                                          here is still a hard fail, not silently ignored just
                                          because nothing has been enrolled yet).

Read-only (`evaluate_key_continuity`) vs. enforcing (`enforce_key_continuity_at_startup`) are
kept separate: `GET /system/health` only ever reads state, `backend/main.py`'s startup hook is
the one place allowed to perform State A's one-time initialization or raise `KeyContinuityError`.
"""

from __future__ import annotations

import enum
import hmac

from sqlalchemy.orm import Session

from backend.config import Settings
from backend.database import crud
from backend.secret_fingerprint import fingerprint_master_secret


class KeyContinuityState(str, enum.Enum):
    #: Fingerprint recorded and matches the currently configured MASTER_SECRET (State C, and
    #: State E when it matches).
    OK = "ok"
    #: Protected templates exist but no fingerprint has ever been recorded (State B) - the exact
    #: ambiguous situation this module exists to catch, rather than silently guessing.
    NO_FINGERPRINT_RECORDED = "no_fingerprint_recorded"
    #: A fingerprint is recorded but does not match the currently configured MASTER_SECRET
    #: (State D, and State E when it mismatches).
    MISMATCH = "mismatch"
    #: No templates and no fingerprint recorded yet (State A, pre-initialization) - only
    #: observable before `enforce_key_continuity_at_startup` has ever run against this database.
    NO_TEMPLATES_YET = "no_templates_yet"


class KeyContinuityError(RuntimeError):
    """Raised at startup when MASTER_SECRET cannot be trusted against this database's existing
    protected templates. Never includes the secret or the fingerprint bytes - only the state and
    plain counts, both safe to log or print."""


def evaluate_key_continuity(db: Session, settings: Settings) -> KeyContinuityState:
    """Read-only: which of the four reportable states `(db, settings.master_secret)` is in.

    Never writes anything - callers decide what to do with the result. `GET /system/health` uses
    this directly (see `backend/api/system.py`); `enforce_key_continuity_at_startup` below uses it
    too, then acts on the result.
    """
    row = crud.get_master_secret_fingerprint(db)
    if row is not None:
        current = fingerprint_master_secret(settings.master_secret)
        return KeyContinuityState.OK if hmac.compare_digest(current, row.fingerprint) else KeyContinuityState.MISMATCH

    if crud.count_protected_templates(db) > 0:
        return KeyContinuityState.NO_FINGERPRINT_RECORDED
    return KeyContinuityState.NO_TEMPLATES_YET


def enforce_key_continuity_at_startup(db: Session, settings: Settings) -> KeyContinuityState:
    """Startup-only. Evaluates, then either self-initializes (State A) or hard-fails (B/D).

    Called once from `backend/main.py`'s `lifespan()`, after `init_db()`. Raising here prevents
    uvicorn from completing startup at all - the server never begins accepting requests with a
    MASTER_SECRET that can't be verified against existing templates.
    """
    state = evaluate_key_continuity(db, settings)

    if state is KeyContinuityState.NO_TEMPLATES_YET:
        # State A: nothing recorded, nothing enrolled yet - the one case where recording a
        # fingerprint automatically is safe, because nothing is at risk of being silently
        # mismatched. From this point on, this same database is in State C on every future start
        # (or, if MASTER_SECRET later changes, State D - a hard fail, not a silent drift).
        crud.initialize_master_secret_fingerprint(db, fingerprint_master_secret(settings.master_secret))
        return KeyContinuityState.OK

    if state is KeyContinuityState.NO_FINGERPRINT_RECORDED:
        raise KeyContinuityError(
            "Protected biometric templates exist in this database, but no MASTER_SECRET "
            "fingerprint has ever been recorded for it. Refusing to start: authenticating with "
            "the currently configured MASTER_SECRET would silently fail for every already-"
            "enrolled user if it is not the exact secret that protected those templates - "
            "there is no way to verify that from here. If this MASTER_SECRET is intentionally a "
            "new key epoch (e.g. the original secret is unrecoverable), run "
            "`python scripts/rotate_master_secret.py` to acknowledge this explicitly before "
            "starting the server again. This never modifies, revokes, or deletes any existing "
            "protected template - it only records which secret is now the current one."
        )

    if state is KeyContinuityState.MISMATCH:
        raise KeyContinuityError(
            "The currently configured MASTER_SECRET does not match the fingerprint already "
            "recorded for this database. Refusing to start: this would silently authenticate "
            "every enrolled user against effectively-random data instead of a real biometric "
            "comparison. If this MASTER_SECRET change is intentional, run "
            "`python scripts/rotate_master_secret.py` to acknowledge the new key epoch "
            "explicitly before starting the server again."
        )

    return state  # OK
