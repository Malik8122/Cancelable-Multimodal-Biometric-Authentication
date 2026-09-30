"""Explicitly acknowledge a MASTER_SECRET rotation / new key epoch.

    python scripts/rotate_master_secret.py

Deliberately NOT run automatically at server startup. `backend/main.py`'s startup hook
(`backend/key_continuity.py::enforce_key_continuity_at_startup`) only ever *reads* the recorded
MASTER_SECRET fingerprint and hard-fails the server on a mismatch or a missing fingerprint when
protected templates already exist - it never writes one in that situation. This script is the
one place that creates or overwrites the recorded fingerprint for a database that already has
templates, and it only ever does so after an interactive operator types out an explicit
confirmation phrase acknowledging the exact consequence:

    every previously enrolled biometric template remains in the database exactly as it is -
    nothing is deleted or modified - but will never again verify under the newly recorded
    secret unless it happens to be the same secret that protected them. This is the same
    outcome as a lost password, and this script cannot undo it.

This script never prints MASTER_SECRET or the derived fingerprint, and never touches
`protected_templates` - it only reads (for the summary printed below) and writes the single
`master_secret_fingerprint` row.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import get_settings  # noqa: E402
from backend.database import crud  # noqa: E402
from backend.database.session import get_session_factory, init_db  # noqa: E402
from backend.secret_fingerprint import fingerprint_master_secret  # noqa: E402

CONFIRMATION_PHRASE = "I understand this invalidates existing biometric templates"


def main() -> None:
    # Raises (via pydantic-settings) if MASTER_SECRET is unset at all - "verify MASTER_SECRET is
    # configured" is already this repository's existing required-field behavior; nothing new to
    # duplicate here.
    settings = get_settings()
    init_db()

    with get_session_factory()() as db:
        existing = crud.get_master_secret_fingerprint(db)
        template_count = crud.count_protected_templates(db)

    print("MASTER_SECRET rotation / key-epoch acknowledgment")
    print("=" * 60)
    print(f"Database: {settings.resolved_database_url.split('@')[-1]}")
    print(f"Existing protected templates: {template_count}")
    print(f"Fingerprint currently recorded: {'yes' if existing is not None else 'no'}")
    print()
    print(
        "Recording the currently configured MASTER_SECRET as the trusted one means: if it does\n"
        "not match whatever secret protected the existing templates counted above, those\n"
        "templates will remain in the database exactly as they are (nothing is deleted or\n"
        "modified) but will never again verify - the same outcome as a lost password. This\n"
        "cannot be undone by this script."
    )
    print()
    print(f'Type exactly: "{CONFIRMATION_PHRASE}"')
    typed = input("> ").strip()
    if typed != CONFIRMATION_PHRASE:
        print("Confirmation phrase did not match. Nothing was changed.")
        raise SystemExit(1)

    with get_session_factory()() as db:
        crud.rotate_master_secret_fingerprint(db, fingerprint_master_secret(settings.master_secret))

    print()
    print("Recorded. The currently configured MASTER_SECRET is now the trusted one for this database.")
    print("(Neither the secret nor the fingerprint value is ever printed by this script.)")


if __name__ == "__main__":
    main()
