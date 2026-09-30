"""Upgrade an existing database to the template-set schema.

    python scripts/migrate_to_multi_template.py            # uses DATABASE_URL / DATABASE_PATH / .env
    python scripts/migrate_to_multi_template.py --report   # print each user's template sets afterwards

What it does (idempotent; the same code runs automatically at app startup):
  * adds the new `protected_templates` / `audit_logs` columns and the
    one-live-template-per-set-and-modality index,
  * groups legacy rows into TEMPLATE SETS: a pre-pool user's active templates
    become set 1 (ACTIVE); the previous phase's per-modality pools (T1..TN)
    become sets 1..N, with the ACTIVE set being the highest version any
    modality was active at (drifted modalities are converged; see
    backend/database/migration.py::_backfill_template_sets),
  * leaves extra STANDBY sets to `POST /templates/{user_id}/generate`.

It cannot create STANDBY sets itself: that needs the user's embeddings, which
are never stored. A migrated pre-pool user therefore has a pool of one set until
they present a fresh capture that matches their ACTIVE set. Their existing
templates keep authenticating meanwhile.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import get_settings  # noqa: E402
from backend.database import crud  # noqa: E402
from backend.database.migration import upgrade_schema  # noqa: E402
from backend.database.session import get_engine, get_session_factory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true", help="print each user's template pool afterwards")
    args = parser.parse_args()

    settings = get_settings()
    result = upgrade_schema(get_engine())
    print(f"database: {settings.resolved_database_url.split('@')[-1]}")
    print(f"columns added: {result['columns_added'] or 'none (already up to date)'}")
    print(f"legacy templates backfilled: {result['legacy_templates_backfilled']}")

    print(f"template set rows backfilled: {result['template_set_rows_backfilled']}")
    print(f"retired modalities (fingerprint, iris) removed: {result['retired_modalities_removed']}")

    if args.report:
        from sqlalchemy import select

        from backend.database.models import User

        with get_session_factory()() as db:
            for user in db.execute(select(User)).scalars():
                for application_id in crud.get_application_ids(db, user.id):
                    sets = crud.get_template_sets(db, user.id, application_id)
                    summary = ", ".join(f"Set {s.version}:{s.status}[{'+'.join(s.modalities)}]" for s in sets)
                    print(f"  {user.id} / {application_id}: {summary}")


if __name__ == "__main__":
    main()
