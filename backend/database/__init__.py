"""SQLAlchemy models, session management, and CRUD for protected templates.

Nothing in this package ever defines a column for a raw image or a raw
(unprotected) embedding - see models.py::ProtectedTemplate.
"""

from __future__ import annotations
