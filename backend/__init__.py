"""FastAPI backend: enrollment/authentication REST API over protected templates.

See docs/BACKEND_API.md for the endpoint reference. This package depends on
`template_protection/` and `embeddings/` (via `backend/services/`); the
dependency direction is strictly one-way - neither of those packages import
anything from `backend/`.
"""

from __future__ import annotations
