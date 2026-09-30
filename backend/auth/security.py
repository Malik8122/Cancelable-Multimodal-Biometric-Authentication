"""Constant-time comparison for biometric-template matching.

Re-exports `template_protection.utils.constant_time_equals` rather than
reimplementing it, so there is exactly one implementation of
"compare two byte strings without leaking timing information about where
they differ" in the whole codebase; `template_protection.matcher.compare`
already uses it internally for the exact-match fast path, and this
re-export exists for any `backend/api/*.py` code that needs the same check
directly (e.g. comparing an uploaded template_id against a stored one)
without importing across the backend/template_protection boundary by hand
each time.
"""

from __future__ import annotations

from template_protection.utils import constant_time_equals

__all__ = ["constant_time_equals"]
