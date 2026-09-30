"""Biometric-comparison security helpers.

Not a user session/login system - there is no OAuth, JWT, or password auth
anywhere in this project. "Authentication" throughout this codebase means
biometric verification (does this face/voice sample match what's
enrolled?), and this package holds the narrow security helper that supports
it: constant-time template comparison. See docs/BACKEND_API.md for the full
scope statement.
"""

from __future__ import annotations
