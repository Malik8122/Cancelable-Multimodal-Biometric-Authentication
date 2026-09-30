"""Human-readable display names for users.

A display name is only a label for people ("Sanya Malik"). It is never the user's identity: the internal
`users.id` stays the primary key, the key-derivation input and the only thing templates are attached to, so two
people can share a name. The name is not a biometric secret and never enters any biometric computation.

Stored in the existing nullable `users.username` column, so no schema migration is needed. Users enrolled before
names existed have NULL there; `fallback_display_name` gives them a stable "User <short id>" label instead.
"""

from __future__ import annotations

MAX_DISPLAY_NAME_LENGTH = 64

#: Punctuation that appears in real names (O'Brien, Jean-Luc, J. R. R.) besides letters and single spaces.
_NAME_PUNCTUATION = frozenset("'’-.")


class InvalidDisplayName(ValueError):
    """The submitted name fails validation (mapped to HTTP 422)."""


def normalize_display_name(raw: str) -> str:
    """Trim, collapse runs of whitespace to one space, and validate. Returns the name to store.

    Accepted: letters from any script, single spaces between words, and ' ’ - . inside a name. It must start with a
    letter and be at most `MAX_DISPLAY_NAME_LENGTH` characters. Rejected: empty / whitespace-only input, digits,
    symbols, control characters and anything longer.
    """
    name = " ".join(raw.split())
    if not name:
        raise InvalidDisplayName("Please enter your name.")
    if len(name) > MAX_DISPLAY_NAME_LENGTH:
        raise InvalidDisplayName(f"The name must be at most {MAX_DISPLAY_NAME_LENGTH} characters.")
    if not name[0].isalpha():
        raise InvalidDisplayName("The name must start with a letter.")
    if any(not (ch.isalpha() or ch == " " or ch in _NAME_PUNCTUATION) for ch in name):
        raise InvalidDisplayName("The name may only contain letters, spaces, apostrophes, hyphens and periods.")
    return name


def fallback_display_name(user_id: str) -> str:
    """Label for a user with no stored name (enrolled before names existed): "User <last 6 id characters>"."""
    return f"User {user_id[-6:].upper()}"
