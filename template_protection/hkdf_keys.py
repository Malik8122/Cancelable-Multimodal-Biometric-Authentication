"""HKDF-SHA256 key derivation for cancelable biometric templates.

Every protected template is generated from key material that is derived, not
stored — nothing here ever persists a key to disk. Given the same
(master secret, application ID, user ID, modality, key version), derivation
is fully deterministic, which is what lets re-authentication reproduce the
same protected template as enrollment did. Changing any one of those inputs
yields independent, unlinkable key material: changing `key_version` alone is
the revocation mechanism (see `revoke.py`); changing `application_id` for the
same user/modality is the diversity mechanism (see `docs/TEMPLATE_PROTECTION.md`).

The master secret is a caller-supplied argument, never read from the
environment here. `template_protection` has no dependency on `backend/` —
`backend/config.py` reads `MASTER_SECRET` and passes it in, keeping the
dependency direction one-way (backend -> template_protection) and this
module independently testable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

#: Length, in bytes, of each derived seed. 32 bytes (256 bits) gives each
#: seed enough entropy to seed a NumPy PRNG (which only consumes a subset of
#: this) with headroom to spare.
SEED_LENGTH_BYTES = 32


@dataclass(frozen=True)
class KeyMaterial:
    """The three independent seeds behind one cancelable template.

    Each seed feeds exactly one stage of `transform.py`'s pipeline:

    - `projection_seed`  -> the random orthonormal projection matrix.
    - `permutation_seed` -> the user-specific bit permutation.
    - `threshold_seed`   -> the per-position keyed quantization thresholds.

    Deriving three independent seeds (rather than reusing one seed for every
    stage) means recovering one stage's randomness doesn't help an attacker
    predict another stage's.
    """

    projection_seed: bytes
    permutation_seed: bytes
    threshold_seed: bytes
    key_version: int


def _hkdf_sha256(master_secret: bytes, salt: bytes, info: bytes, length: int = SEED_LENGTH_BYTES) -> bytes:
    """One HKDF-SHA256 extract-then-expand call.

    `cryptography`'s `HKDF` object performs both the RFC 5869 "extract" and
    "expand" steps in a single `derive()` call and is single-use, so each
    distinct (salt, info) pair needs its own instance.
    """
    hkdf = HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info)
    return hkdf.derive(master_secret)


def derive_key(
    master_secret: bytes | str,
    *,
    application_id: str,
    user_id: str,
    modality: str,
    key_version: int = 1,
) -> KeyMaterial:
    """Derive the key material for one (user, modality, application, version) context.

    The HKDF "salt" is a SHA-256 hash of the four identifying fields (rather
    than the raw, variable-length strings) so the salt has a fixed length and
    no delimiter-collision risk (e.g. user_id="a|b" colliding with a
    literal separator). The HKDF "info" field distinguishes the three seeds
    drawn from the same salt.
    """
    if isinstance(master_secret, str):
        master_secret = master_secret.encode("utf-8")

    context = f"{application_id}|{user_id}|{modality}|{key_version}".encode("utf-8")
    salt = hashlib.sha256(context).digest()

    return KeyMaterial(
        projection_seed=_hkdf_sha256(master_secret, salt, b"template_protection/projection"),
        permutation_seed=_hkdf_sha256(master_secret, salt, b"template_protection/permutation"),
        threshold_seed=_hkdf_sha256(master_secret, salt, b"template_protection/threshold"),
        key_version=key_version,
    )
