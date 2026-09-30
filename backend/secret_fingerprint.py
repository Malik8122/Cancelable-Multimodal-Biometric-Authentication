"""One-way fingerprint of MASTER_SECRET, used only to detect whether the currently configured
secret matches the one that protected any existing biometric templates - never to derive any
template-protection key material.

Deliberately kept separate from `template_protection/hkdf_keys.py::derive_key`: that function
derives per-(user, modality, application, key_version) template keys from MASTER_SECRET, and this
module must never be confused with, share code with, or be reachable from that path. Different
module, different fixed public HKDF context, different purpose - one answers "what key protects
this user's template", the other only ever answers "is this the same secret as before?".
"""

from __future__ import annotations

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

#: Fixed, public HKDF "info" context. Not a secret itself - it exists purely to make this
#: fingerprint's derivation namespace-distinct from anything else ever derived from
#: MASTER_SECRET (in particular, `template_protection/hkdf_keys.py::derive_key`'s per-template
#: seeds, which use an entirely different salt/info construction). "v1" so a future change to
#: this derivation (e.g. a different length) is a distinguishable new context, not a silent
#: reinterpretation of old fingerprint rows.
FINGERPRINT_CONTEXT = b"master_secret_fingerprint/v1"

FINGERPRINT_LENGTH_BYTES = 32


def fingerprint_master_secret(master_secret: str | bytes) -> bytes:
    """A deterministic, one-way 32-byte fingerprint of `master_secret`.

    Same secret -> same fingerprint, every time, on any machine (HKDF-SHA256 is a fixed, portable
    construction - the same property `template_protection/hkdf_keys.py` already relies on for
    "same key -> same template"). A different secret produces a statistically unrelated
    fingerprint. This is NOT reversible back to the secret (HKDF is a one-way expansion), so
    persisting this value never exposes MASTER_SECRET - it only ever lets a later process ask "is
    this the same secret as before?", never "what is the secret?".
    """
    if isinstance(master_secret, str):
        master_secret = master_secret.encode("utf-8")
    hkdf = HKDF(algorithm=hashes.SHA256(), length=FINGERPRINT_LENGTH_BYTES, salt=None, info=FINGERPRINT_CONTEXT)
    return hkdf.derive(master_secret)
