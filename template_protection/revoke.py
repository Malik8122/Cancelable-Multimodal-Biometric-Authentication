"""Template revocation: rotate a user's key and regenerate their template.

Revocation never touches a stored template directly (there is nothing to
"decrypt" or "update in place" - a protected template is not invertible from
the outside). Instead, the caller supplies the same biometric embedding again
(a fresh capture, or one re-derived from a still-enrolled sample) and this
module bumps `key_version`, re-derives key material, and regenerates the
template from scratch:

    Template v1 (key_version=1)
        |  revoke_template(..., new_key_version=2)
        v
    Template v2 (key_version=2)   <- unlinkable to v1 (see biohash.py's
                                      docstring: changing key_version alone
                                      changes every derived seed)

`backend/services/*.py` is responsible for persisting the new template
(inserting a new `ProtectedTemplate` row / bumping `key_version` in
`backend/database/crud.py`) and for making sure old templates for a
revoked key_version are no longer accepted during authentication - this
module only computes the new template, it does not know about the database.
"""

from __future__ import annotations

import numpy as np

from template_protection.biohash import DEFAULT_OUTPUT_BITS, generate_template
from template_protection.hkdf_keys import derive_key


def revoke_template(
    embedding: np.ndarray,
    master_secret: bytes | str,
    *,
    application_id: str,
    user_id: str,
    modality: str,
    new_key_version: int,
    output_bits: int = DEFAULT_OUTPUT_BITS,
) -> np.ndarray:
    """Rotate to `new_key_version` and generate the resulting protected template.

    `new_key_version` must be supplied by the caller (typically
    `current_key_version + 1`, tracked in
    `backend/database/models.py::ProtectedTemplate.key_version`) rather than
    incremented here, since only the caller (the backend, backed by the
    database) knows what the current version actually is.
    """
    rotated_key = derive_key(
        master_secret,
        application_id=application_id,
        user_id=user_id,
        modality=modality,
        key_version=new_key_version,
    )
    return generate_template(embedding, rotated_key, output_bits=output_bits)
