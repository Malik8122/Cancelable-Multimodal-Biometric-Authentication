"""Cancelable biometric template protection.

Public API, wiring the pipeline described in `docs/TEMPLATE_PROTECTION.md`:

    embedding -> derive_key -> generate_template -> (stored) protected template
    embedding -> derive_key -> generate_template -> compare(against stored)  -> accept/reject
    revoke_template(..., new_key_version)                                   -> new, unlinkable template

Nothing in this package ever touches raw images or calls into `models/` or
`embeddings/` directly - it operates purely on the fixed-length, L2-normalized
embeddings that `embeddings.pipelines.ModalityPipeline.embed()` already
produces.
"""

from __future__ import annotations

from template_protection.biohash import TEMPLATE_FORMAT_VERSION, generate_template
from template_protection.hkdf_keys import KeyMaterial, derive_key
from template_protection.matcher import accept, compare
from template_protection.revoke import revoke_template

__all__ = [
    "TEMPLATE_FORMAT_VERSION",
    "KeyMaterial",
    "accept",
    "compare",
    "derive_key",
    "generate_template",
    "revoke_template",
]
