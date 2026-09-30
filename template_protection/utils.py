"""Small, dependency-free helpers shared across template_protection/.

Deliberately does not import `models/` or `embeddings/` - this package only
ever operates on plain numpy arrays (embeddings, bit vectors), so it stays
usable and testable independent of the recognition models it protects the
output of.
"""

from __future__ import annotations

import hmac

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Scale `vector` to unit L2 norm; returns the input unchanged if it's all-zero.

    `models/common/base_embedder.py::BaseEmbedder.extract_embedding` already
    L2-normalizes every embedding, but this transform re-normalizes
    defensively (e.g. after `models/common/base_embedder.py`'s mock-mode path,
    or if a caller ever hands this module a non-normalized vector) rather than
    trusting an upstream invariant it can't independently verify.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def seed_to_uint64(seed: bytes) -> int:
    """Fold a variable-length seed (as produced by HKDF) into a 64-bit unsigned int.

    `numpy.random.default_rng` accepts an integer seed; using the first 8
    bytes of an HKDF-derived seed keeps the mapping deterministic across
    processes and machines (PCG64's seeding algorithm is stable and does not
    depend on system entropy), which is required for "same key -> same
    template" to hold when enrollment and authentication run in different
    processes.
    """
    return int.from_bytes(seed[:8], byteorder="big", signed=False)


def pack_bits(bits: np.ndarray) -> bytes:
    """Pack a 0/1 array into bytes for compact storage (e.g. a DB BLOB column)."""
    return np.packbits(bits.astype(np.uint8)).tobytes()


def unpack_bits(packed: bytes, num_bits: int) -> np.ndarray:
    """Inverse of `pack_bits`; `num_bits` must be the original (pre-padding) length."""
    return np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:num_bits]


def constant_time_equals(a: bytes, b: bytes) -> bool:
    """Constant-time byte-string equality, for the exact-match fast path in `matcher.py`.

    Wraps `hmac.compare_digest`, which is implemented to take time
    independent of *where* `a` and `b` first differ (though not independent
    of their length - see `matcher.py`'s docstring for why this doesn't
    extend to full constant-time Hamming-distance computation).
    """
    return hmac.compare_digest(a, b)
