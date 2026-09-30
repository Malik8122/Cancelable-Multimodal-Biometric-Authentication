"""Keyed, revocable transform for temporal feature sequences (the hand-gesture template).

The face/voice templates are 256-bit BioHashes. A gesture is compared with DTW over frame sequences, which a single
bit string cannot support, so the hand template is protected differently:

    stored = features_weighted @ Q_k.T         Q_k: (D, D) orthonormal, seeded by HKDF(MASTER_SECRET, user, "hand",
                                                    application, key_version).projection_seed

- Revocable / renewable: a new key version gives a different Q, so the stored sequences change completely; the
  template sets T1..T4 each use their own key version exactly like face and voice.
- Accuracy-neutral: Q is orthonormal and DTW's local cost is Euclidean, so every DTW distance - and every decision -
  is identical to comparing the untransformed sequences (template_protection/dtw.py).
- Honest limitation: unlike BioHash (which quantizes and discards information), this transform is INVERTIBLE by
  anyone who holds the key: Q_k.T @ stored recovers the feature sequence. Its protection rests on MASTER_SECRET
  staying secret. docs/HAND_GESTURE_MODALITY.md discusses this.

Serialization: a small binary container (magic, version, attempts, length, dim, float32 payload); no pickle.
"""

from __future__ import annotations

import struct

import numpy as np

from template_protection.hkdf_keys import KeyMaterial
from template_protection.transform import build_orthonormal_projection

_MAGIC = b"HGT1"
_HEADER = struct.Struct("<4sHHHH")  # magic, format version, attempts, length, dim


def key_matrix(key: KeyMaterial, dim: int) -> np.ndarray:
    """The (dim, dim) orthonormal matrix for `key` (square, so fully orthonormal - see transform.py)."""
    return build_orthonormal_projection(key.projection_seed, dim, dim)


def protect(sequences: list[np.ndarray], key: KeyMaterial) -> list[np.ndarray]:
    """Apply the keyed orthonormal transform to every frame of every sequence."""
    if not sequences:
        raise ValueError("at least one sequence is required")
    dim = sequences[0].shape[1]
    q = key_matrix(key, dim)
    return [(np.asarray(s, dtype=np.float64) @ q.T).astype(np.float32) for s in sequences]


def unprotect(sequences: list[np.ndarray], key: KeyMaterial) -> list[np.ndarray]:
    """Inverse of `protect` (needs the same key). Used only to re-key a template into a new set."""
    dim = sequences[0].shape[1]
    q = key_matrix(key, dim)
    return [(np.asarray(s, dtype=np.float64) @ q).astype(np.float32) for s in sequences]


def serialize(sequences: list[np.ndarray], format_version: int) -> bytes:
    """Pack equally shaped (length, dim) float32 sequences into bytes."""
    stacked = np.stack([np.asarray(s, dtype="<f4") for s in sequences])
    attempts, length, dim = stacked.shape
    return _HEADER.pack(_MAGIC, format_version, attempts, length, dim) + stacked.tobytes()


def deserialize(blob: bytes) -> tuple[int, list[np.ndarray]]:
    """Inverse of `serialize`: (format_version, sequences). Raises ValueError on a malformed blob."""
    blob = bytes(blob)
    if len(blob) < _HEADER.size:
        raise ValueError("hand template too short")
    magic, version, attempts, length, dim = _HEADER.unpack_from(blob)
    if magic != _MAGIC:
        raise ValueError("not a hand-gesture template")
    expected = _HEADER.size + attempts * length * dim * 4
    if len(blob) != expected or attempts == 0:
        raise ValueError("hand template size does not match its header")
    data = np.frombuffer(blob, dtype="<f4", offset=_HEADER.size).reshape(attempts, length, dim)
    return version, [data[i].astype(np.float32) for i in range(attempts)]
