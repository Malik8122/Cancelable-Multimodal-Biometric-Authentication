"""The BioHashing-style cancelable transform's math, stage by stage.

Every function here is a pure numpy computation: no key derivation (that's
`hkdf_keys.py`), no orchestration (that's `biohash.py`), no I/O. Each stage
takes an already-derived seed (raw bytes) and is deterministic given that
seed, which is the property the whole cancelable-template scheme depends on:
the same seed always reproduces the same projection matrix / permutation /
thresholds, on any machine or process, because `numpy.random.default_rng`
(PCG64) is a fixed, portable algorithm that does not consult system entropy.

Pipeline (see `biohash.py::generate_template` for the orchestration):

    embedding (unit vector, dim D)
        -> build_orthonormal_projection(seed, bits, D)   -> matrix (bits, D)
        -> project(embedding, matrix)                    -> real vector (bits,)
        -> quantize(projected, seed)                      -> bit vector (bits,)
        -> apply_permutation(bits, seed)                  -> bit vector (bits,)  [final template]
"""

from __future__ import annotations

import numpy as np

from template_protection.utils import seed_to_uint64


def build_orthonormal_projection(seed: bytes, output_bits: int, embedding_dim: int) -> np.ndarray:
    """Build a deterministic, seed-derived (output_bits, embedding_dim) projection matrix.

    Each row is a unit-norm direction in embedding space; rows within the
    same "block" of at most `embedding_dim` rows are mutually orthonormal,
    built via QR decomposition of a Gaussian random matrix (a standard way to
    sample a uniformly random orthonormal basis, "Haar-distributed").

    Honest limitation: true orthonormality across *all* `output_bits` rows is
    only mathematically possible when `output_bits <= embedding_dim` (rank of
    a real matrix is bounded by its smaller dimension). When `output_bits`
    exceeds `embedding_dim` - e.g. a 256-bit template requested from a
    192-dim voice embedding - this function composes multiple
    independent orthonormal blocks: rows are orthonormal *within* a block but
    not orthogonal *across* blocks. This is deliberately not hidden; see
    docs/TEMPLATE_PROTECTION.md for what it does and does not imply for
    template security.
    """
    if output_bits <= 0:
        raise ValueError(f"output_bits must be positive, got {output_bits}")
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

    seeding_rng = np.random.default_rng(seed_to_uint64(seed))
    blocks: list[np.ndarray] = []
    remaining = output_bits
    while remaining > 0:
        block_size = min(remaining, embedding_dim)
        # Each block gets its own sub-seed, drawn from the same deterministic
        # stream, so multi-block matrices are still fully reproducible from
        # one input seed.
        block_seed = int(seeding_rng.integers(0, 2**63 - 1))
        block_rng = np.random.default_rng(block_seed)
        gaussian = block_rng.standard_normal((embedding_dim, block_size))
        q, _ = np.linalg.qr(gaussian)  # q: (embedding_dim, block_size), orthonormal columns
        blocks.append(q.T)  # (block_size, embedding_dim): orthonormal rows within this block
        remaining -= block_size

    return np.concatenate(blocks, axis=0)  # (output_bits, embedding_dim)


def project(embedding: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    """Apply the projection matrix: (bits, dim) @ (dim,) -> (bits,).

    Because `projection_matrix` has unit-norm rows and `embedding` is a unit
    vector (see `utils.l2_normalize`), each resulting component is a
    dot product of two unit vectors and therefore bounded in [-1, 1]
    (Cauchy-Schwarz) - `quantize` relies on this bound.
    """
    return projection_matrix @ embedding


def quantize(projected: np.ndarray, seed: bytes) -> np.ndarray:
    """Binarize `projected` using per-position, key-derived random thresholds.

    Thresholds are centered at zero (so quantization is primarily a sign
    split of `projected`, which is what carries the actual embedding signal
    and gives two different embeddings under the same key discriminably
    different bits) with small random jitter drawn from
    `Normal(0, 0.5 * std(projected))` per position, seeded by
    `threshold_seed`.

    The jitter's scale is tied to `projected`'s own standard deviation rather
    than a fixed constant, because `project`'s output magnitude depends on
    `embedding_dim` and `output_bits` (each component is a dot product of two
    unit vectors, so its typical size shrinks as ~1/sqrt(embedding_dim)) - a
    fixed-scale threshold (e.g. Uniform(-1, 1)) would dwarf that signal in
    high dimensions and make quantization ignore the embedding almost
    entirely, which would silently break both recognition (same embedding
    should reproduce its template) and discrimination (different embeddings
    should differ). Scaling relative to the data keeps the threshold's
    influence calibrated regardless of dimensionality, while still being
    keyed: a different `threshold_seed` shifts *which* positions land on
    which side of the (embedding-driven) sign split.
    """
    rng = np.random.default_rng(seed_to_uint64(seed))
    spread = float(np.std(projected)) or 1.0
    thresholds = rng.normal(loc=0.0, scale=0.5 * spread, size=projected.shape)
    return (projected > thresholds).astype(np.uint8)


def apply_permutation(bits: np.ndarray, seed: bytes) -> np.ndarray:
    """Permute `bits` with a key-derived, deterministic random permutation.

    This is the "user-specific permutation" stage of classical BioHashing: it
    doesn't add entropy on its own (a permutation of a bit vector is exactly
    as guessable bit-for-bit as the original), but it does mean two templates
    derived from related embeddings under *different* keys no longer share
    positional structure, which is what the revocability/diversity
    experiments in evaluation/privacy_metrics.py measure.
    """
    rng = np.random.default_rng(seed_to_uint64(seed))
    permutation = rng.permutation(len(bits))
    return bits[permutation]
