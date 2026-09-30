"""BioHashing-inspired cancelable template generation.

Orchestrates `transform.py`'s math stages into the single call the rest of
the system uses:

    template = generate_template(embedding, key)

Full pipeline, matching docs/TEMPLATE_PROTECTION.md:

1. **Normalize** the embedding to unit L2 norm (defensively re-applied; see
   `utils.l2_normalize`).
2. **Project**: build a `key.projection_seed`-derived orthonormal projection
   matrix (`transform.build_orthonormal_projection`) and apply it
   (`transform.project`), mapping the embedding into `output_bits` real
   values bounded in [-1, 1].
3. **Quantize**: binarize each projected value against a
   `key.threshold_seed`-derived per-position random threshold
   (`transform.quantize`).
4. **Permute**: reorder the resulting bits with a `key.permutation_seed`-derived
   permutation (`transform.apply_permutation`).

The result is a fixed-length `np.uint8` array of 0s/1s - the "protected
template" that `backend/` persists in place of the raw embedding. Producing
the *same* template again requires the *same* `KeyMaterial` (i.e. the same
master secret + application/user/modality/key_version) and the *same*
embedding (i.e. successfully re-authenticating the same biometric); anything
else yields a statistically unrelated bit string.

**Non-invertibility, stated honestly:** this pipeline is a many-to-one,
information-lossy map (a `dim`-dimensional real vector collapses to
`output_bits` bits), which makes reconstructing the original embedding from
the template *alone* infeasible. This is **not** a cryptographic one-way
function with a computational-hardness proof - it is a data-transformation
argument. If an attacker also recovers the key material (and hence the exact
projection matrix, thresholds, and permutation), partial reconstruction of
the embedding becomes a studied attack in the BioHashing literature. See
docs/TEMPLATE_PROTECTION.md for the full assumptions/limitations discussion;
nothing in this codebase claims stronger security than that.
"""

from __future__ import annotations

import numpy as np

from template_protection.hkdf_keys import KeyMaterial
from template_protection.transform import apply_permutation, build_orthonormal_projection, project, quantize
from template_protection.utils import l2_normalize

#: Bumped only if the transform algorithm itself changes (e.g. a different
#: quantization scheme) - stored alongside each protected template
#: (backend/database/models.py::ProtectedTemplate.template_version) so a
#: future migration can tell old-format templates apart from new ones.
#: This is independent of `KeyMaterial.key_version`, which tracks per-user
#: key rotation, not the algorithm version.
TEMPLATE_FORMAT_VERSION = 1

#: Default protected-template length in bits. Chosen so it's always <=
#: every modality's embedding dimension (512 face / 192 voice - see
#: models/*/inference.py), which keeps the projection
#: matrix fully orthonormal (see transform.build_orthonormal_projection's
#: docstring) without the caller needing to know per-modality dimensions.
DEFAULT_OUTPUT_BITS = 128


def generate_template(embedding: np.ndarray, key: KeyMaterial, output_bits: int = DEFAULT_OUTPUT_BITS) -> np.ndarray:
    """Transform a biometric embedding into a protected, cancelable template.

    `output_bits` may be 128, 256, 512, or any other positive integer; values
    larger than `embedding.shape[0]` are supported but only orthonormal
    *within* blocks of size `embedding.shape[0]` (see
    `transform.build_orthonormal_projection`), not across the full output -
    an explicit, documented tradeoff rather than a silent one.
    """
    normalized = l2_normalize(np.asarray(embedding, dtype=np.float64))
    embedding_dim = normalized.shape[0]

    projection_matrix = build_orthonormal_projection(key.projection_seed, output_bits, embedding_dim)
    projected = project(normalized, projection_matrix)
    quantized = quantize(projected, key.threshold_seed)
    template = apply_permutation(quantized, key.permutation_seed)

    return template.astype(np.uint8)
