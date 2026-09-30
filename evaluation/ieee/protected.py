"""Batched BioHash (bit-identical to the shipped transform) and the verification comparison protocol.

`templates(X, key)` reproduces `template_protection.biohash.generate_template` for many embeddings under one key:
the projection matrix is built once per key instead of once per embedding. `assert_matches_production()` checks it
bit-for-bit against `generate_template` and runs on import, so a drift in the production transform stops every
experiment rather than silently changing results.

Protocol (mirrors deployment): one enrolled reference per identity; every other sample of that identity is a genuine
probe compared under the identity's OWN key; impostor probes (other identities) are compared under the TARGET
identity's key, exactly as `ModalityService.authenticate` derives the key from the claimed user_id.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation.ieee.common import EVAL_SECRET, SEED
from template_protection.biohash import generate_template
from template_protection.hkdf_keys import KeyMaterial, derive_key
from template_protection.metric_estimation import cosine_from_euclidean, euclidean_from_cosine, get_curve
from template_protection.transform import build_orthonormal_projection
from template_protection.utils import seed_to_uint64

BITS = 256
APP = "ieee-eval"


def key_for(identity: str, modality: str, key_version: int = 1, application_id: str = APP) -> KeyMaterial:
    return derive_key(EVAL_SECRET, application_id=application_id, user_id=str(identity), modality=modality, key_version=key_version)


def templates(X: np.ndarray, key: KeyMaterial, bits: int = BITS) -> np.ndarray:
    """(n, D) embeddings -> (n, bits) uint8 templates under `key` (same math as generate_template, batched)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms == 0, 1, norms)
    W = build_orthonormal_projection(key.projection_seed, bits, X.shape[1])
    P = X @ W.T  # (n, bits)
    z = np.random.default_rng(seed_to_uint64(key.threshold_seed)).normal(loc=0.0, scale=1.0, size=bits)
    spread = P.std(axis=1, keepdims=True)
    spread = np.where(spread == 0, 1.0, spread)
    Q = (P > (0.5 * spread) * z).astype(np.uint8)
    perm = np.random.default_rng(seed_to_uint64(key.permutation_seed)).permutation(bits)
    return Q[:, perm]


def assert_matches_production(dims=(512, 192), trials: int = 6) -> None:
    rng = np.random.default_rng(7)
    for dim in dims:
        for t in range(trials):
            key = key_for(f"check-{t}", "check")
            X = rng.standard_normal((3, dim))
            batch = templates(X, key)
            for row, x in zip(batch, X):
                if not np.array_equal(row, generate_template(x, key, output_bits=BITS)):
                    raise RuntimeError("evaluation BioHash no longer matches template_protection.biohash.generate_template")


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise fraction of agreeing bits (the production `matcher.compare(..., 'hamming')`)."""
    return (np.asarray(a) == np.asarray(b)).mean(axis=-1)


# ----------------------------------------------------------------------------- estimates (vectorized modality_metrics)


def estimated_cosine(h: np.ndarray, modality: str) -> np.ndarray:
    curve = get_curve(modality, BITS)
    h = np.asarray(h, float)
    return np.where(h >= 1.0, 1.0, np.interp(h, curve.hamming_similarity_mean, curve.cosine))


def estimated_distance(h: np.ndarray, modality: str) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * estimated_cosine(h, modality)))


def assert_estimates_match_backend() -> None:
    """The vectorized estimates must equal backend/services/modality_metrics.decide for the same inputs."""
    from backend.config import Settings
    from backend.services.modality_metrics import decide

    s = Settings(master_secret="x")
    for h in (0.5, 0.62, 0.7813, 0.8176, 0.85, 0.93, 1.0):
        f = decide("face", h, BITS, s)
        v = decide("voice", h, BITS, s)
        assert abs(f.value - float(estimated_cosine(h, "face"))) < 1e-12
        assert abs(v.value - float(estimated_distance(h, "voice"))) < 1e-12
    assert abs(cosine_from_euclidean(euclidean_from_cosine(0.8)) - 0.8) < 1e-12


assert_matches_production()


# ----------------------------------------------------------------------------- protocol


@dataclass
class Comparisons:
    """Parallel arrays: one row per comparison."""

    modality: str
    target: np.ndarray  # identity of the enrolled reference (whose key is used)
    probe_identity: np.ndarray
    genuine: np.ndarray  # bool
    ref_index: np.ndarray  # into the embedding array
    probe_index: np.ndarray
    raw_cosine: np.ndarray
    hamming: np.ndarray  # same-key protected-template Hamming similarity

    def split(self, field: str = "hamming"):
        v = getattr(self, field)
        return v[self.genuine], v[~self.genuine]


def build_comparisons(
    modality: str,
    E: np.ndarray,
    labels: np.ndarray,
    impostors_per_identity: int | None = 50,
    references: dict | None = None,
    probe_mask: np.ndarray | None = None,
    seed: int = SEED,
) -> Comparisons:
    """Enrollment-vs-probe comparisons on real embeddings.

    - reference: first sample (sorted order) of each identity with >= 2 usable samples, unless `references` maps
      identity -> index;
    - genuine probes: that identity's other samples (restricted to `probe_mask` if given);
    - impostor probes: `impostors_per_identity` random probes of other identities (None = all of them), each
      templated under the TARGET identity's key.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    probe_mask = np.ones(len(labels), bool) if probe_mask is None else probe_mask
    identities = sorted(set(labels.tolist()))
    refs = {}
    for ident in identities:
        idx = np.nonzero(labels == ident)[0]
        if references is not None:
            if ident in references:
                refs[ident] = references[ident]
        elif len(idx) >= 2:
            refs[ident] = int(idx[0])
    probe_all = np.nonzero(probe_mask)[0]
    rows = {k: [] for k in ("target", "probe_identity", "genuine", "ref_index", "probe_index", "raw_cosine", "hamming")}
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    for ident, r in refs.items():
        gen = [int(p) for p in np.nonzero((labels == ident) & probe_mask)[0] if p != r]
        others = probe_all[labels[probe_all] != ident]
        if impostors_per_identity is not None and len(others) > impostors_per_identity:
            others = rng.choice(others, impostors_per_identity, replace=False)
        probes = np.array(gen + [int(o) for o in others], dtype=int)
        if len(probes) == 0:
            continue
        key = key_for(ident, modality)
        T = templates(np.vstack([E[r][None, :], E[probes]]), key)
        h = hamming_similarity(T[1:], T[0][None, :])
        c = En[probes] @ En[r]
        rows["target"] += [ident] * len(probes)
        rows["probe_identity"] += labels[probes].tolist()
        rows["genuine"] += [True] * len(gen) + [False] * (len(probes) - len(gen))
        rows["ref_index"] += [r] * len(probes)
        rows["probe_index"] += probes.tolist()
        rows["raw_cosine"] += c.tolist()
        rows["hamming"] += h.tolist()
    return Comparisons(modality=modality, **{k: np.asarray(v) for k, v in rows.items()})
