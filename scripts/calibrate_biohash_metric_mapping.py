"""Offline calibration: embedding cosine similarity <-> BioHash template Hamming similarity.

Writes `evaluation/results/biohash_metric_calibration.json`. At runtime
`template_protection/metric_estimation.py` uses only the fitted coefficients
(plus the per-point estimate spread) from it. No embedding, template or key is kept.

Why synthetic pairs are a valid calibration (and what they are not)
--------------------------------------------------------------------
The project stores no face/voice embeddings (the cancelable template is the
privacy mechanism), and no paired raw-embedding data exists in this repo, so
this curve is NOT measured on real biometric data. It does not need to be,
because the relationship is a property of the transform, not of the data:

- `transform.build_orthonormal_projection` draws a Haar-random orthonormal
  projection per key, so the joint distribution of the two projected vectors
  depends on the two embeddings only through their inner product (cosine) -
  not on where in embedding space they sit.
- `transform.quantize`'s jitter scales with `std(projected)`, which has the
  same rotation-invariant distribution, and the permutation does not change
  how many bits agree.

Hence two real embeddings with cosine c and two synthetic unit vectors with
cosine c produce the same distribution of Hamming similarities for the same
embedding dimension and template length. For the same reason one key can be
reused for many independently-oriented pairs (a fixed rotation applied to a
uniformly random direction is still uniformly random), which keeps this run
to a few minutes. The fast path below reproduces `generate_template` stage by
stage; `_check_fast_path` asserts it is bit-for-bit identical before sampling.

Model
-----
With theta = arccos(cosine), the fraction of disagreeing bits is fitted as

    1 - hamming_similarity = (p1 * theta + p2 * theta**2 + p3 * theta**3) / pi

which is exactly 0 at theta = 0 (identical embeddings give identical
templates) and reduces to the classical random-hyperplane result theta / pi
when p1 = 1, p2 = p3 = 0. The key-derived quantization jitter makes the real
curve differ from theta / pi, which is why it is fitted rather than assumed.

What is still an assumption: real embeddings must be L2-normalized (they are:
`models/common/base_embedder.py`), and the curve says nothing about which
cosine a genuine user or an impostor actually produces - that needs real
genuine/impostor data (future work).

Run from the repo root:  python -m scripts.calibrate_biohash_metric_mapping
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from template_protection.biohash import TEMPLATE_FORMAT_VERSION, generate_template
from template_protection.hkdf_keys import KeyMaterial, derive_key
from template_protection.metric_estimation import CALIBRATION_PATH, fitted_hamming_similarity
from template_protection.transform import apply_permutation, build_orthonormal_projection, project, quantize

#: Embedding dimensions - must equal the *_EMBEDDING_DIM constants (models/face/inference.py, models/voice/inference.py).
MODALITY_DIMS = {"face": 512, "voice": 192}

#: A fixed, obviously-synthetic secret: calibration keys only need to be independent draws of the key-derived
#: transform, never real user keys.
_CALIBRATION_SECRET = "biohash-metric-calibration-synthetic-secret"


def _template(unit: np.ndarray, projection: np.ndarray, key: KeyMaterial) -> np.ndarray:
    """`biohash.generate_template` with a precomputed projection matrix (see `_check_fast_path`)."""
    return apply_permutation(quantize(project(unit, projection), key.threshold_seed), key.permutation_seed).astype(np.uint8)


def _check_fast_path(dim: int, bits: int) -> None:
    rng = np.random.default_rng(123)
    for trial in range(5):
        key = derive_key(_CALIBRATION_SECRET, application_id="check", user_id=str(trial), modality="check", key_version=1)
        projection = build_orthonormal_projection(key.projection_seed, bits, dim)
        v = rng.standard_normal(dim)
        v /= np.linalg.norm(v)
        if not np.array_equal(_template(v, projection, key), generate_template(v, key, bits)):
            raise RuntimeError("The calibration fast path no longer matches template_protection.biohash.generate_template.")


def _pair_with_cosine(rng: np.random.Generator, dim: int, cosine: float) -> tuple[np.ndarray, np.ndarray]:
    a = rng.standard_normal(dim)
    a /= np.linalg.norm(a)
    r = rng.standard_normal(dim)
    r -= (r @ a) * a
    r /= np.linalg.norm(r)
    return a, cosine * a + np.sqrt(max(0.0, 1.0 - cosine * cosine)) * r


def _fit(grid: np.ndarray, mean: np.ndarray) -> list[float]:
    theta = np.arccos(np.clip(grid, -1.0, 1.0))
    design = np.stack([theta, theta**2, theta**3], axis=1) / np.pi
    coefficients, *_ = np.linalg.lstsq(design, 1.0 - mean, rcond=None)
    return [float(c) for c in coefficients]


def calibrate_modality(modality: str, dim: int, bits: int, grid: np.ndarray, keys: int, pairs: int, seed: int) -> dict:
    _check_fast_path(dim, bits)
    rng = np.random.default_rng(seed)
    samples = np.empty((len(grid), keys * pairs))
    for k in range(keys):
        key = derive_key(_CALIBRATION_SECRET, application_id="calibration", user_id=f"{modality}-{k}", modality=modality, key_version=1)
        projection = build_orthonormal_projection(key.projection_seed, bits, dim)
        for i, cosine in enumerate(grid):
            for p in range(pairs):
                a, b = _pair_with_cosine(rng, dim, float(cosine))
                samples[i, k * pairs + p] = float(np.mean(_template(a, projection, key) == _template(b, projection, key)))

    mean = samples.mean(axis=1)
    coefficients = _fit(grid, mean)
    fitted = fitted_hamming_similarity(grid, coefficients)
    if np.any(np.diff(fitted) <= 0):
        raise RuntimeError(f"{modality}: the fitted curve is not strictly increasing over the grid.")
    # Spread of a single runtime estimate: invert every sample through the fitted curve.
    dense = np.linspace(grid[0], 1.0, 4001)
    estimates = np.interp(samples, fitted_hamming_similarity(dense, coefficients), dense)
    return {
        "embedding_dim": dim,
        "coefficients": [round(c, 6) for c in coefficients],
        "max_fit_residual": round(float(np.max(np.abs(fitted - mean))), 5),
        # SYNTHETIC CALIBRATION fit quality of the curve against the per-grid-point mean Hamming similarity
        # (not biometric accuracy; see scripts/validate_calibration_real.py for real-embedding validation).
        "fit_statistics": {
            "evidence_label": "SYNTHETIC CALIBRATION",
            "R2": round(float(1 - np.sum((fitted - mean) ** 2) / np.sum((mean - mean.mean()) ** 2)), 8),
            "RMSE": round(float(np.sqrt(np.mean((fitted - mean) ** 2))), 6),
            "MAE": round(float(np.mean(np.abs(fitted - mean))), 6),
            "single_estimate_cosine_sd_mean": round(float(estimates.std(axis=1).mean()), 5),
            "single_estimate_cosine_sd_at_0.80": round(float(estimates.std(axis=1)[int(np.argmin(np.abs(grid - 0.8)))]), 5),
        },
        "fit_residuals": [round(float(r), 6) for r in (mean - fitted)],
        "cosine": [round(float(c), 4) for c in grid],
        "hamming_similarity_mean": [round(float(m), 5) for m in mean],
        "hamming_similarity_std": [round(float(s), 5) for s in samples.std(axis=1)],
        "cosine_estimate_std": [round(float(s), 5) for s in estimates.std(axis=1)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bits", type=int, default=256)
    parser.add_argument("--keys", type=int, default=50)
    parser.add_argument("--pairs", type=int, default=8, help="pairs per key per grid cosine")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=CALIBRATION_PATH)
    args = parser.parse_args()

    grid = np.round(np.arange(-0.2, 1.0 + 1e-9, 0.025), 4)
    report = {
        "description": "Embedding cosine similarity <-> BioHash Hamming similarity, per modality.",
        "source": (
            "Synthetic unit-vector pairs at known cosines through the real BioHash transform (verified bit-identical to "
            "template_protection.biohash.generate_template). Not measured on real biometric data; valid because the keyed "
            "transform is rotation-invariant (see scripts/calibrate_biohash_metric_mapping.py)."
        ),
        "model": "1 - hamming_similarity = (p1*theta + p2*theta^2 + p3*theta^3) / pi, theta = arccos(cosine)",
        "template_bits": args.bits,
        "template_format_version": TEMPLATE_FORMAT_VERSION,
        "keys": args.keys,
        "pairs_per_key_per_point": args.pairs,
        "seed": args.seed,
        "modalities": {
            modality: calibrate_modality(modality, dim, args.bits, grid, args.keys, args.pairs, args.seed + offset)
            for offset, (modality, dim) in enumerate(MODALITY_DIMS.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    for modality, entry in report["modalities"].items():
        print(f"  {modality}: coefficients={entry['coefficients']} max_fit_residual={entry['max_fit_residual']}")
        for c in (0.9, 0.8, 0.725):
            i = entry["cosine"].index(c)
            print(f"    cos={c:.3f}  hamming={entry['hamming_similarity_mean'][i]:.4f}  estimate_std={entry['cosine_estimate_std'][i]:.3f}")


if __name__ == "__main__":
    main()
