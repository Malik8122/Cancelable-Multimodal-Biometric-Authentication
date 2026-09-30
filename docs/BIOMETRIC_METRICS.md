# Biometric Similarity Metrics

How each stage of authentication measures similarity, which direction is "better", and what the face and voice
numbers actually are.

## Summary

| Component | Metric | Threshold / decision | Direction |
|---|---|---|---|
| Face | Cosine similarity (**calibrated estimate**) | estimate **≥ 0.80** | higher is better |
| Voice | Euclidean distance (**calibrated estimate**) | estimate **≤ 0.75** | **lower** is better |
| Fusion | Mean of the per-modality scores on the estimated-cosine scale | per `fusion_policy` (default `ALL_REQUIRED`: every presented modality must pass) | higher is better |
| Cancelable template | Hamming distance between 256-bit BioHash templates | the comparison every decision above is derived from | lower distance (higher similarity) is better |

```text
Face:                        cosine similarity, estimated          (higher = better)
Voice:                       Euclidean distance, estimated         (LOWER = better)
Fusion:                      modality score fusion, one scale      (higher = better)
Cancelable transformation:   BioHash (HKDF-keyed projection, quantization, permutation), 256 bits
Stored-template comparison:  Hamming distance
```

## Why face and voice are *estimates*

The system never stores a face or voice embedding. What it stores, and all it can ever compare against, is the 256-bit
cancelable template (`template_protection/biohash.py`). So at verification time there is no enrolled embedding to
compute a true cosine similarity or Euclidean distance with, and neither can be recovered exactly from two templates.
Storing embeddings to get exact values would undo the privacy guarantee this project is built on, so it is not done.

What *can* be done is to estimate them. The BioHash transform approximately preserves the angle between two
embeddings: the closer two embeddings are, the more of their template bits agree. That relationship is calibrated
offline and used at runtime to report, from the observed Hamming similarity, the embedding cosine that on average
produces it.

These numbers are therefore always labelled **estimated** cosine similarity / **estimated** Euclidean distance, and are
never described as exact.

### The calibration

`scripts/calibrate_biohash_metric_mapping.py` (run: `python -m scripts.calibrate_biohash_metric_mapping`) writes
`evaluation/results/biohash_metric_calibration.json`.

- **Data:** synthetic pairs of unit vectors at known cosines (−0.2 to 1.0), sent through the real BioHash transform. The
  script checks that its fast path is bit-for-bit identical to `generate_template`. **No real biometric data was used,
  and none exists in this repository.** (The datasets are downloaded inside the Kaggle training kernels, and no
  paired embeddings were kept.)
- **Why synthetic pairs are valid:** each key's projection is a Haar-random orthonormal matrix, and the quantization
  jitter scales with the spread of the projected values, so the transform treats every direction in embedding space the
  same way. The Hamming similarity of two templates therefore depends only on the cosine between the two embeddings,
  the embedding size and the template length, not on the data. A real face pair at cosine 0.8 produces the same
  distribution of Hamming similarities as a synthetic pair at cosine 0.8.
- **Model**, with θ = arccos(cosine):

  ```text
  1 − hamming_similarity = (p1·θ + p2·θ² + p3·θ³) / π
  ```

  This is exactly 0 at θ = 0 (identical embeddings give identical templates). With p1 = 1 and p2 = p3 = 0 it becomes
  the classical random-hyperplane result θ/π. The key-derived jitter makes the real curve differ from θ/π, so the
  coefficients are fitted (p1 ≈ 0.88 for both modalities). The maximum fit residual is about 0.004 in Hamming
  similarity.
- **Runtime:** only the three coefficients per modality (plus the per-point spread) are used
  (`template_protection/metric_estimation.py`). The curve is inverted to turn an observed Hamming similarity into an
  estimated cosine.
- **Uncertainty:** a single estimate near the thresholds has a standard deviation of about **±0.05 in cosine** (it
  grows toward low, impostor-level cosines). It is reported next to every estimate as `metric_uncertainty`.

### Face and voice from the same estimate

Every embedder returns L2-normalized vectors (`models/common/base_embedder.py`). For unit vectors, cosine similarity
and Euclidean distance describe the same angle:

```text
d = sqrt(2 − 2·cos)        cos = 1 − d² / 2
```

- **Face:** estimated cosine ĉ; match if ĉ ≥ `FACE_COSINE_THRESHOLD` (0.80).
- **Voice:** estimated distance d̂ = sqrt(2 − 2ĉ); match if d̂ ≤ `VOICE_EUCLIDEAN_THRESHOLD` (0.75). This is a distance:
  0 means identical, 2 means opposite, and **smaller is better**. It is never compared with `>=`.

Both estimates increase or decrease steadily with the Hamming similarity, so each rule is exactly a threshold on the
template comparison:

| Rule | Equivalent template Hamming similarity (256 bits) | Previous rule |
|---|---|---|
| Face estimated cosine ≥ 0.80 | ≥ ≈ 0.818 | ≥ 0.80 (operator-specified) — the new rule is **stricter** |
| Voice estimated distance ≤ 0.75 (cosine ≥ 0.719) | ≥ ≈ 0.782 | ≥ 0.80 (operator-specified) — the new rule is **looser** |

The Hamming comparison of the stored template is still what decides. The teacher's thresholds are applied in their own
units and translated into template space through the calibration curve.

## Fusion: one direction, one scale

Averaging a similarity with a distance is meaningless (`(face_similarity + voice_distance) / 2` rewards a *worse* voice
match). So every modality enters fusion as a higher-is-better score on the **estimated-cosine scale**:

| Modality | Fusion score | Fusion-scale threshold |
|---|---|---|
| Face | ĉ | 0.80 |
| Voice | 1 − d̂² / 2 (= ĉ) | 1 − 0.75² / 2 = 0.71875 |

`fusion_similarity` = the equal-weight mean of those scores. Under the default `ALL_REQUIRED` policy, access is
granted only if every presented modality passes its own rule; the fused score is informational. Under `WEIGHTED`, the
fused score must reach the mean fusion-scale threshold, and any modality below `DEFAULT_WEIGHTED_FLOOR` = 0.0 (chance
on this scale, formerly Hamming 0.5) vetoes access.

## Configured, not calibrated

`FACE_COSINE_THRESHOLD = 0.80` and `VOICE_EUCLIDEAN_THRESHOLD = 0.75` (`backend/config.py`, overridable by environment
variable) are **teacher-requested project thresholds**. The calibration above only connects cosine to Hamming. It says
nothing about which cosines genuine users and impostors actually produce, so **no FAR/FRR/EER is claimed for these
thresholds**. The next experimental step is to collect genuine and impostor attempts from several real users, measure
FAR/FRR at these thresholds, and choose operating points from that data.

They replace `evaluation/results/face_threshold.json` and `voice_threshold.json` (0.80 on Hamming similarity), which
were removed so that each modality has exactly one authoritative threshold.

## What the API exposes

Production responses (`DEBUG_SCORES=false`) carry only the single decision, the fused similarity and, **only when
access is granted**, the user's display name. With `DEBUG_SCORES=true` (local development), `results.<modality>` also
carries `metric`, `metric_value`, `metric_threshold`, `metric_higher_is_better`, `metric_uncertainty`,
`hamming_similarity`, `hamming_distance_bits` and `template_bits`. No response ever contains an embedding, a template
or key material.
