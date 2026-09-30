"""TEMPORARY, LOCAL-DEVELOPMENT-ONLY face diagnostics (DEBUG_SCORES=true).

Exists to gather real, one-time scalar measurements from a live browser enrollment +
authentication, to help decide whether insufficient geometric face alignment (MTCNN crops the
detected bounding box but performs no landmark-based rotation/affine alignment - see the
alignment investigation) is why genuine face Hamming similarity has measured ~0.70-0.74 against
this project's 0.80 threshold. Delete this module and its two call sites in
`backend/services/base_service.py` once that investigation is resolved - this is not a permanent
feature.

Hard rules, enforced by construction, not by convention:
- Every value computed and logged here is a single Python float (a cosine similarity, or a mean/
  min/max/std of several) - never a vector, never an array, never anything that could be used to
  reconstruct an embedding, image, or template.
- Nothing here is persisted: no database write, no file write, no return value added to any API
  response - `logging.Logger.info` only, exactly like this project's existing `ENROLL-DEBUG` /
  `AUTH-DEBUG` lines (`backend/services/base_service.py`).
- Callers gate every call on `Settings.debug_scores` - this module has no gate of its own, so it
  never accidentally runs when a caller forgets to check, but it also never runs unless a caller
  explicitly asked (see `ModalityService.enroll_poses` / `.authenticate_embedding`).
- The real authentication decision is computed entirely before these functions are ever called
  (`template_protection.matcher.compare`/`accept`, unchanged) - nothing here can influence it;
  these functions return `None` and are called purely for their logging side effect.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("backend.face_debug")


def _pairwise_cosine_stats(vectors: list[np.ndarray]) -> tuple[float, float, float, float]:
    """(mean, std, min, max) cosine similarity over every distinct pair in `vectors`.

    Every embedding this project produces (`models/common/base_embedder.py::BaseEmbedder.extract_embedding`)
    is already L2-normalized, so a plain dot product IS the cosine similarity - no extra
    normalization needed here.
    """
    pairs = [float(np.dot(vectors[i], vectors[j])) for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    if not pairs:
        nan = float("nan")
        return nan, nan, nan, nan
    return float(np.mean(pairs)), float(np.std(pairs)), float(np.min(pairs)), float(np.max(pairs))


def log_enrollment_pose_diagnostics(pose_embeddings: list[np.ndarray], centroid: np.ndarray) -> None:
    """[FACE DEBUG] pose-to-pose and pose-to-centroid cosine similarity for one enrollment.

    Call from `ModalityService.enroll_poses` BEFORE its embeddings list is cleared, only when
    `Settings.debug_scores` is true. Does not change how `centroid` was built - `centroid` is
    passed in only to be measured against, never recomputed here (see
    `embeddings/centroid.py::centroid_embedding`, unchanged).
    """
    pose_mean, pose_std, pose_min, pose_max = _pairwise_cosine_stats(pose_embeddings)
    centroid_similarities = [float(np.dot(e, centroid)) for e in pose_embeddings]

    logger.info(
        "[FACE DEBUG] stage=enrollment poses=%d\n"
        "enrollment_pose_similarity: mean=%.4f std=%.4f min=%.4f max=%.4f (pairwise, %d pairs)\n"
        "enrollment_centroid_similarity: mean=%.4f std=%.4f min=%.4f max=%.4f (per-pose vs final centroid)",
        len(pose_embeddings),
        pose_mean, pose_std, pose_min, pose_max, len(pose_embeddings) * (len(pose_embeddings) - 1) // 2,
        float(np.mean(centroid_similarities)), float(np.std(centroid_similarities)),
        float(np.min(centroid_similarities)), float(np.max(centroid_similarities)),
    )


def log_authentication_diagnostics(protected_hamming_similarity: float) -> None:
    """[FACE DEBUG] authentication-side diagnostics for one attempt.

    `live_vs_centroid_cosine` (the live embedding compared with the ENROLLED centroid, before
    template protection) is reported as unavailable rather than fabricated: the enrolled
    centroid's raw embedding is never stored anywhere in this system (by design - see
    docs/PRIVACY_AND_SECURITY.md and `template_protection/biohash.py`'s non-invertibility
    discussion) - only its protected, one-way-transformed bits are, in a separate HTTP request
    from any given authentication attempt. There is no raw vector left to compare the live
    embedding against at this point, so this function does not - and architecturally cannot -
    compute that number. `protected_hamming_similarity` is the one real, already-computed
    measurement available here: the same value `AuthenticationResult.hamming_similarity` carries, restated in
    the "[FACE DEBUG]" format for a single place to look.
    """
    logger.info(
        "[FACE DEBUG] stage=authentication\n"
        "live_vs_centroid_cosine: unavailable (enrolled raw embedding is never stored - see docstring)\n"
        "protected_hamming_similarity: %.4f",
        protected_hamming_similarity,
    )


def log_capture_quality(stage: str, quality: dict) -> None:
    """[FACE DEBUG] capture-quality scalars of one capture (the enrollment gate's signals and verdict).

    Authentication captures are never quality-gated, so a low genuine score cannot otherwise be told apart from
    a poor capture (dark, small, off-centre, turned, blurred). `quality` comes from
    `FacePipeline.capture_quality`: a verdict string plus single floats - no image, landmark or embedding.
    Called only after the decision is made, and only when `Settings.debug_scores` is true.
    """
    fields = ("detection_confidence", "sharpness", "face_size_ratio", "center_offset", "roll_degrees", "yaw_ratio")
    values = " ".join(f"{k}={quality[k]:.3f}" for k in fields if k in quality)
    logger.info("[FACE DEBUG] stage=%s capture_quality verdict=%s %s", stage, quality.get("verdict"), values)
