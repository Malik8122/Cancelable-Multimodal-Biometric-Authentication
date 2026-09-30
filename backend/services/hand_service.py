"""Dynamic hand gesture modality ("hand"): the Z gesture - three-sample enrollment, one-sample authentication, DTW.

Landmarks come from MediaPipe Hands in the browser; this service receives the landmark sequence only
(`preprocessing/hand_gesture.py::parse_capture`). It plugs into the SAME template-set architecture as face and voice
(`backend/services/base_service.py`): one hand row per template set (T1 ACTIVE, T2-T4 STANDBY), each under its own
HKDF key version, rotated/revoked/activated together with the other modalities.

Template (per set): the three enrollment feature sequences S1..S3, weighted and transformed by the set's keyed
orthonormal matrix (`template_protection/sequence_transform.py`), serialized in one blob. They are kept separate -
never averaged point by point (the samples are not time-aligned, so an average would blur the Z's corners). The
feature sequences themselves are never stored and never leave the backend.

Enrollment robustness: the pairwise distances D(S1,S2), D(S1,S3), D(S2,S3) identify the medoid (most representative
sample) and an outlier - a sample far from BOTH others while those two agree - which is refused as a capture-quality
problem (nothing stored, samples recorded again) instead of being allowed to poison the template.

Decision (lower is better):

    d_i = min over r in R  DTW(rotate(candidate, r), S_i)    i = 1..3; R = small rotation search (+/- 12 deg);
                                                              both sides in the set's keyed space (DTW unchanged by it)
    d   = aggregate(d_1..d_3)                                 Settings.hand_gesture_aggregation
    MATCH  <=>  d <= Settings.hand_gesture_dtw_threshold

Fusion scale (higher is better, like the estimated cosines of face and voice): s = clip(1 - d / (2 T), -1, 1), so the
threshold T maps to s = 0.5 (`HAND_FUSION_THRESHOLD`), d = 0 to 1 and d >= 2T to <= 0. This is a transparent,
uncalibrated mapping - not a probability and not a cosine.
"""

from __future__ import annotations

import itertools
import logging
import time
from functools import lru_cache

import numpy as np
from sqlalchemy.orm import Session

from backend.config import Settings, get_settings
from backend.database import crud
from backend.database.models import ProtectedTemplate
from backend.security_validation import assert_valid, validate_hand_authentication
from backend.services.base_service import AuthenticationResult, _active_row
from preprocessing.hand_gesture import (
    CAPTURE_QUALITY_FAILURE,
    FEATURE_DIM,
    FEATURE_GROUPS,
    FEATURE_WEIGHTS,
    HAND_TEMPLATE_FORMAT,
    HAND_TEMPLATE_VERSION,
    RESAMPLED_LENGTH,
    HandCapture,
    HandCaptureRejected,
    ProcessedGesture,
    process_capture,
    rotate_features,
    weighted,
)
from template_protection.dtw import aggregate, dtw_alignment, dtw_distance
from template_protection.hkdf_keys import derive_key
from template_protection.sequence_transform import deserialize, protect, serialize, unprotect

logger = logging.getLogger("backend.services.hand_service")

MODALITY = "hand"
HAND_FUSION_THRESHOLD = 0.5
METRIC = "dtw_distance"
#: Decision codes reported for the hand modality (in addition to the capture failure codes of preprocessing/hand_gesture.py).
MATCH = "MATCH"
GESTURE_MISMATCH = "GESTURE_MISMATCH"
#: Step of the rotation search (degrees) within +/- Settings.hand_gesture_rotation_tolerance_deg.
HAND_ROTATION_STEP_DEG = 6.0
#: DEVELOPMENT DIAGNOSTICS ONLY (DEBUG_SCORES): alternative feature weightings whose distances are computed next to the
#: production one ("shadow" distances, never used for a decision) - evaluation/hand_representation_experiment.py
#: compares them properly on multi-person data. Only time-resampled variants: they can be computed from the stored
#: (time-resampled) enrolled samples. Group order: trajectory, direction, speed, acceleration, orientation, fingers.
DIAGNOSTIC_WEIGHT_SETS = {
    "A_current": (1.0, 0.6, 0.3, 0.15, 0.3, 0.1),
    "A_no_fingers": (1.0, 0.6, 0.3, 0.15, 0.3, 0.0),
    "A_fingers_0.02": (1.0, 0.6, 0.3, 0.15, 0.3, 0.02),
    "B_no_speed": (1.0, 0.6, 0.0, 0.15, 0.3, 0.1),
    "E1_tempo_robust": (1.0, 0.6, 0.1, 0.05, 0.2, 0.05),
}
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def _weight_vector(group_weights: tuple) -> np.ndarray:
    w = np.zeros(FEATURE_DIM)
    for (group, cols), weight in zip(FEATURE_GROUPS.items(), group_weights):
        w[list(cols)] = weight
    return w


def shadow_diagnostics(live: np.ndarray, enrolled_plain: list[np.ndarray], band: float, angles: list[float],
                       best_rotations: list[float]) -> dict:
    """DEVELOPMENT ONLY. From the weighted live sequence and the weighted, UN-KEYED enrolled sequences: per-finger share
    of the squared DTW cost and mean fingertip extension (palm sizes, unweighted) live vs each enrolled sample, and the
    median distance under each DIAGNOSTIC_WEIGHT_SETS weighting. Scalars only."""
    from preprocessing.hand_gesture import FEATURE_WEIGHTS

    fingers = list(FEATURE_GROUPS["fingers"])
    live_raw = np.asarray(live, float) / FEATURE_WEIGHTS
    out = {"finger_shares": [], "finger_extension_live": {n: round(float(live_raw[:, c].mean()), 3)
                                                          for n, c in zip(FINGER_NAMES, fingers)},
           "finger_extension_enrolled": [], "shadow_distances": {}}
    for rotation, reference in zip(best_rotations, enrolled_plain):
        candidate = rotate_features(live, rotation).astype(np.float64)
        _, path = dtw_alignment(candidate, reference, band)
        diff2 = np.array([(candidate[i] - np.asarray(reference[j], float)) ** 2 for i, j in path]).sum(axis=0)
        total = float(diff2.sum()) or 1.0
        out["finger_shares"].append({n: round(float(diff2[c]) / total, 4) for n, c in zip(FINGER_NAMES, fingers)})
        ref_raw = np.asarray(reference, float) / FEATURE_WEIGHTS
        out["finger_extension_enrolled"].append({n: round(float(ref_raw[:, c].mean()), 3) for n, c in zip(FINGER_NAMES, fingers)})
    for name, group_weights in DIAGNOSTIC_WEIGHT_SETS.items():
        w = _weight_vector(group_weights)
        cand = [rotate_features(live_raw * w, a) for a in angles]
        refs = [np.asarray(r, float) / FEATURE_WEIGHTS * w for r in enrolled_plain]
        per = [gesture_distance(cand, r, band) for r in refs]
        out["shadow_distances"][name] = {"per_sample": [round(d, 4) for d in per], "median": round(float(np.median(per)), 4)}
    return out


#: Enrollment outlier rule: sample i is refused when its distance to the CLOSER of the other two samples exceeds both
#: the matching threshold and OUTLIER_FACTOR x the distance between those two (they agree with each other, it agrees
#: with neither). Development default (docs/HAND_GESTURE_MODALITY.md).
OUTLIER_FACTOR = 2.0


class HandEnrollmentIncomplete(ValueError):
    """Enrollment needs exactly `Settings.hand_gesture_enrollment_attempts` independent gesture samples (and no duplicates)."""


class _LandmarkPipeline:
    """The backend has no hand model: MediaPipe runs in the browser. Present for interface parity (`is_mock`)."""

    is_mock = False
    embedding_dim = FEATURE_DIM


def fusion_score(distance: float, threshold: float) -> float:
    return float(np.clip(1.0 - distance / (2.0 * threshold), -1.0, 1.0))


def rotation_angles(tolerance_deg: float, step_deg: float = HAND_ROTATION_STEP_DEG) -> list[float]:
    """The rotations the live gesture is tried at: 0, +/- step, ... up to +/- tolerance (0 first)."""
    if tolerance_deg <= 0:
        return [0.0]
    count = int(np.floor(tolerance_deg / step_deg + 1e-9))
    angles = [0.0] + [sign * k * step_deg for k in range(1, count + 1) for sign in (1, -1)]
    if count * step_deg < tolerance_deg - 1e-9:
        angles += [tolerance_deg, -tolerance_deg]
    return angles


def gesture_distance(candidates: list[np.ndarray], reference: np.ndarray, band: float) -> float:
    """min over the rotated versions of the live gesture of DTW(version, reference)."""
    return best_alignment(candidates, reference, band)[0]


def best_alignment(candidates: list[np.ndarray], reference: np.ndarray, band: float) -> tuple[float, int]:
    """(min DTW distance over the rotated versions of the live gesture, index of the version that achieved it)."""
    distances = [dtw_distance(c, reference, band) for c in candidates]
    index = int(np.argmin(distances))
    return distances[index], index


def feature_group_shares(candidate: np.ndarray, reference: np.ndarray, band: float) -> dict[str, float]:
    """DEVELOPMENT DIAGNOSTIC: each feature group's share of the squared (weighted) frame differences along the optimal
    DTW path between two PLAINTEXT sequences - which part of the representation dominates a distance."""
    _, path = dtw_alignment(candidate, reference, band)
    diff2 = np.array([(np.asarray(candidate[i], float) - np.asarray(reference[j], float)) ** 2 for i, j in path]).sum(axis=0)
    total = float(diff2.sum()) or 1.0
    return {group: round(float(diff2[list(cols)].sum()) / total, 4) for group, cols in FEATURE_GROUPS.items()}


def enrollment_analysis(sequences: list[np.ndarray], band: float, angles: list[float], threshold: float) -> dict:
    """Pairwise distances between the (plaintext, weighted) enrollment samples, the medoid, and a possible outlier.

    Distances use the same rotation search as authentication (symmetrized: the smaller of the two directions)."""
    n = len(sequences)
    rotated = [[rotate_features(s, a) for a in angles] for s in sequences]
    matrix = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        matrix[i, j] = matrix[j, i] = min(gesture_distance(rotated[i], sequences[j], band),
                                          gesture_distance(rotated[j], sequences[i], band))
    pairwise = [float(matrix[i, j]) for i, j in itertools.combinations(range(n), 2)]
    medoid = int(np.argmin(matrix.sum(axis=1)))
    outlier = None
    for i in range(n):
        others = [j for j in range(n) if j != i]
        nearest = min(matrix[i, j] for j in others)
        between_others = np.median([matrix[a, b] for a, b in itertools.combinations(others, 2)])
        if nearest > threshold and nearest > OUTLIER_FACTOR * between_others:
            if outlier is None or nearest > outlier[1]:
                outlier = (i, float(nearest))
    return {"matrix": matrix, "pairwise": pairwise, "medoid": medoid, "outlier": outlier}


class HandGestureService:
    """Enroll / authenticate the dynamic hand gesture. Same outward interface as `ModalityService`."""

    def __init__(self, settings: Settings):
        self.modality = MODALITY
        self.settings = settings
        self.pipeline = _LandmarkPipeline()

    @property
    def angles(self) -> list[float]:
        return rotation_angles(self.settings.hand_gesture_rotation_tolerance_deg)

    # ------------------------------------------------------------------ capture -> features

    def process(self, capture: HandCapture, attempt: int | None = None) -> ProcessedGesture:
        """Validate one execution and build its feature sequence (raises `HandCaptureRejected`)."""
        return process_capture(capture, attempt=attempt)

    def gate(self, raw: HandCapture, sentence: int | None = None) -> None:
        """Capture-quality gate, as for voice: raises `HandCaptureRejected` for an unusable capture."""
        self.process(raw)

    def embed(self, raw: HandCapture) -> np.ndarray:
        """The weighted feature sequence DTW compares, (RESAMPLED_LENGTH, FEATURE_DIM)."""
        return weighted(self.process(raw).features)

    # ------------------------------------------------------------------ enrollment

    def enroll_attempts(
        self, db: Session, captures: list[HandCapture], user_id: str, application_id: str
    ) -> tuple[list[ProtectedTemplate], dict]:
        """Three independent Z samples -> validate each -> features -> consistency analysis -> one template per set.

        Each sample is a separate performance of the gesture (its own trajectory, timing and speed); all of them are
        kept as separate sequences inside the protected template, never averaged, overwritten or copied. Every sample
        must pass the capture gate; the first unusable one raises `HandCaptureRejected` (with its sample number) and
        NOTHING is stored. The same captured sequence submitted twice is refused: it is one performance, not two. A
        sample that disagrees with both others (while they agree) is refused as CAPTURE_QUALITY_FAILURE with its
        number (nothing is stored; the user records the samples again). Returns (rows, summary); the summary holds only non-biometric metadata.
        """
        required = self.settings.hand_gesture_enrollment_attempts
        if len(captures) != required:
            raise HandEnrollmentIncomplete(
                f"hand enrollment needs exactly {required} independent gesture samples, got {len(captures)}")
        gesture_types = {c.gesture_type for c in captures}
        if len(gesture_types) != 1:
            raise HandEnrollmentIncomplete("every enrollment attempt must use the same gesture_type")
        processed = [self.process(capture, attempt=n) for n, capture in enumerate(captures, start=1)]
        sequences = [weighted(p.features) for p in processed]
        for (i, a), (j, b) in itertools.combinations(enumerate(sequences, start=1), 2):
            if np.array_equal(a, b):
                raise HandEnrollmentIncomplete(
                    f"gesture samples {i} and {j} are the same captured sequence - each of the {required} samples must "
                    "be a separate performance of the gesture")
        band = self.settings.hand_gesture_dtw_band
        threshold = self.settings.hand_gesture_dtw_threshold
        analysis = enrollment_analysis(sequences, band, self.angles, threshold)
        if analysis["outlier"] is not None:
            index, nearest = analysis["outlier"]
            if self.settings.debug_scores:
                logger.info("ENROLL-DEBUG modality=hand outlier_sample=%d nearest_dtw=%.4f pairwise=%s", index + 1,
                            nearest, [round(d, 4) for d in analysis["pairwise"]])
            raise HandCaptureRejected(
                CAPTURE_QUALITY_FAILURE,
                f"sample {index + 1} was very different from the other two samples (a capture problem, not a "
                "mismatch) - record the samples again, drawing your natural Z each time", attempt=index + 1)
        pairwise = analysis["pairwise"]
        summary = {
            "template_format": HAND_TEMPLATE_FORMAT,
            "gesture_type": processed[0].gesture_type,
            "attempts": len(processed),
            "resampled_length": RESAMPLED_LENGTH,
            "feature_dim": FEATURE_DIM,
            "feature_weights": [float(w) for w in FEATURE_WEIGHTS],
            "dtw_band": band,
            "rotation_tolerance_deg": self.settings.hand_gesture_rotation_tolerance_deg,
            "aggregation": self.settings.hand_gesture_aggregation,
            "attempt_frames_detected": [p.metadata["frames_detected"] for p in processed],
            "attempt_durations_s": [p.metadata["duration_s"] for p in processed],
            "attempt_detection_ratio": [p.metadata["detection_ratio"] for p in processed],
            "handedness": processed[0].metadata["handedness"],
            # Intra-enrollment consistency (DTW between the user's own samples) - kept for calibration research.
            "intra_distance_median": round(float(np.median(pairwise)), 5),
            "intra_distance_max": round(float(np.max(pairwise)), 5),
            "medoid_sample": analysis["medoid"] + 1,
            "attempt_tracking_fps": [p.metadata["tracking_fps"] for p in processed],
            "attempt_motion_durations_s": [p.metadata["motion_duration_s"] for p in processed],
            # Tempo diagnostics per sample (never part of the decision).
            "attempt_path_lengths_palms": [p.metadata["path_length_palms"] for p in processed],
            # Share of time per Z stroke, "top/diagonal/bottom" per sample (scalars only - never feature sequences).
            "attempt_segment_times": ["/".join(f"{p.metadata[f'segment_time_{k}']:.2f}" for k in ("top", "diagonal", "bottom"))
                                      for p in processed],
        }
        if self.settings.debug_scores:
            logger.info("ENROLL-DEBUG modality=hand samples=%d pairwise_dtw=%s medoid=S%d frames=%s motion_s=%s "
                        "tracking_fps=%s", len(processed), [round(d, 4) for d in pairwise], summary["medoid_sample"],
                        summary["attempt_frames_detected"], summary["attempt_motion_durations_s"],
                        summary["attempt_tracking_fps"])
        try:
            first_key_version = crud.next_key_version(db, user_id, self.modality, application_id)
            crud.get_or_create_user(db, user_id)
            set_versions = crud.plan_enrollment_sets(db, user_id, application_id, self.settings.template_pool_size)
            entries = {
                set_version: self._entry(sequences, user_id, application_id, first_key_version + offset, summary)
                for offset, set_version in enumerate(set_versions)
            }
            rows = crud.save_modality_templates(
                db,
                user_id=user_id,
                application_id=application_id,
                modality=self.modality,
                template_version=HAND_TEMPLATE_VERSION,
                output_bits=0,
                entries=entries,
            )
        finally:
            del sequences
        return rows, summary

    def _entry(self, sequences: list[np.ndarray], user_id: str, application_id: str, key_version: int,
               metadata: dict) -> crud.PoolEntry:
        key = derive_key(self.settings.master_secret, application_id=application_id, user_id=user_id,
                         modality=self.modality, key_version=key_version)
        return crud.PoolEntry(
            key_version=key_version,
            protected_template=serialize(protect(sequences, key), HAND_TEMPLATE_VERSION),
            template_version=HAND_TEMPLATE_VERSION,
            output_bits=0,
            template_format=HAND_TEMPLATE_FORMAT,
            gesture_type=metadata["gesture_type"],
            template_metadata=metadata,
        )

    def rekey_entry(self, pool: list, user_id: str, application_id: str, key_version: int) -> crud.PoolEntry:
        """A new set's hand template: the ACTIVE set's enrolled sequences re-keyed under `key_version`.

        Used when a new template set is generated (template_sets.generate_template_set): one authorization gesture
        cannot replace three enrollment samples, so the enrolled samples are carried over under the new key.
        """
        stored = _active_row(pool, self.modality)
        if stored is None:
            raise crud.TemplateNotFoundError("No active hand-gesture template to re-key.")
        if stored.template_format != HAND_TEMPLATE_FORMAT:
            raise crud.TemplateNotFoundError("The active hand-gesture template uses a retired format; re-enroll the hand.")
        old_key = derive_key(self.settings.master_secret, application_id=application_id, user_id=user_id,
                             modality=self.modality, key_version=stored.key_version)
        _, protected = deserialize(stored.protected_template)
        sequences = unprotect(protected, old_key)
        try:
            return self._entry(sequences, user_id, application_id, key_version,
                               dict(stored.template_metadata or {}, gesture_type=stored.gesture_type))
        finally:
            del sequences

    # ------------------------------------------------------------------ authentication

    def authenticate(
        self, db: Session, raw: HandCapture, user_id: str, application_id: str, pool: list | None = None
    ) -> AuthenticationResult:
        """One execution -> validate -> features -> DTW against the ACTIVE set's enrolled samples -> threshold.

        Raises `HandCaptureRejected` for an unusable capture (a capture failure, never a mismatch).
        """
        pool = crud.get_set_rows(db, user_id, application_id) if pool is None else pool
        if _active_row(pool, self.modality) is None:
            return self._not_enrolled()
        started = time.perf_counter()
        processed = self.process(raw)
        features = weighted(processed.features)
        feature_ms = (time.perf_counter() - started) * 1000
        try:
            return self.authenticate_embedding(db, features, user_id, application_id, pool=pool,
                                               capture_metadata=processed.metadata, feature_ms=feature_ms)
        finally:
            del features

    def authenticate_embedding(
        self, db: Session, features: np.ndarray, user_id: str, application_id: str, pool: list | None = None,
        capture_metadata: dict | None = None, feature_ms: float | None = None,
    ) -> AuthenticationResult:
        """Compare an already-built (weighted, plaintext) feature sequence with the ACTIVE set's hand template."""
        pool = crud.get_set_rows(db, user_id, application_id) if pool is None else pool
        stored = _active_row(pool, self.modality)
        if stored is None:
            return self._not_enrolled()
        gesture_type = (capture_metadata or {}).get("gesture_type")
        if gesture_type and stored.gesture_type and gesture_type != stored.gesture_type:
            raise ValueError(f"the {gesture_type!r} gesture was presented but {stored.gesture_type!r} is enrolled")
        key = derive_key(self.settings.master_secret, application_id=application_id, user_id=user_id,
                         modality=self.modality, key_version=stored.key_version)
        assert_valid(
            validate_hand_authentication(settings=self.settings, stored=stored, key=key, candidate=features, pool=pool),
            modality=self.modality,
            user_id=user_id,
        )
        started = time.perf_counter()
        angles = self.angles
        # Rotation happens on the plaintext live sequence, THEN the set's key is applied: one transform per version,
        # the same key, dimensionality and feature order as the enrolled samples (no double transformation).
        candidates = protect([rotate_features(features, a) for a in angles], key)
        _, enrolled = deserialize(stored.protected_template)
        band = (stored.template_metadata or {}).get("dtw_band", self.settings.hand_gesture_dtw_band)
        aligned = [best_alignment(candidates, reference, band) for reference in enrolled]
        distances = [d for d, _ in aligned]
        best_rotations = [angles[i] for _, i in aligned]
        distance = aggregate(distances, self.settings.hand_gesture_aggregation)
        dtw_ms = (time.perf_counter() - started) * 1000
        threshold = self.settings.hand_gesture_dtw_threshold
        matched = distance <= threshold
        diagnostics = {
            "gesture_type": stored.gesture_type,
            "decision": MATCH if matched else GESTURE_MISMATCH,
            "dtw_distance": round(distance, 6),
            "dtw_distances": [round(d, 6) for d in distances],
            "threshold": threshold,
            "threshold_source": self.settings.hand_gesture_threshold_source,
            "aggregation": self.settings.hand_gesture_aggregation,
            "rotation_search_deg": self.settings.hand_gesture_rotation_tolerance_deg,
            "dtw_best_rotation_deg": best_rotations,
            "template_format": stored.template_format,
            "timing_ms": {"dtw_matching": round(dtw_ms, 3)},
        }
        if self.settings.debug_scores:
            # Development only: which feature groups dominate each distance. Needs the enrolled samples in plaintext:
            # un-keyed IN MEMORY with the set's key (as template re-keying does), never stored or returned.
            plain = unprotect(enrolled, key)
            try:
                diagnostics["feature_group_shares"] = [
                    feature_group_shares(rotate_features(features, r), reference, band)
                    for r, reference in zip(best_rotations, plain)]
                diagnostics.update(shadow_diagnostics(features, plain, band, angles, best_rotations))
            finally:
                del plain
        if feature_ms is not None:
            diagnostics["timing_ms"]["feature_extraction"] = round(feature_ms, 3)
        enrolled_durations = (stored.template_metadata or {}).get("attempt_motion_durations_s") or []
        live_duration = (capture_metadata or {}).get("motion_duration_s")
        if enrolled_durations and live_duration:
            # Tempo diagnostic: this gesture's movement time relative to the enrolled samples' median (1.0 = same pace).
            diagnostics["normalized_duration"] = round(float(live_duration) / float(np.median(enrolled_durations)), 3)
        for key_name, value in (capture_metadata or {}).items():
            if key_name.startswith("client_"):
                diagnostics["timing_ms"][key_name[len("client_"):]] = value
            elif key_name != "gesture_type":
                diagnostics[key_name] = value
        if self.settings.debug_scores:
            logger.info("AUTH-DEBUG modality=hand set=%d key=%d dtw=%.4f (%s) per_sample=%s best_rotation=%s threshold=%.3f "
                        "match=%s frames=%s duration=%s motion=%s idle_trimmed=%s tracking_fps=%s z=%s group_shares=%s",
                        stored.template_set_version, stored.key_version, distance, self.settings.hand_gesture_aggregation,
                        diagnostics["dtw_distances"], best_rotations,
                        threshold, matched, diagnostics.get("frames_detected"), diagnostics.get("duration_s"),
                        diagnostics.get("motion_duration_s"), diagnostics.get("idle_trimmed_s"),
                        diagnostics.get("tracking_fps"), diagnostics.get("z_validity"),
                        diagnostics.get("feature_group_shares"))
        return AuthenticationResult(
            score=fusion_score(distance, threshold),
            threshold=HAND_FUSION_THRESHOLD,
            authenticated=matched,
            distance=distance,
            metric=METRIC,
            metric_value=distance,
            metric_threshold=threshold,
            metric_higher_is_better=False,
            template_version=stored.template_version,
            key_version=stored.key_version,
            template_set_version=stored.template_set_version,
            template_status=stored.template_status or "",
            diagnostics=diagnostics,
        )

    def _not_enrolled(self) -> AuthenticationResult:
        return AuthenticationResult(score=0.0, threshold=HAND_FUSION_THRESHOLD, authenticated=False)


@lru_cache
def get_hand_service() -> HandGestureService:
    return HandGestureService(get_settings())


__all__ = ["HandCaptureRejected", "HandEnrollmentIncomplete", "HandGestureService", "get_hand_service"]
