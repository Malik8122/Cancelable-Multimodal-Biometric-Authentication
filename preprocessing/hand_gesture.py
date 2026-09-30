"""Dynamic hand gesture preprocessing: MediaPipe Hands landmark sequence -> validated, normalized Z-gesture trajectory.

Landmark extraction (MediaPipe Hands, 21 landmarks per frame) runs in the browser (`@mediapipe/tasks-vision`
HandLandmarker, see docs/HAND_GESTURE_MODALITY.md); only the landmark sequence - never video - reaches the backend.
This module is pure numpy: it parses that sequence, rejects unusable captures, and builds the compact temporal
representation that `template_protection/dtw.py` compares.

The gesture is a handwritten-style Z drawn in the air with one hand: top-left -> top-right, diagonally down-left,
then left -> right. The identity signal is HOW the hand moves through the Z (palm trajectory, direction changes,
rhythm), not the exact finger pose.

Pipeline (one gesture execution; enrollment and authentication run exactly the same function):

    frames [(t_ms, 21 x (x, y, z) | None)]         MediaPipe image landmarks (x, y in [0, 1] of the frame)
            |
    parse + capture validation                       NO_HAND_DETECTED / INSUFFICIENT_FRAMES / CAPTURE_QUALITY_FAILURE /
            |                                        INVALID_TRAJECTORY / MALFORMED_CAPTURE  (never a biometric mismatch)
    user-view isotropic coordinates                  x * (width / height) so both axes share one unit; mirrored back to
            |                                        what the user saw in the selfie preview (so "left -> right" is theirs)
    gap filling + Gaussian smoothing (60 ms)         brief hand losses interpolated; jitter averaged out, same blur at
            |                                        any frame rate
            |
    idle trimming                                    still hand before / after the Z removed (pauses INSIDE it kept)
            |
    uniform time resampling to RESAMPLED_LENGTH      frame-rate independent; DTW absorbs the remaining speed changes
            |
    features (FEATURE_DIM = 13 per frame)            palm-centre trajectory normalized for position (centroid) and Z
            |                                        size (RMS radius); direction; relative speed / acceleration; hand
            |                                        orientation; fingertip extension (low weight) - see FEATURE_NAMES
    Z-structure check                                horizontal + down-left diagonal + horizontal (validity only)
            v
    (RESAMPLED_LENGTH, FEATURE_DIM) float32 sequence

Small camera / hand rotations are handled at matching time (`rotate_features`, a bounded search in
backend/services/hand_service.py), not by making the representation rotation invariant.

Every numeric default here is a development default, documented in docs/HAND_GESTURE_MODALITY.md; the ones that
affect the decision were checked with evaluation/hand_gesture_evaluation.py --synthetic (SYNTHETIC data only).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

#: Template/feature format. Bump when the representation changes so stored templates are never compared with a
#: different representation (`backend/services/hand_service.py` / `security_validation.py` refuse a mismatched stored
#: format; `backend/services/enrollment.py` reports such a user as not enrolled for the hand, i.e. re-enrollment).
#: v1 = infinity gesture (removed); v2 = Z gesture, trajectory-scaled representation.
HAND_TEMPLATE_FORMAT = "hand_gesture_v2"
HAND_TEMPLATE_VERSION = 2

NUM_LANDMARKS = 21
#: MediaPipe Hands landmark indices used below.
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, RING_MCP, RING_TIP, PINKY_MCP, PINKY_TIP = (
    0, 4, 5, 8, 9, 12, 13, 16, 17, 20)
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
#: Palm centre = mean of the wrist and the four finger MCP joints: the stable hand reference point whose trajectory is
#: the primary behavioural signal (far less jittery than any fingertip, and unaffected by finger posture).
PALM_POINTS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

#: Frames per gesture after uniform time resampling. A Z takes ~1.5-3.5 s; at the 10-30 fps the browser tracks that is
#: 15-105 frames, so 64 neither discards detail of a fast capture nor invents much for a slow one.
RESAMPLED_LENGTH = 64

FEATURE_NAMES = (
    "traj_x", "traj_y",            # PRIMARY: palm centre relative to the trajectory centroid, in units of the Z's RMS
                                   #          radius (position- and size-invariant); user view, y down
    "dir_x", "dir_y",              # PRIMARY: unit direction of palm-centre motion (fades to 0 while nearly still)
    "speed",                       # TEMPORAL: palm-centre speed / this gesture's mean speed (speed-invariant rhythm)
    "accel",                       # TEMPORAL: change of that relative speed per resampled step (scaled, clipped)
    "orient_x", "orient_y",        # SECONDARY: unit vector wrist -> middle-finger MCP (hand orientation)
    "tip_thumb", "tip_index", "tip_middle", "tip_ring", "tip_pinky",  # LOW: fingertip-to-wrist distance, palm sizes
)
FEATURE_DIM = len(FEATURE_NAMES)
#: Per-feature weights applied before DTW (Euclidean frame cost): the palm trajectory dominates, finger posture barely
#: counts. Within each rotatable 2-D pair the two weights are equal, so rotating the weighted features is the same as
#: rotating the raw ones. Development defaults (docs/HAND_GESTURE_MODALITY.md); stored in the template metadata.
FEATURE_WEIGHTS = np.array([1.0, 1.0, 0.6, 0.6, 0.3, 0.15, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
#: Feature groups (columns) - for development diagnostics of which group dominates a distance.
FEATURE_GROUPS = {
    "trajectory": (0, 1), "direction": (2, 3), "speed": (4,), "acceleration": (5,), "orientation": (6, 7),
    "fingers": (8, 9, 10, 11, 12),
}
#: Feature column pairs that are 2-D vectors in the image plane (rotated together by `rotate_features`).
ROTATABLE_PAIRS = ((0, 1), (2, 3), (6, 7))
#: Scale of the "accel" feature (per resampled step) and its clip, so a jittery frame cannot dominate the frame cost.
ACCEL_SCALE = 5.0
ACCEL_CLIP = 3.0

# Capture failure codes (never a biometric decision).
NO_HAND_DETECTED = "NO_HAND_DETECTED"
INSUFFICIENT_FRAMES = "INSUFFICIENT_FRAMES"
CAPTURE_QUALITY_FAILURE = "CAPTURE_QUALITY_FAILURE"
INVALID_TRAJECTORY = "INVALID_TRAJECTORY"
MALFORMED_CAPTURE = "MALFORMED_CAPTURE"
#: The sample could not be used at all (-> CAPTURE_ERROR) vs it was usable but too poor to compare (-> QUALITY_INSUFFICIENT).
CAPTURE_ERROR_CODES = (NO_HAND_DETECTED, INSUFFICIENT_FRAMES, MALFORMED_CAPTURE)
QUALITY_CODES = (CAPTURE_QUALITY_FAILURE, INVALID_TRAJECTORY)

REASONS = {
    NO_HAND_DETECTED: "no hand was detected - keep one hand fully in view of the camera",
    INSUFFICIENT_FRAMES: "the hand was tracked in too few frames - perform the gesture more slowly and keep the hand in view",
    CAPTURE_QUALITY_FAILURE: "the hand was lost for too much of the gesture, or the gesture was too short or too long",
    INVALID_TRAJECTORY: "the movement was not a complete Z - draw left to right, diagonally down to the left, then left "
                        "to right",
    MALFORMED_CAPTURE: "the gesture capture could not be read",
}


@dataclass(frozen=True)
class GestureSpec:
    """One supported gesture type. Adding a gesture later = one new entry in `GESTURE_TYPES` (plus its shape check)."""

    name: str
    description: str
    min_duration_s: float = 0.5
    max_duration_s: float = 10.0


GESTURE_TYPES: dict[str, GestureSpec] = {
    "z": GestureSpec("z", "handwritten-style Z drawn in the air with one hand: top-left -> top-right -> "
                          "diagonally down-left -> bottom-right"),
}
DEFAULT_GESTURE_TYPE = "z"

# Capture validation defaults (development values; see docs/HAND_GESTURE_MODALITY.md).
MIN_DETECTED_FRAMES = 15
MIN_DETECTION_RATIO = 0.6
MAX_FRAMES = 1200
#: Minimum extent of the palm-centre trajectory (bounding-box diagonal) in palm sizes: below this the hand barely moved.
MIN_TRAJECTORY_EXTENT = 1.0
#: Minimum palm size in normalized image units (a hand this small is too far away / a false detection).
MIN_PALM_SIZE = 0.01
#: Minimum rate at which the hand was actually tracked across the gesture (detected frames per second). 20 and 15 fps
#: are good, 12 acceptable, 10 the minimum; below it the Z's corners are sampled too sparsely. Evidence: real samples
#: tracked at ~5 fps were far from the same user's ~20 fps samples. A capture-quality failure, never a mismatch.
MIN_TRACKING_FPS = 10.0
#: Landmark smoothing: Gaussian kernel in TIME with this standard deviation (FWHM ~140 ms), applied on the real frame
#: timestamps - so the blur is the same at 12, 15 or 30 fps. A frame-count window smoothed a 30 fps capture but not a
#: 15 fps one, and the same Z tracked at the two rates then differed by ~0.1 DTW (synthetic check); with this kernel
#: ~0.05. Wide enough to average out MediaPipe jitter, narrow enough to keep the Z's corners and pauses.
SMOOTHING_SIGMA_S = 0.06
#: Motion segmentation: the gesture is the stretch from the first to the last frame where the palm centre moves faster
#: than max(MIN_ACTIVE_SPEED, ACTIVE_SPEED_FRACTION x the capture's 90th-percentile speed); the hand held still before
#: and after it is trimmed (kept with a short margin). Slow-downs and pauses at the Z's corners lie between the first
#: and last moving frames, so they are kept.
MIN_ACTIVE_SPEED = 0.5            # palm sizes per second
ACTIVE_SPEED_FRACTION = 0.15
MOTION_MARGIN_S = 0.05
#: See build_features: the direction features fade out below this fraction of the gesture's mean speed.
DIRECTION_SPEED_FLOOR = 0.25

# Z-structure check (validity only - never used for identity). The trajectory is resampled by arc length; a valid Z
# can be split at two corners into three strokes whose chords point right (+/- Z_HORIZONTAL_TOLERANCE_DEG), down-left
# (Z_DIAGONAL_RANGE_DEG, image y pointing down) and right again, each stroke at least Z_MIN_STROKE_FRACTION of the
# three chords together, and the chords must explain at least Z_MIN_EXPLAINED of the path length (a loop, scribble
# or back-and-forth does not). Deliberately lenient (loosened after real use): a sloppy, slanted, bowed, wobbly,
# rotated (~25 deg) or tall/wide Z passes; an incomplete / backwards / mirrored Z, an N, a circle or a swipe does not.
Z_CHECK_POINTS = 48
Z_MIN_STROKE_POINTS = 3
Z_HORIZONTAL_TOLERANCE_DEG = 55.0
Z_DIAGONAL_RANGE_DEG = (90.0, 180.0)
Z_MIN_STROKE_FRACTION = 0.10
Z_MIN_EXPLAINED = 0.55


class HandCaptureRejected(ValueError):
    """An unusable capture. `code` is one of the capture failure codes above; it is never a biometric mismatch."""

    def __init__(self, code: str, detail: str | None = None, attempt: int | None = None):
        self.code = code
        self.attempt = attempt
        self.reason = REASONS[code]
        prefix = f"gesture {attempt}: " if attempt else ""
        super().__init__(prefix + (detail or self.reason))

    @property
    def is_capture_error(self) -> bool:
        return self.code in CAPTURE_ERROR_CODES


@dataclass
class HandCapture:
    """One gesture execution as sent by the client (landmarks only)."""

    gesture_type: str
    timestamps_ms: np.ndarray                 # (F,) frame times, increasing
    landmarks: list[np.ndarray | None]        # F entries: (21, 3) or None when no hand was detected
    image_width: int
    image_height: int
    handedness: list[str | None] = field(default_factory=list)
    #: Client-side measurements (not biometric): e.g. mean MediaPipe inference ms per frame.
    client_metrics: dict = field(default_factory=dict)
    #: The user saw a mirrored (selfie) preview while drawing: raw x is flipped back to the user's left/right.
    mirrored: bool = True

    @property
    def frame_count(self) -> int:
        return len(self.landmarks)

    @property
    def detected_frames(self) -> int:
        return sum(lm is not None for lm in self.landmarks)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps_ms[-1] - self.timestamps_ms[0]) / 1000.0 if self.frame_count > 1 else 0.0


@dataclass
class ProcessedGesture:
    """A validated gesture: the feature sequence plus non-biometric capture metadata."""

    features: np.ndarray                      # (RESAMPLED_LENGTH, FEATURE_DIM) float32, unweighted
    gesture_type: str
    metadata: dict


@dataclass(frozen=True)
class ZCheck:
    """Outcome of the Z-structure check: valid, how much of the path the three strokes explain, and where the two
    corners fall (fraction of the path length). No coordinates."""

    valid: bool
    explained: float = 0.0
    corners: tuple[float, float] | None = None


# ----------------------------------------------------------------------------- parsing


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_capture(payload, attempt: int | None = None) -> HandCapture:
    """Parse one gesture execution: `{"gesture_type", "mirrored", "image_width", "image_height", "frames": [{"t",
    "landmarks"}]}`.

    `landmarks` is a list of 21 `[x, y, z]` triples or null (no hand in that frame). `mirrored` (default true: the
    browser preview is a selfie view) says the user saw the image flipped horizontally. Raises
    `HandCaptureRejected(MALFORMED_CAPTURE)` for anything structurally wrong.
    """
    def bad(detail: str) -> HandCaptureRejected:
        return HandCaptureRejected(MALFORMED_CAPTURE, f"malformed gesture capture: {detail}", attempt)

    if isinstance(payload, (bytes, str)):
        try:
            payload = json.loads(payload)
        except (ValueError, UnicodeDecodeError) as error:
            raise bad("not valid JSON") from error
    if not isinstance(payload, dict):
        raise bad("expected a JSON object")
    gesture_type = payload.get("gesture_type", DEFAULT_GESTURE_TYPE)
    if gesture_type not in GESTURE_TYPES:
        raise bad(f"unsupported gesture_type {gesture_type!r}; supported: {sorted(GESTURE_TYPES)}")
    mirrored = payload.get("mirrored", True)
    if not isinstance(mirrored, bool):
        raise bad("`mirrored` must be true or false")
    width, height = payload.get("image_width"), payload.get("image_height")
    if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0):
        raise bad("image_width and image_height must be positive integers")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise bad("`frames` must be a non-empty list")
    if len(frames) > MAX_FRAMES:
        raise bad(f"too many frames ({len(frames)} > {MAX_FRAMES})")

    timestamps, landmarks, handedness = [], [], []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or not _finite_number(frame.get("t")):
            raise bad(f"frame {index} needs a numeric `t` (milliseconds)")
        timestamps.append(float(frame["t"]))
        points = frame.get("landmarks")
        if points is None:
            landmarks.append(None)
            handedness.append(None)
            continue
        if not isinstance(points, list) or len(points) != NUM_LANDMARKS:
            raise bad(f"frame {index} must have {NUM_LANDMARKS} landmarks or null")
        if not all(isinstance(p, list) and len(p) == 3 and all(_finite_number(v) for v in p) for p in points):
            raise bad(f"frame {index}: every landmark must be [x, y, z] with finite numbers")
        landmarks.append(np.asarray(points, dtype=np.float64))
        side = frame.get("handedness")
        handedness.append(side if side in ("Left", "Right") else None)
    stamps = np.asarray(timestamps, dtype=np.float64)
    if np.any(np.diff(stamps) <= 0):
        raise bad("frame timestamps must be strictly increasing")
    metrics = payload.get("client_metrics") if isinstance(payload.get("client_metrics"), dict) else {}
    metrics = {k: float(v) for k, v in metrics.items() if isinstance(k, str) and _finite_number(v)}
    return HandCapture(gesture_type, stamps, landmarks, width, height, handedness, metrics, mirrored)


# ----------------------------------------------------------------------------- normalization


def isotropic(landmarks: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """MediaPipe normalizes x by the frame width and y by the frame height; rescale x by the aspect ratio so one unit
    has the same physical length on both axes. z is kept as MediaPipe reports it (relative depth, x-like scale)."""
    out = landmarks.astype(np.float64).copy()
    out[..., 0] *= image_width / image_height
    return out


def user_view(landmarks: np.ndarray, image_width: int, image_height: int, mirrored: bool) -> np.ndarray:
    """Isotropic coordinates as the user saw them: for a mirrored (selfie) preview x is flipped, so the user's own
    left -> right is +x. Every length and angle is unchanged; only the Z-structure check needs the true left/right."""
    out = isotropic(landmarks, image_width, image_height)
    if mirrored:
        out[..., 0] = image_width / image_height - out[..., 0]
    return out


def palm_size(frame: np.ndarray) -> float:
    """Hand scale reference: 2D distance wrist -> middle-finger MCP (stable under finger movement)."""
    return float(np.linalg.norm(frame[MIDDLE_MCP, :2] - frame[WRIST, :2]))


def normalize_shape(frame: np.ndarray) -> np.ndarray:
    """Translation + scale normalization of one frame: wrist at the origin, palm size 1 (x, y and z)."""
    scale = palm_size(frame)
    if scale <= 0:
        raise ValueError("degenerate hand: zero palm size")
    return (frame - frame[WRIST]) / scale


def _fill_gaps(values: np.ndarray, present: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Linear interpolation in time over frames where the hand was not detected (edges held constant)."""
    filled = values.copy()
    flat = filled.reshape(len(filled), -1)
    for column in range(flat.shape[1]):
        flat[~present, column] = np.interp(times[~present], times[present], flat[present, column])
    return filled


def _smooth(values: np.ndarray, times_ms: np.ndarray, sigma_s: float = SMOOTHING_SIGMA_S) -> np.ndarray:
    """Gaussian smoothing along time on the actual (possibly irregular) frame timestamps: every frame becomes the
    kernel-weighted mean of the frames around it. Frame-rate independent; reduces per-frame landmark jitter."""
    if sigma_s <= 0 or len(values) < 3:
        return values
    t = np.asarray(times_ms, dtype=np.float64) / 1000.0
    weights = np.exp(-0.5 * ((t[:, None] - t[None, :]) / sigma_s) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)
    return (weights @ values.reshape(len(values), -1)).reshape(values.shape)


def resample(values: np.ndarray, times: np.ndarray, length: int = RESAMPLED_LENGTH) -> np.ndarray:
    """Uniform resampling in time to `length` frames (the gesture's overall duration and frame rate are factored out)."""
    target = np.linspace(times[0], times[-1], length)
    flat = values.reshape(len(values), -1)
    out = np.stack([np.interp(target, times, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    return out.reshape((length,) + values.shape[1:])


# ----------------------------------------------------------------------------- features


def build_features(frames: np.ndarray) -> np.ndarray:
    """(T, 21, 3) user-view, gap-free, resampled landmark frames -> (T, FEATURE_DIM) features (see `FEATURE_NAMES`)."""
    palm_centre = frames[:, PALM_POINTS, :2].mean(axis=1)                     # (T, 2)
    centred = palm_centre - palm_centre.mean(axis=0)                          # position of the Z in the frame removed
    radius = float(np.sqrt(np.mean(np.sum(centred ** 2, axis=1))))
    if radius <= 0:
        raise ValueError("degenerate trajectory: the palm centre never moved")
    trajectory = centred / radius                                             # size of the Z removed
    velocity = np.gradient(trajectory, axis=0)                                # per resampled step
    speed = np.linalg.norm(velocity, axis=1)
    mean_speed = max(float(speed.mean()), 1e-9)
    # Direction is only meaningful while the hand actually moves: below DIRECTION_SPEED_FLOOR x the mean speed it fades
    # to 0 instead of becoming a random unit vector of landmark jitter (e.g. during a pause at a corner).
    direction = velocity / np.maximum(speed, DIRECTION_SPEED_FLOOR * mean_speed)[:, None]
    relative_speed = speed / mean_speed
    acceleration = np.clip(np.gradient(relative_speed) * ACCEL_SCALE, -ACCEL_CLIP, ACCEL_CLIP)
    shapes = np.stack([normalize_shape(f) for f in frames])                   # (T, 21, 3), wrist origin, palm size 1
    tip_distances = np.linalg.norm(shapes[:, FINGERTIPS, :], axis=2)          # (T, 5), uses x, y and z
    orientation = shapes[:, MIDDLE_MCP, :2]
    orientation = orientation / np.maximum(np.linalg.norm(orientation, axis=1), 1e-9)[:, None]
    return np.column_stack([trajectory, direction, relative_speed, acceleration, orientation,
                            tip_distances]).astype(np.float32)


def weighted(features: np.ndarray) -> np.ndarray:
    """Apply `FEATURE_WEIGHTS` (the representation DTW actually compares)."""
    return (features.astype(np.float64) * FEATURE_WEIGHTS).astype(np.float32)


def rotate_features(features: np.ndarray, degrees: float) -> np.ndarray:
    """The same gesture as if the camera / hand were rotated by `degrees` in the image plane: every 2-D vector feature
    (trajectory about its centroid, direction, orientation) is rotated; speeds and distances are rotation invariant.
    Works on weighted or unweighted features (the two weights of a pair are equal)."""
    if not degrees:
        return features
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    out = np.asarray(features, dtype=np.float64).copy()
    for i, j in ROTATABLE_PAIRS:
        x, y = out[:, i].copy(), out[:, j].copy()
        out[:, i], out[:, j] = c * x - s * y, s * x + c * y
    return out.astype(np.float32)


# ----------------------------------------------------------------------------- Z structure (validity only)


def _arc_resample(points: np.ndarray, count: int) -> tuple[np.ndarray, float]:
    """`count` points equally spaced along the path, and the path length."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    length = float(cumulative[-1])
    if length <= 0:
        return np.repeat(points[:1], count, axis=0), 0.0
    target = np.linspace(0.0, length, count)
    return np.column_stack([np.interp(target, cumulative, points[:, 0]), np.interp(target, cumulative, points[:, 1])]), length


def z_structure(trajectory: np.ndarray) -> ZCheck:
    """Does the (user-view, y down) palm trajectory have the broad structure of a Z - rightwards, down-left, rightwards?

    A lightweight validity check, NOT an identity signal: it only decides whether the capture is a complete Z worth
    comparing. Exhaustive search over the two corner positions on an arc-length resampled path (48 points)."""
    points, length = _arc_resample(np.asarray(trajectory, dtype=np.float64)[:, :2], Z_CHECK_POINTS)
    if length <= 0:
        return ZCheck(False)
    n, k = Z_CHECK_POINTS, Z_MIN_STROKE_POINTS
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    usable = (i >= k) & (j - i >= k) & (n - 1 - j >= k)
    s1, s2, s3 = points[i] - points[0], points[j] - points[i], points[-1] - points[j]      # (n, n, 2) stroke chords

    def angle(v):
        return np.degrees(np.arctan2(v[..., 1], v[..., 0]))

    l1, l2, l3 = (np.linalg.norm(v, axis=-1) for v in (s1, s2, s3))
    chords = l1 + l2 + l3
    shape_ok = (
        (np.abs(angle(s1)) <= Z_HORIZONTAL_TOLERANCE_DEG)
        & (np.abs(angle(s3)) <= Z_HORIZONTAL_TOLERANCE_DEG)
        & (angle(s2) >= Z_DIAGONAL_RANGE_DEG[0]) & (angle(s2) <= Z_DIAGONAL_RANGE_DEG[1])
        & (np.minimum(np.minimum(l1, l2), l3) >= Z_MIN_STROKE_FRACTION * chords)
    )
    explained = np.where(usable & shape_ok, chords / length, 0.0)
    best = np.unravel_index(int(np.argmax(explained)), explained.shape)
    score = float(explained[best])
    if score < Z_MIN_EXPLAINED:
        return ZCheck(False, round(score, 3))
    return ZCheck(True, round(score, 3), (round(best[0] / (n - 1), 3), round(best[1] / (n - 1), 3)))


# ----------------------------------------------------------------------------- the whole pipeline


def active_span(palm_centre: np.ndarray, times_ms: np.ndarray, palm: float) -> tuple[int, int]:
    """Indices [start, end] of the moving part of a gesture (see MIN_ACTIVE_SPEED): the idle hand before and after the
    movement is excluded, with MOTION_MARGIN_S of context; pauses between the first and last movement are kept.
    Returns the full range when nothing clearly moves (the trajectory-extent check then rejects the capture)."""
    t = times_ms / 1000.0
    if len(t) < 3:
        return 0, len(t) - 1
    speed = np.linalg.norm(np.gradient(palm_centre, t, axis=0), axis=1) / max(palm, 1e-9)
    threshold = max(MIN_ACTIVE_SPEED, ACTIVE_SPEED_FRACTION * float(np.percentile(speed, 90)))
    active = np.flatnonzero(speed > threshold)
    if len(active) == 0:
        return 0, len(t) - 1
    start = int(np.searchsorted(t, t[active[0]] - MOTION_MARGIN_S, side="left"))
    end = int(np.searchsorted(t, t[active[-1]] + MOTION_MARGIN_S, side="right")) - 1
    return max(start, 0), min(end, len(t) - 1)


@dataclass
class MotionSample:
    """A validated capture's moving part before feature building: user-view, gap-filled, smoothed, idle-trimmed
    landmark frames with their timestamps, plus capture metadata. Shared by `process_capture` and the representation
    experiments (evaluation/hand_representation_experiment.py), so every variant sees exactly the same gated input."""

    frames: np.ndarray                        # (T, 21, 3)
    times_ms: np.ndarray                      # (T,)
    palm: float                               # median palm size (normalized image units)
    metadata: dict


def motion_sample(capture: HandCapture, attempt: int | None = None) -> MotionSample:
    """Every capture check up to (not including) feature building and the Z check - see `process_capture`."""
    spec = GESTURE_TYPES[capture.gesture_type]
    present = np.array([lm is not None for lm in capture.landmarks])
    detected = int(present.sum())
    if detected == 0:
        raise HandCaptureRejected(NO_HAND_DETECTED, attempt=attempt)
    if detected < MIN_DETECTED_FRAMES:
        raise HandCaptureRejected(
            INSUFFICIENT_FRAMES, f"the hand was tracked in {detected} frames (at least {MIN_DETECTED_FRAMES} needed)", attempt)
    ratio = detected / capture.frame_count
    # Only the span from the first to the last detected frame is the gesture; lead-in/lead-out without a hand is ignored.
    first, last = int(np.argmax(present)), int(len(present) - 1 - np.argmax(present[::-1]))
    span_present = present[first:last + 1]
    span_ratio = float(span_present.mean())
    times = capture.timestamps_ms[first:last + 1]
    duration = float(times[-1] - times[0]) / 1000.0
    if span_ratio < MIN_DETECTION_RATIO:
        raise HandCaptureRejected(
            CAPTURE_QUALITY_FAILURE, f"the hand was lost in {1 - span_ratio:.0%} of the gesture frames", attempt)
    tracking_fps = (int(span_present.sum()) - 1) / duration if duration > 0 else 0.0
    if tracking_fps < MIN_TRACKING_FPS:
        raise HandCaptureRejected(
            CAPTURE_QUALITY_FAILURE,
            f"the hand was tracked at only {tracking_fps:.1f} frames per second (at least {MIN_TRACKING_FPS:g} needed) - "
            "improve the lighting and close other apps using the camera", attempt)

    raw = np.stack([user_view(lm, capture.image_width, capture.image_height, capture.mirrored) if lm is not None
                    else np.zeros((NUM_LANDMARKS, 3)) for lm in capture.landmarks[first:last + 1]])
    sizes = np.array([palm_size(f) for f, ok in zip(raw, span_present) if ok])
    palm = float(np.median(sizes))
    if palm < MIN_PALM_SIZE:
        raise HandCaptureRejected(CAPTURE_QUALITY_FAILURE, "the hand is too small in the frame - move closer", attempt)
    frames = _smooth(_fill_gaps(raw, span_present, times), times)
    # Keep only the moving part: the idle hand before and after the gesture would otherwise shift the gesture inside
    # the resampled sequence by a different amount in every sample.
    start, end = active_span(frames[:, PALM_POINTS, :2].mean(axis=1), times, palm)
    frames, times = frames[start:end + 1], times[start:end + 1]
    motion_duration = float(times[-1] - times[0]) / 1000.0
    if not spec.min_duration_s <= motion_duration <= spec.max_duration_s:
        raise HandCaptureRejected(
            CAPTURE_QUALITY_FAILURE,
            f"the gesture movement lasted {motion_duration:.2f} s (expected {spec.min_duration_s:g}-{spec.max_duration_s:g} s)",
            attempt)
    palm_centre = frames[:, PALM_POINTS, :2].mean(axis=1)
    extent = float(np.linalg.norm(palm_centre.max(axis=0) - palm_centre.min(axis=0)) / palm)
    if extent < MIN_TRAJECTORY_EXTENT:
        raise HandCaptureRejected(
            INVALID_TRAJECTORY, f"the hand moved {extent:.2f} palm sizes (at least {MIN_TRAJECTORY_EXTENT:g} needed)", attempt)
    sides = [h for h in capture.handedness if h]
    metadata = {
        "gesture_type": capture.gesture_type,
        "frames_total": capture.frame_count,
        "frames_detected": detected,
        "detection_ratio": round(ratio, 4),
        "duration_s": round(duration, 3),
        "motion_duration_s": round(motion_duration, 3),
        "idle_trimmed_s": round(duration - motion_duration, 3),
        "tracking_fps": round(tracking_fps, 1),
        "trajectory_extent_palms": round(extent, 3),
        # Tempo diagnostics (never part of the decision): how far and how fast the palm travelled.
        "path_length_palms": round(float(np.linalg.norm(np.diff(palm_centre, axis=0), axis=1).sum()) / palm, 3),
        "handedness": max(set(sides), key=sides.count) if sides else None,
    }
    metadata["mean_speed_palms_per_s"] = round(metadata["path_length_palms"] / max(motion_duration, 1e-9), 3)
    return MotionSample(frames=frames, times_ms=times, palm=palm, metadata=metadata)


def segment_time_fractions(trajectory: np.ndarray, corners: tuple[float, float]) -> tuple[float, float, float]:
    """Share of the gesture's time spent in each Z stroke (top / diagonal / bottom), from a TIME-uniform trajectory and
    the Z check's corner positions (fractions of the path length). A tempo diagnostic: 0.30 / 0.40 / 0.30 at 2 s and
    0.29 / 0.42 / 0.29 at 3 s describe the same rhythm."""
    steps = np.linalg.norm(np.diff(np.asarray(trajectory, dtype=np.float64), axis=0), axis=1)
    progress = np.concatenate([[0.0], np.cumsum(steps)]) / max(float(steps.sum()), 1e-12)
    time = np.linspace(0.0, 1.0, len(progress))
    t1, t2 = (float(np.interp(c, progress, time)) for c in corners)
    return round(t1, 3), round(t2 - t1, 3), round(1.0 - t2, 3)


def process_capture(capture: HandCapture, attempt: int | None = None) -> ProcessedGesture:
    """Validate one capture and build its feature sequence. Raises `HandCaptureRejected` for an unusable capture.

    The SAME function serves enrollment and authentication, so both produce the same feature space: detected span ->
    tracking-rate check -> user-view coordinates -> gap filling + smoothing -> idle trimming -> duration / extent
    checks -> resampling -> features -> Z-structure check."""
    motion = motion_sample(capture, attempt)
    features = build_features(resample(motion.frames, motion.times_ms))
    if not np.all(np.isfinite(features)):
        raise HandCaptureRejected(INVALID_TRAJECTORY, "the trajectory produced non-finite features", attempt)
    shape = z_structure(features[:, :2])
    if not shape.valid:
        raise HandCaptureRejected(
            INVALID_TRAJECTORY,
            "the movement was not recognised as a complete Z (left to right, diagonally down to the left, then left to "
            "right) - draw the whole Z in one continuous movement", attempt)
    segments = segment_time_fractions(features[:, :2], shape.corners)
    metadata = {
        **motion.metadata,
        "z_validity": "PASS",
        "z_explained": shape.explained,
        "segment_time_top": segments[0],
        "segment_time_diagonal": segments[1],
        "segment_time_bottom": segments[2],
        "resampled_length": RESAMPLED_LENGTH,
        "feature_dim": FEATURE_DIM,
    }
    metadata.update({f"client_{k}": round(v, 3) for k, v in capture.client_metrics.items()})
    return ProcessedGesture(features=features, gesture_type=capture.gesture_type, metadata=metadata)
