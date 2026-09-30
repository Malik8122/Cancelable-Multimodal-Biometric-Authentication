"""Face preprocessing: detection + alignment.

Uses MTCNN (via facenet-pytorch) for face detection and landmark localization. This is the
standard preprocessing step expected by the InceptionResnetV1 embedding model used in
models/face/inference.py.

Classical CV vs deep learning split:
    - Detection & landmark localization: deep learning (MTCNN).
    - Alignment (crop/resize to canonical size): classical geometric transform, driven by the
      detected bounding box. NOTE: facenet-pytorch's MTCNN computes 5-point landmarks but its
      `extract_face()` crops by bounding box alone - there is no landmark-based rotation/affine
      correction anywhere in this pipeline (confirmed directly against the installed
      facenet_pytorch source: `mtcnn.py::MTCNN.extract` -> `detect_face.py::extract_face` takes
      only a box). `detect_and_align()` below uses the same landmarks purely as QUALITY SIGNALS
      (is the face roughly frontal enough to enroll?), not to warp the image - true landmark
      alignment is a separate, not-yet-implemented change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FACE_INPUT_SIZE = 160  # required input size for InceptionResnetV1

#: Enrollment rejects a capture whose aligned crop has a Laplacian variance below this. Measured on real MTCNN crops of a
#: portrait: normal captures 116-920 (JPEG q30-q90, rotated, resized, dimmed); a visibly blurred face (gaussian sigma >= 2)
#: <= 34 and sigma >= 3 <= 13. 25 rejects genuine blur while passing ordinary webcam images. (Used for enrollment
#: pose validation only - authentication preprocessing is unchanged.)
BLUR_MIN_SHARPNESS = 25.0

#: Enrollment-quality gates below, all measured directly against MTCNN's real output on a real
#: portrait (matplotlib's bundled grace_hopper.jpg, the same image tests/test_face_enrollment.py
#: already uses) under controlled synthetic variation - not arbitrary guesses. See
#: docs/... (alignment investigation) for the measurement script. These are used for FACE
#: ENROLLMENT quality gating only (`FacePreprocessor.detect_and_align`'s callers in
#: embeddings/pipelines.py); authentication's `preprocess()` is completely unaffected.

#: MTCNN's own detection probability for the selected face. Measured: every real, in-frame,
#: reasonably-lit face in the test portrait (including far-away, off-center, and rotated variants)
#: scored 0.999+; only genuinely ambiguous/borderline detections would fall meaningfully below
#: 0.90, so this is a generous floor that only screens out real ambiguity, not normal variation.
MIN_DETECTION_CONFIDENCE = 0.90

#: Detected face bounding-box height as a fraction of the full frame height. Measured: a normally
#: -framed webcam-style capture of the test portrait was ~0.37; a face pushed far back in frame
#: (simulated by downscaling to 40%) measured ~0.14. 0.15 catches "much too far away" while
#: leaving generous room for normal seating-distance variation.
MIN_FACE_SIZE_RATIO = 0.15

#: Face bounding-box center's distance from the frame center, as a fraction of frame size
#: (Euclidean, normalized per-axis). Measured: normal/centered real-photo framing was ~0.12-0.17
#: (real photos are rarely pixel-perfect centered); a face shifted a quarter of the frame width
#: off-axis measured ~0.30. 0.35 leaves headroom above normal framing variance while catching a
#: face that's genuinely off to one side.
MAX_CENTER_OFFSET = 0.35

#: In-plane head tilt ("roll"), estimated from the eye-to-eye angle in MTCNN's 5-point landmarks.
#: Measured: natural small tilts (as a real webcam capture would show) stayed under ~12 degrees;
#: this project's OLD guided-pose protocol's deliberate left/right turns measured ~16 degrees of
#: apparent roll from the same synthetic-rotation test. 20 degrees allows natural head tilt while
#: excluding that old, no-longer-requested style of capture.
MAX_ROLL_DEGREES = 20.0

#: Horizontal yaw proxy: how far the nose sits from the eye-line midpoint, normalized by
#: inter-eye distance (0 = nose exactly centered between the eyes, as in a frontal face). Measured
#: the same way as roll above: natural variation stayed under ~0.10; the old deliberate-turn
#: protocol measured ~0.16-0.19. 0.20 mirrors the same allow-natural/exclude-deliberate-turn
#: margin as MAX_ROLL_DEGREES.
MAX_YAW_RATIO = 0.20


class MultipleFacesDetected(ValueError):
    """More than one face was detected in the frame. A `ValueError` subclass so any caller that
    only handles the older "no usable face" case still treats this as a detection failure, but
    callers that want to distinguish it (see embeddings/pipelines.py::FacePipeline) can catch it
    specifically."""

    def __init__(self, count: int):
        super().__init__(f"{count} faces detected in the provided image; expected exactly one.")
        self.count = count


@dataclass(frozen=True)
class FaceDetection:
    """One detected, aligned face plus the raw quality-signal measurements enrollment gates on.

    `aligned` is byte-for-byte what the old `detect_and_align` returned as its first element -
    nothing about the actual embedding input changed, only what else is measured and returned
    alongside it. Every field besides `aligned` is a single scalar - never a vector, image, or
    anything that could reconstruct one.
    """

    aligned: np.ndarray
    probability: float
    sharpness: float
    face_size_ratio: float
    center_offset: float
    roll_degrees: float
    yaw_ratio: float


class FacePreprocessor:
    """Detects, aligns, and crops a face from a raw RGB image."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._detector = None  # lazy-loaded; MTCNN is only needed when real preprocessing runs

    def _get_detector(self):
        if self._detector is None:
            from facenet_pytorch import MTCNN

            self._detector = MTCNN(
                image_size=FACE_INPUT_SIZE,
                margin=0,
                post_process=True,
                device=self.device,
            )
        return self._detector

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Detect + align the largest face in `image`.

        Returns an RGB uint8 array of shape (FACE_INPUT_SIZE, FACE_INPUT_SIZE, 3).
        Raises ValueError if no face is detected.

        UNCHANGED (authentication path): still the plain `detector(pil_image)` call - no quality
        gating here, matching this project's existing "authentication preprocessing is unchanged"
        invariant (see `detect_and_align`'s docstring below).
        """
        from PIL import Image

        detector = self._get_detector()
        pil_image = Image.fromarray(image)
        aligned = detector(pil_image)
        if aligned is None:
            raise ValueError("No face detected in the provided image.")

        # facenet-pytorch returns a normalized CHW torch tensor; convert back
        # to an HWC uint8 RGB array so the rest of the pipeline stays
        # framework-agnostic.
        arr = aligned.permute(1, 2, 0).detach().cpu().numpy()
        arr = ((arr * 128.0) + 127.5).clip(0, 255).astype(np.uint8)
        return arr

    def detect_and_align(self, image: np.ndarray) -> FaceDetection:
        """`preprocess` plus every quality signal enrollment gates on, from a SINGLE detection pass.

        Calls MTCNN's lower-level `detect(..., landmarks=True)` directly (rather than the `preprocess()`/
        `forward()` wrapper) so the bounding box and 5-point landmarks - both already computed internally by
        MTCNN regardless - are available for the size/centering/angle measurements below, instead of being
        discarded. The face crop itself (`FaceDetection.aligned`) is produced by the exact same
        `extract_face()` + `fixed_image_standardization()` steps `MTCNN.extract()` would otherwise run - not
        a different alignment method, just the same one with its intermediate values kept.

        Raises `ValueError` if no face is detected, `MultipleFacesDetected` if more than one is - both
        BEFORE any embedding-relevant work happens, matching the existing "faceless/unusable capture is
        rejected before touching the model" behavior.
        """
        import cv2
        from facenet_pytorch import extract_face, fixed_image_standardization
        from PIL import Image

        detector = self._get_detector()
        pil_image = Image.fromarray(image)
        box, probability, landmarks = self._detect_single(pil_image)

        face_tensor = extract_face(pil_image, box, image_size=detector.image_size, margin=detector.margin)
        face_tensor = fixed_image_standardization(face_tensor)
        arr = face_tensor.permute(1, 2, 0).detach().cpu().numpy()
        arr = ((arr * 128.0) + 127.5).clip(0, 255).astype(np.uint8)
        return self._with_quality_signals(arr, image, box, probability, landmarks)

    def _detect_single(self, pil_image):
        """(box, probability, landmarks) of the single confident face; ValueError / MultipleFacesDetected otherwise."""
        detector = self._get_detector()
        boxes, probs, points = detector.detect(pil_image, landmarks=True)
        if boxes is None or len(boxes) == 0:
            raise ValueError("No face detected in the provided image.")

        # `detect()` returns every candidate that clears MTCNN's internal cascade
        # thresholds (~0.6-0.7), including low-confidence background clutter (wall
        # decor, ceiling fixtures, shadows) well below the MIN_DETECTION_CONFIDENCE
        # bar a real, in-frame face clears. Filter to confident detections before
        # counting, so background noise isn't mistaken for a second face.
        confident = probs >= MIN_DETECTION_CONFIDENCE
        if confident.any():
            boxes, probs, points = boxes[confident], probs[confident], points[confident]
        else:
            best = int(np.argmax(probs))
            boxes, probs, points = boxes[best : best + 1], probs[best : best + 1], points[best : best + 1]

        if len(boxes) > 1:
            raise MultipleFacesDetected(len(boxes))

        return boxes[0], float(probs[0]), points[0]  # landmarks (5, 2): left_eye, right_eye, nose, mouth_left, mouth_right

    @staticmethod
    def _with_quality_signals(arr, image, box, probability, landmarks) -> "FaceDetection":
        """Quality signals: geometry from the ORIGINAL capture (box, landmarks), sharpness from `arr` (the crop that
        will be embedded)."""
        import cv2

        sharpness = float(cv2.Laplacian(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

        image_height, image_width = image.shape[0], image.shape[1]
        x1, y1, x2, y2 = box
        face_size_ratio = float((y2 - y1) / image_height)
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        center_offset = float(np.hypot(
            (center[0] - image_width / 2.0) / image_width,
            (center[1] - image_height / 2.0) / image_height,
        ))

        left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
        roll_degrees = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
        inter_eye_distance = float(np.linalg.norm(right_eye - left_eye))
        eye_midpoint_x = (left_eye[0] + right_eye[0]) / 2.0
        yaw_ratio = float((nose[0] - eye_midpoint_x) / inter_eye_distance) if inter_eye_distance > 0 else 0.0

        return FaceDetection(
            aligned=arr,
            probability=probability,
            sharpness=sharpness,
            face_size_ratio=face_size_ratio,
            center_offset=center_offset,
            roll_degrees=roll_degrees,
            yaw_ratio=yaw_ratio,
        )


# ----------------------------------------------------------------------------- landmark-based alignment

#: Canonical 5-point template in the 160x160 output frame (x right, y down), order: left eye, right eye, nose,
#: mouth left, mouth right ("left" = image-left, MTCNN's ordering). Derived, not chosen by hand: the mean landmark
#: position inside the BASELINE bounding-box crops of the face model's own fine-tuning images (LFW, >= 20 images per
#: identity, train split), left-right symmetrized so the eyes are level and the nose centred. Keeping the framing the
#: checkpoint was trained on makes geometric normalization the only change. Reproduce:
#: `python -m evaluation.ieee.face_alignment_template` -> evaluation/results/face_alignment_template.json.
ALIGNMENT_TEMPLATE_160 = np.array([
    [43.60, 63.15],
    [116.40, 63.15],
    [80.00, 94.05],
    [46.92, 120.99],
    [113.08, 120.99],
])

#: Landmark geometry below these is treated as degenerate (ALIGNMENT_FAILED) rather than warped.
MIN_ALIGNMENT_EYE_DISTANCE_PX = 4.0
ALIGNMENT_SCALE_RANGE = (0.05, 20.0)

ALIGNMENT_FAILED = "ALIGNMENT_FAILED"
LANDMARK_FAILURE = "LANDMARK_FAILURE"


class AlignmentFailed(ValueError):
    """Landmarks missing, malformed or degenerate, or no valid similarity transform. A ValueError subclass so every
    existing caller that turns a preprocessing ValueError into a clean rejection keeps doing so (no API crash)."""


class LandmarkFailure(AlignmentFailed):
    """The detector's landmarks are missing, malformed or non-finite (status LANDMARK_FAILURE) - as opposed to
    present-but-degenerate geometry, which is ALIGNMENT_FAILED. Subclass, so `except AlignmentFailed` still covers it."""


def validate_landmarks(landmarks) -> np.ndarray:
    """Return landmarks as a finite (5, 2) float array with a usable eye geometry. Raises LandmarkFailure if they are
    missing/malformed/non-finite, AlignmentFailed if they are degenerate (eyes too close, collinear)."""
    if landmarks is None:
        raise LandmarkFailure("No facial landmarks were returned by the detector.")
    try:
        pts = np.asarray(landmarks, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise LandmarkFailure(f"Malformed facial landmarks: {error}") from error
    if pts.shape != (5, 2) or not np.all(np.isfinite(pts)):
        raise LandmarkFailure(f"Expected 5 finite (x, y) landmarks, got shape {pts.shape}.")
    eye_distance = float(np.linalg.norm(pts[1] - pts[0]))
    if eye_distance < MIN_ALIGNMENT_EYE_DISTANCE_PX:
        raise AlignmentFailed(f"Degenerate landmarks: eye distance {eye_distance:.2f}px.")
    # all five points (nearly) on one line -> rotation is not determined
    centred = pts - pts.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    if singular[1] < 1e-3 * singular[0]:
        raise AlignmentFailed("Degenerate landmarks: points are collinear.")
    return pts


def estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation, uniform scale, translation; no reflection) mapping `src` onto
    `dst` (Umeyama 1991). Returns the 2x3 matrix M with dst ~= M @ [x, y, 1]. Raises AlignmentFailed if undefined."""
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s0, d0 = src - mu_s, dst - mu_d
    var_s = float((s0 ** 2).sum() / len(src))
    if var_s <= 1e-12:
        raise AlignmentFailed("Degenerate landmarks: zero spread.")
    cov = d0.T @ s0 / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:  # exclude reflections
        S[1, 1] = -1.0
    R = U @ S @ Vt
    scale = float(np.trace(np.diag(D) @ S) / var_s)
    if not np.isfinite(scale) or not (ALIGNMENT_SCALE_RANGE[0] <= scale <= ALIGNMENT_SCALE_RANGE[1]):
        raise AlignmentFailed(f"Implausible alignment scale {scale:.3f}.")
    t = mu_d - scale * R @ mu_s
    return np.c_[scale * R, t]


def align_face(image: np.ndarray, landmarks, size: int = FACE_INPUT_SIZE) -> np.ndarray:
    """Warp `image` so its 5 landmarks land on ALIGNMENT_TEMPLATE_160 (scaled to `size`). Bilinear interpolation,
    constant black border (the same fill MTCNN's box crop uses outside the image). Returns (size, size, 3) uint8."""
    import cv2

    pts = validate_landmarks(landmarks)
    M = estimate_similarity_transform(pts, ALIGNMENT_TEMPLATE_160 * (size / 160.0))
    return cv2.warpAffine(np.ascontiguousarray(image), M, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


#: The unchanged bounding-box pipeline, named explicitly for A/B experiments.
FacePreprocessorBaseline = FacePreprocessor


class FacePreprocessorAligned(FacePreprocessor):
    """MTCNN detection -> 5 landmarks -> similarity alignment to ALIGNMENT_TEMPLATE_160 -> 160x160 RGB uint8.

    Face selection mirrors the baseline exactly: `preprocess` (authentication) takes the LARGEST detected face, like
    MTCNN.forward(select_largest=True); `detect_and_align` (enrollment) applies the same confidence filter and
    single-face rule. Quality gates are unchanged and still reject BEFORE embedding: geometric signals (size, centring,
    roll, yaw) are measured on the original capture, sharpness on the aligned crop that is embedded. Landmarks are
    transient - never returned, stored or logged.
    """

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        boxes, _, points = self._get_detector().detect(Image.fromarray(image), landmarks=True)
        if boxes is None or len(boxes) == 0:
            raise ValueError("No face detected in the provided image.")
        boxes = np.asarray(boxes, dtype=np.float64)
        k = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        if points is None or points[k] is None:
            raise AlignmentFailed("No facial landmarks were returned by the detector.")
        return align_face(image, points[k])

    def detect_and_align(self, image: np.ndarray) -> FaceDetection:
        from PIL import Image

        box, probability, landmarks = self._detect_single(Image.fromarray(image))
        arr = align_face(image, landmarks)
        return self._with_quality_signals(arr, image, box, probability, validate_landmarks(landmarks))
