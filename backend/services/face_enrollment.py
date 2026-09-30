"""Face enrollment vocabulary: the five guided captures and the per-capture verdicts.

Face enrollment is a ONE-TIME, five-capture protocol - all five mostly-frontal (see
GuidedFaceCapture.tsx's guidance text): neutral, natural expression, slight natural left
variation, slight natural right variation, and a slight natural distance/expression variation.
Deliberately NOT extreme head turns (an earlier version of this protocol asked for front/left/
right/up/down poses with real head rotation - reverted because `preprocessing/face.py`'s MTCNN
step has no landmark-based rotation correction, so a rotated capture measurably hurts the
resulting centroid's similarity to a normal, frontal live capture - see the alignment
investigation). Per capture: MTCNN detection, alignment, quality gating (see
`preprocessing/face.py` and `embeddings/pipelines.py::_evaluate_face_quality` for the exact
checks and how each threshold was measured), one 512-d embedding for captures that pass. The
valid embeddings are averaged into a centroid, the temporary embeddings are discarded immediately,
and the T1-T4 template sets are generated from the centroid alone.
"""

from __future__ import annotations

FACE_POSES = ("front", "left", "right", "up", "down")

#: Per-capture verdicts. NO_FACE/BLURRY/VALID are this project's original three; the rest are the
#: enrollment-quality gates added alongside the mostly-frontal protocol above (see
#: `embeddings/pipelines.py::_evaluate_face_quality`, `preprocessing/face.py`).
VALID = "VALID"
NO_FACE = "NO_FACE"
BLURRY = "BLURRY"
MULTIPLE_FACES = "MULTIPLE_FACES"
TOO_SMALL = "TOO_SMALL"
OFF_CENTER = "OFF_CENTER"
TOO_ANGLED = "TOO_ANGLED"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
ALIGNMENT_FAILED = "ALIGNMENT_FAILED"  # aligned preprocessing only: landmarks degenerate (no valid similarity transform)
LANDMARK_FAILURE = "LANDMARK_FAILURE"  # aligned preprocessing only: landmarks missing / malformed / non-finite
#: The task-level name POSE_INVALID is this module's existing TOO_ANGLED (roll > 20 deg or yaw ratio > 0.20).
POSE_INVALID = TOO_ANGLED

#: How many valid captures are needed to enroll. The guided UI collects all five (each is checked as it is captured and
#: retaken if rejected); the backend tolerates a couple of rejected captures so one bad capture never blocks enrollment,
#: but a centroid of fewer than three captures would not be a meaningful average.
MIN_VALID_POSES = 3

POSE_HINTS = {
    NO_FACE: "No face was detected. Face the camera in good light.",
    BLURRY: "The image is too blurry. Hold still and try again.",
    MULTIPLE_FACES: "More than one face was detected. Make sure you're alone in the frame.",
    TOO_SMALL: "Your face is too small in the frame. Move a little closer to the camera.",
    OFF_CENTER: "Your face isn't centered in the frame. Center your face and try again.",
    TOO_ANGLED: "Your head is turned or tilted too much. Face the camera more directly.",
    LOW_CONFIDENCE: "The camera couldn't get a clear, confident view of your face. Check your lighting and try again.",
    ALIGNMENT_FAILED: "Your facial features couldn't be located reliably. Face the camera in good light and try again.",
    LANDMARK_FAILURE: "Your eyes, nose and mouth couldn't be located. Face the camera in good light and try again.",
}


class FaceCaptureRejected(RuntimeError):
    """Too few valid poses to enroll; nothing was stored. `report` is the per-pose verdict list."""

    def __init__(self, report: list[dict]):
        valid = sum(1 for r in report if r["status"] == VALID)
        super().__init__(
            f"Only {valid} of {len(report)} captures were usable (need at least {MIN_VALID_POSES}). "
            "Nothing was enrolled - retake the rejected poses."
        )
        self.report = report
