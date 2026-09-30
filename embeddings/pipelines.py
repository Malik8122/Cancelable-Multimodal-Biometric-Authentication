"""Modality pipelines: preprocessing + embedding model behind one call.

These are the objects the rest of the system (backend services in Phase 2,
fusion in Phase 3, and evaluation/experiments.py) should import - they hide
the fact that face and voice use completely different preprocessing
under the hood, exposing the same `embed(...)` method for both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from embeddings.constants import DEFAULT_CHECKPOINTS
from models.face.inference import FaceEmbedder
from models.voice.inference import VoiceEmbedder
from preprocessing.face import (
    ALIGNMENT_FAILED,
    LANDMARK_FAILURE,
    BLUR_MIN_SHARPNESS,
    MAX_CENTER_OFFSET,
    MAX_ROLL_DEGREES,
    MAX_YAW_RATIO,
    MIN_DETECTION_CONFIDENCE,
    MIN_FACE_SIZE_RATIO,
    AlignmentFailed,
    LandmarkFailure,
    FacePreprocessor,
    FacePreprocessorAligned,
    MultipleFacesDetected,
)
from preprocessing.voice import TARGET_SAMPLE_RATE, VoicePreprocessor


def _evaluate_face_quality(detection) -> str:
    """VALID / BLURRY / TOO_SMALL / OFF_CENTER / TOO_ANGLED / LOW_CONFIDENCE for one detected,
    single face (`preprocessing.face.FaceDetection`).

    Checked in a fixed order - confidence, then sharpness (the project's existing gate), then
    size, centering, and angle (new: see preprocessing/face.py for how each threshold was
    measured) - the first gate a capture fails is its verdict. `MULTIPLE_FACES`/`NO_FACE` are
    decided earlier, by `FacePreprocessor.detect_and_align` itself raising before this is called.
    """
    if detection.probability < MIN_DETECTION_CONFIDENCE:
        return "LOW_CONFIDENCE"
    if detection.sharpness < BLUR_MIN_SHARPNESS:
        return "BLURRY"
    if detection.face_size_ratio < MIN_FACE_SIZE_RATIO:
        return "TOO_SMALL"
    if detection.center_offset > MAX_CENTER_OFFSET:
        return "OFF_CENTER"
    if abs(detection.roll_degrees) > MAX_ROLL_DEGREES or abs(detection.yaw_ratio) > MAX_YAW_RATIO:
        return "TOO_ANGLED"
    return "VALID"


class ModalityPipeline:
    """Base class: preprocess(raw_image) -> embedder.extract_embedding(...)."""

    def __init__(self, preprocessor, embedder):
        self._preprocessor = preprocessor
        self._embedder = embedder

    def embed(self, raw_image: np.ndarray) -> np.ndarray:
        processed = self._preprocessor.preprocess(raw_image)
        if processed.ndim == 2:
            processed = np.stack([processed] * 3, axis=-1)
        return self._embedder.extract_embedding(processed)

    @property
    def embedding_dim(self) -> int:
        return self._embedder.embedding_dim

    @property
    def is_mock(self) -> bool:
        return self._embedder.mock_mode


#: Face preprocessing variants: "bbox" = MTCNN bounding-box crop (baseline, the model's training preprocessing),
#: "similarity" = 5-landmark similarity alignment (preprocessing/face.py::FacePreprocessorAligned).
FACE_ALIGNMENT_MODES = ("bbox", "similarity")
#: Task-level names of the two modes.
FACE_BASELINE, FACE_ALIGNED = FACE_ALIGNMENT_MODES


class FacePipeline(ModalityPipeline):
    def __init__(self, checkpoint_path: str | Path | None = DEFAULT_CHECKPOINTS["face"], device: str = "cpu", alignment: str = "bbox"):
        if alignment not in FACE_ALIGNMENT_MODES:
            raise ValueError(f"Unknown face alignment {alignment!r}; expected one of {FACE_ALIGNMENT_MODES}")
        preprocessor = FacePreprocessorAligned(device=device) if alignment == "similarity" else FacePreprocessor(device=device)
        super().__init__(preprocessor, FaceEmbedder(checkpoint_path=checkpoint_path, device=device))
        self.alignment = alignment

    def check_capture(self, image: np.ndarray) -> str:
        """Verdict for one enrollment pose (nothing is embedded): VALID, NO_FACE, MULTIPLE_FACES,
        BLURRY, TOO_SMALL, OFF_CENTER, TOO_ANGLED, LOW_CONFIDENCE, or LANDMARK_FAILURE / ALIGNMENT_FAILED (aligned mode only)."""
        try:
            detection = self._preprocessor.detect_and_align(image)
        except MultipleFacesDetected:
            return "MULTIPLE_FACES"
        except LandmarkFailure:
            return LANDMARK_FAILURE
        except AlignmentFailed:
            return ALIGNMENT_FAILED
        except ValueError:
            return "NO_FACE"
        return _evaluate_face_quality(detection)

    def capture_quality(self, image: np.ndarray) -> dict:
        """The enrollment quality signals (scalars only) and the verdict the enrollment gate WOULD give, for any
        capture - including an authentication capture, which is never gated. Diagnostics only: nothing here is
        used in any decision."""
        verdict = self.check_capture(image)
        try:
            d = self._preprocessor.detect_and_align(image)
        except ValueError:
            return {"verdict": verdict}
        return {"verdict": verdict, "detection_confidence": d.probability, "sharpness": d.sharpness,
                "face_size_ratio": d.face_size_ratio, "center_offset": d.center_offset,
                "roll_degrees": d.roll_degrees, "yaw_ratio": d.yaw_ratio}

    def embed_poses(self, captures: list[tuple[str, np.ndarray]]) -> tuple[list[np.ndarray], list[dict]]:
        """Multi-pose enrollment: one embedding per VALID capture, plus a per-pose report.

        Each capture goes through the existing MTCNN detection + alignment and the same quality
        gates `check_capture` applies (no face, more than one face, blurry, too small, off-center,
        too angled, or low detection confidence all reject a capture before it is embedded - see
        `_evaluate_face_quality` and `preprocessing/face.py` for how each threshold was measured).
        Valid ones are embedded by the unchanged FaceNet model (the same call authentication
        makes). Returns (embeddings of the valid poses, [{"pose", "status"}, ...] for every capture).
        """
        embeddings: list[np.ndarray] = []
        report: list[dict] = []
        for pose, image in captures:
            try:
                detection = self._preprocessor.detect_and_align(image)
            except MultipleFacesDetected:
                report.append({"pose": pose, "status": "MULTIPLE_FACES"})
                continue
            except LandmarkFailure:
                report.append({"pose": pose, "status": LANDMARK_FAILURE})
                continue
            except AlignmentFailed:
                report.append({"pose": pose, "status": ALIGNMENT_FAILED})
                continue
            except ValueError:
                report.append({"pose": pose, "status": "NO_FACE"})
                continue
            verdict = _evaluate_face_quality(detection)
            if verdict != "VALID":
                report.append({"pose": pose, "status": verdict})
                continue
            embeddings.append(self._embedder.extract_embedding(detection.aligned))
            report.append({"pose": pose, "status": "VALID"})
        return embeddings, report


class VoicePipeline(ModalityPipeline):
    """Like FacePipeline, but audio needs one extra piece of
    information the base class's single-argument `embed(raw_image)` doesn't
    carry: the input's sample rate (a raw waveform alone doesn't say whether
    it's 16kHz or 44.1kHz). `embed()` is overridden accordingly; the
    constructor shape, `embedding_dim`, and `is_mock` are unchanged.
    """

    def __init__(self, checkpoint_path: str | Path | None = DEFAULT_CHECKPOINTS["voice"], device: str = "cpu"):
        super().__init__(VoicePreprocessor(), VoiceEmbedder(checkpoint_path=checkpoint_path, device=device))

    def embed(self, raw_audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
        processed = self._preprocessor.preprocess(raw_audio, sample_rate=sample_rate, training=False)
        if processed.ndim == 2:
            processed = np.stack([processed] * 3, axis=-1)
        return self._embedder.extract_embedding(processed)
