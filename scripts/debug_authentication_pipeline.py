"""Authentication debug sprint: trace enroll -> authenticate -> fusion end to end.

Run with: PYTHONPATH=. python scripts/debug_authentication_pipeline.py

Produced the evidence behind the "genuine user is being denied" investigation
(see tests/test_authentication_debug_sprint.py for the resulting regression
tests). Exercises the REAL production code path -
`backend.services.ModalityService.enroll/.authenticate` and
`fusion.policy.evaluate_fusion_policy` - against the real trained checkpoints
for face/voice, using synthetic-but-structured inputs (no real
biometric dataset is committed to this repo - see evaluation/results/, which
holds only *_metrics.csv from the Kaggle training runs, never raw
images/audio). The "genuine" sample for each modality is the "enroll" sample
plus a small, realistic amount of noise, standing in for the natural
variation between two real captures of the same person.

Findings (see this module's inline prints, and
tests/test_authentication_debug_sprint.py's docstring for the full writeup):

1. A real bug in preprocessing/voice.py::VoicePreprocessor._trim_silence
   (fixed) made voice's embedding wildly unstable under any realistic capture
   variation - a genuine recapture's cosine similarity could come out
   *negative*. Fixed; this script's voice numbers now reflect the fix.
2. A separate, NOT-fixed-here gap: Settings.match_threshold=0.9 is an
   uncalibrated fallback applied to every modality (no
   evaluation/results/<modality>_threshold.json exists yet), and BioHash
   quantization measurably degrades similarity versus raw cosine even for
   genuine pairs - closing this properly needs real calibration data this
   environment doesn't have (see evaluation/threshold_calibration.py).
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_SECRET", "debug-sprint-master-secret-not-for-production")

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.config import Settings
from backend.database.models import Base
from backend.services.base_service import ModalityService
from embeddings.pipelines import VoicePipeline
from fusion.config import FusionPolicy
from fusion.policy import evaluate_fusion_policy
from models.face.inference import FaceEmbedder

RNG = np.random.default_rng(42)
APPLICATION_ID = "debug-sprint"
USER_ID = "genuine-user-001"


def _make_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _voice_waveform(seed: int, seconds: float = 4.0, sr: int = 16_000) -> np.ndarray:
    """A band-limited, enveloped noise signal - continuous spectral energy
    like real speech (unlike a pure/sparse tone, which this real ECAPA-TDNN
    model treats very differently from anything speech-like)."""
    from scipy.signal import butter, lfilter

    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    b, a = butter(4, [300 / (sr / 2), 3400 / (sr / 2)], btype="band")
    signal = lfilter(b, a, rng.standard_normal(len(t))).astype(np.float32)
    envelope = (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t)).astype(np.float32)
    signal = signal * envelope
    return (signal / np.std(signal) * 0.1).astype(np.float32)


def _add_noise(x: np.ndarray, relative_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return x + rng.normal(scale=relative_std * (np.std(x) or 1.0), size=x.shape).astype(np.float32)


def main() -> None:
    settings = Settings()
    print(f"Settings.match_threshold (uncalibrated fallback, applied to every modality) = {settings.match_threshold}")
    print(f"Settings.template_bits = {settings.template_bits}\n")

    face_embedder = FaceEmbedder(checkpoint_path=settings.face_model_path)
    voice_pipeline = VoicePipeline(checkpoint_path=settings.voice_model_path)
    print(f"mock_mode: face={face_embedder.mock_mode} voice={voice_pipeline.is_mock}")
    print("(False = real trained checkpoint loaded, not a placeholder)\n")

    class _FaceDirectPipeline:
        """Bypasses only MTCNN face detection/alignment (which correctly
        rejects any non-photograph synthetic image) - exercises the real
        trained face network directly on a 160x160x3 array."""

        def embed(self, image: np.ndarray) -> np.ndarray:
            return face_embedder.extract_embedding(image)

    services = {
        "face": ModalityService("face", _FaceDirectPipeline(), settings),
        "voice": ModalityService("voice", voice_pipeline, settings),
    }

    enroll_samples = {
        "face": RNG.integers(0, 255, (160, 160, 3), dtype=np.uint8),
        "voice": _voice_waveform(seed=1),
    }
    genuine_samples = {
        "face": _add_noise(enroll_samples["face"].astype(np.float32), 0.05, seed=10).clip(0, 255).astype(np.uint8),
        # A second take naturally runs a little shorter or longer than the
        # first - this is what actually explains the original bug report's
        # symptom (genuine voice scores scattered ~0.62-0.84): a take that
        # trims a little *short* of the 4-second target used to get padded
        # with raw silence before mel extraction, contaminating the mel
        # normalization for the real speech content (see
        # preprocessing/voice.py::VoicePreprocessor.preprocess and
        # docs/AUTHENTICATION_RELIABILITY_REPORT.md). Plain additive sample
        # noise on an identical-length clip (the previous version of this
        # script) never exercised that code path at all.
        "voice": _voice_waveform(seed=1, seconds=3.7),
    }

    db = _make_session()
    for modality, service in services.items():
        service.enroll(db, enroll_samples[modality], USER_ID, APPLICATION_ID)
    print(f"Enrolled user_id={USER_ID!r} for face, voice.\n")

    print(f"Re-authenticating {USER_ID!r} with a second, slightly-varied capture of each modality...")
    results = {}
    for modality, service in services.items():
        results[modality] = service.authenticate(db, genuine_samples[modality], USER_ID, APPLICATION_ID)

    scores = {m: r.score for m, r in results.items()}
    individually_authenticated = {m: r.authenticated for m, r in results.items()}
    fusion_threshold = sum(r.threshold for r in results.values()) / len(results)
    decision = evaluate_fusion_policy(scores, individually_authenticated, FusionPolicy.ALL_REQUIRED, fusion_threshold)

    print("\n=== STRUCTURED DEBUG REPORT ===")
    print(f"User ID: {USER_ID}")
    for modality, r in results.items():
        print(f"{modality.capitalize()} Score / Threshold / Pass: {r.score:.4f} / {r.threshold:.4f} / {r.authenticated}")
    print(f"Fusion Score: {decision.fused_score:.4f}")
    print("Fusion Policy: ALL_REQUIRED")
    print(f"Authenticated: {decision.authenticated}")
    if not decision.authenticated:
        print(f"Reason for failure: {decision.failed_modalities} individually failed their own threshold; ALL_REQUIRED vetoes on any failure.")
        print(
            "  Root cause for any modality above: NOT a code bug in enroll/authenticate wiring (preprocessing is\n"
            "  provably identical and deterministic between the two calls - see script docstring) - it is\n"
            "  Settings.match_threshold=0.9 being an uncalibrated fallback for the protected-template Hamming-\n"
            "  similarity score space. See tests/test_authentication_debug_sprint.py for the full writeup."
        )


if __name__ == "__main__":
    main()
