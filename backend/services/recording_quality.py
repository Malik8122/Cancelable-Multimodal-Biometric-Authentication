"""Quality bands for the voice enrollment consistency check.

The two enrollment recordings are compared by the COSINE SIMILARITY OF THEIR ECAPA-TDNN EMBEDDINGS - never by waveform
or mel-spectrogram comparison, and not through BioHash. Speaker embeddings are built to be robust to natural variation
(pace, phrasing, mild noise), so the check tolerates it and only rejects recordings that really do not belong together.

    cosine >= 0.85          EXCELLENT   enrolled
    0.75 <= cosine < 0.85   GOOD        enrolled
    0.60 <= cosine < 0.75   FAIR        LOW_QUALITY_WARNING: nothing is stored until the user chooses to continue
                                        (or re-records)
    cosine < 0.60           POOR        ENROLLMENT_INCONSISTENT: rejected, nothing stored

Measured with the real model (TTS speech): the same speaker at different paces 0.98-0.99; with moderate microphone noise
0.66-0.73 (FAIR); a different speaker 0.51-0.56 (POOR).

This only gates *enrollment*. It changes no authentication threshold, no fusion logic and no template generation.
"""

from __future__ import annotations

import numpy as np

EXCELLENT = "EXCELLENT"
GOOD = "GOOD"
FAIR = "FAIR"
POOR = "POOR"

EXCELLENT_MIN = 0.85
GOOD_MIN = 0.75
FAIR_MIN = 0.60

#: Wording shared with the frontend (which shows the same sentences).
FAIR_MESSAGE = "Your recordings are usable, but quality is lower than recommended."
POOR_MESSAGE = "The two recordings appear to be from different speakers or are too noisy."


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Cosine of the angle between two embeddings (norm-safe: works whether or not they are unit length)."""
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def classify(similarity: float) -> str:
    """The quality band for a consistency similarity."""
    if similarity >= EXCELLENT_MIN:
        return EXCELLENT
    if similarity >= GOOD_MIN:
        return GOOD
    if similarity >= FAIR_MIN:
        return FAIR
    return POOR
