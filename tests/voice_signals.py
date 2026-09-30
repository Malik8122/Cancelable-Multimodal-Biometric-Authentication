"""Synthetic test "speech": tones with a syllable-rate loudness envelope.

A pure tone has no level variation, so the voice capture-quality gate (backend/services/voice_quality.py) correctly
reads it as all noise (estimated SNR ~0 dB) and asks for a re-recording. Test voices therefore carry a 4 Hz envelope
between 10% and 100% of full level - like syllables and short pauses - which the gate accepts. Test data only.
"""

from __future__ import annotations

import numpy as np


def speech_envelope(t: np.ndarray) -> np.ndarray:
    return 0.1 + 0.9 * (0.5 + 0.5 * np.cos(2 * np.pi * 4 * t))
