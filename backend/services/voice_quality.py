"""Voice capture-quality gate: decide whether a recording is usable BEFORE it is embedded or compared.

A recording that fails is a capture-quality failure (the user is asked to record again) - never a biometric mismatch,
never evidence of an attack, never a reason to change a template. The same gate runs at enrollment and at
authentication, on the same measurements (preprocessing/voice.py::VoicePreprocessor.quality_metrics), so both use the
same signal conditioning. The gate changes nothing the model receives: a recording that passes is processed exactly as
before.

Checks, in order (the first failing one is the verdict):

    INVALID_AUDIO      no samples or non-finite samples
    NO_SPEECH          digital silence, or no frame above the voice-activity rule
    CLIPPED            more than `voice_max_clipping_fraction` of the samples at digital full scale
    TOO_LITTLE_SPEECH  less than `voice_min_speech_seconds` of speech after silence trimming
    TOO_NOISY          estimated SNR below `voice_min_estimated_snr_db`

The thresholds are Settings values. How the two measured ones were chosen (REAL DATA, VoxCeleb1 test split) is recorded
in evaluation/results/voice_capture_pipeline.csv and docs/VOICE_MODEL.md ("Capture-quality gate"); the clipping limit is
a conventional engineering value (the dataset contains no clipped clips to calibrate it on).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from backend.config import Settings

logger = logging.getLogger(__name__)

OK = "OK"
INVALID_AUDIO = "INVALID_AUDIO"
NO_SPEECH = "NO_SPEECH"
CLIPPED = "CLIPPED"
TOO_LITTLE_SPEECH = "TOO_LITTLE_SPEECH"
TOO_NOISY = "TOO_NOISY"

MESSAGES = {
    OK: "Voice sample quality check passed.",
    INVALID_AUDIO: "The recording could not be read. Please record again.",
    NO_SPEECH: "No speech was detected. Please check the microphone and record again.",
    CLIPPED: "The recording is distorted (too loud). Please move slightly away from the microphone and try again.",
    TOO_LITTLE_SPEECH: "Too little speech was recorded. Please speak the whole sentence clearly for 3-5 seconds.",
    TOO_NOISY: "Voice quality is insufficient. Please move closer to the microphone, reduce background noise and try again.",
}


#: Short reasons for the retry hint after a submitted attempt ("Sentence 2: too much background noise ...").
REASONS = {
    INVALID_AUDIO: "the recording could not be read",
    NO_SPEECH: "no speech was detected",
    CLIPPED: "the recording is distorted (too loud)",
    TOO_LITTLE_SPEECH: "too little speech was recorded",
    TOO_NOISY: "too much background noise - move closer to the microphone",
}


@dataclass(frozen=True)
class VoiceQualityReport:
    verdict: str
    #: Scalars only (durations, levels, dB) - never audio. Rounded for display.
    metrics: dict = field(default_factory=dict)
    #: Development diagnostics (DEBUG_SCORES only): extra scalars, the configured limits and the failing check.
    diagnostics: dict | None = None

    @property
    def passed(self) -> bool:
        return self.verdict == OK

    @property
    def message(self) -> str:
        return MESSAGES[self.verdict]


class VoiceCaptureRejected(RuntimeError):
    """Raised instead of embedding a recording that failed the gate. `sentence` is 1-based (None: single recording).

    Deliberately not a ValueError: it must reach the route as a capture-quality failure, not be turned into a generic
    "could not process" error.
    """

    def __init__(self, report: VoiceQualityReport, sentence: int | None = None):
        where = f"voice sentence {sentence}: " if sentence else ""
        super().__init__(f"{where}{report.message}")
        self.report = report
        self.sentence = sentence


#: Which measured value decides each verdict, and how (for the diagnostics' "failed check" line).
_CHECKS = {
    NO_SPEECH: ("speech_seconds", "> 0"),
    CLIPPED: ("clipping_fraction", "<= voice_max_clipping_fraction"),
    TOO_LITTLE_SPEECH: ("speech_seconds", ">= voice_min_speech_seconds"),
    TOO_NOISY: ("estimated_snr_db", ">= voice_min_estimated_snr_db"),
}


def _round(value):
    return round(value, 3) if isinstance(value, float) and np.isfinite(value) else (None if isinstance(value, float) else value)


def assess(raw_input: tuple[np.ndarray, int], settings: Settings, diagnostics: bool | None = None) -> VoiceQualityReport:
    """The gate verdict for one decoded recording `(waveform, sample_rate)`. With diagnostics (default: when the server
    runs with DEBUG_SCORES) the report also carries explanatory scalars and the limits - the verdict is unchanged."""
    from preprocessing.voice import VoicePreprocessor

    waveform, sample_rate = raw_input
    samples = np.asarray(waveform)
    if samples.size == 0 or not np.all(np.isfinite(samples)) or sample_rate <= 0:
        return VoiceQualityReport(INVALID_AUDIO)
    m = VoicePreprocessor().quality_metrics(samples, sample_rate)
    shown = {k: (round(v, 3) if np.isfinite(v) else None) for k, v in m.items()}
    if m["peak"] == 0.0 or m["speech_seconds"] == 0.0:
        verdict = NO_SPEECH
    elif m["clipping_fraction"] > settings.voice_max_clipping_fraction:
        verdict = CLIPPED
    elif m["speech_seconds"] < settings.voice_min_speech_seconds:
        verdict = TOO_LITTLE_SPEECH
    elif m["estimated_snr_db"] < settings.voice_min_estimated_snr_db:
        verdict = TOO_NOISY
    else:
        verdict = OK
    report_diagnostics = None
    if diagnostics if diagnostics is not None else settings.debug_scores:
        extra = VoicePreprocessor().diagnostic_metrics(samples, sample_rate)
        failed = _CHECKS.get(verdict)
        report_diagnostics = {
            **{k: _round(float(v)) if isinstance(v, (float, np.floating)) else v for k, v in extra.items()},
            "limits": {
                "min_speech_seconds": settings.voice_min_speech_seconds,
                "min_estimated_snr_db": settings.voice_min_estimated_snr_db,
                "max_clipping_fraction": settings.voice_max_clipping_fraction,
            },
            "failed_check": None if failed is None else {"metric": failed[0], "required": failed[1],
                                                         "measured": shown.get(failed[0])},
        }
        logger.info("VOICE-QUALITY-DEBUG verdict=%s metrics=%s diagnostics=%s", verdict, shown, report_diagnostics)
    return VoiceQualityReport(verdict, shown, report_diagnostics)


def require(raw_input: tuple[np.ndarray, int], settings: Settings, sentence: int | None = None) -> VoiceQualityReport:
    """`assess`, raising VoiceCaptureRejected unless the recording passed."""
    report = assess(raw_input, settings)
    if not report.passed:
        raise VoiceCaptureRejected(report, sentence)
    return report
