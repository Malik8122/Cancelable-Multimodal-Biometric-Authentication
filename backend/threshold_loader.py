"""Per-modality authentication threshold, loaded from real calibration data.

Replaces the old behavior of comparing every modality's protected-template
Hamming score against one hardcoded `Settings.match_threshold=0.9` - see
`evaluation/threshold_calibration.py` for how `evaluation/results/<modality>_threshold.json`
is produced from real genuine/impostor protected-template scores.

Fails gracefully, per the spec: a modality with no calibration file yet
(e.g. before the calibration script has been run for a given
checkpoint) falls back to `Settings.match_threshold`, logging a clear
warning rather than raising - authentication must keep working even for an
uncalibrated modality.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from evaluation.threshold_calibration import load_threshold_report

logger = logging.getLogger("backend.threshold_loader")

#: `evaluation/results/`, resolved relative to this file rather than the
#: process's current working directory, so it works the same whether uvicorn
#: is started from the repo root or elsewhere (matches backend/api/metrics.py's
#: `_RESULTS_DIR` convention).
RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"


@lru_cache
def _cached_threshold(modality: str, fallback: float) -> float:
    report = load_threshold_report(modality, results_dir=RESULTS_DIR)
    if report is None:
        logger.warning(
            "No calibration file for modality=%s (expected %s) - falling back to the "
            "uncalibrated default threshold=%.2f. Run evaluation/threshold_calibration.py "
            "against real embeddings for this modality to replace this.",
            modality,
            RESULTS_DIR / f"{modality}_threshold.json",
            fallback,
        )
        return fallback
    return float(report["threshold"])


def get_modality_threshold(modality: str, fallback: float) -> float:
    """The threshold `template_protection.matcher.accept` should use for `modality`.

    `fallback` is `Settings.match_threshold` - callers always pass it
    explicitly (rather than this module importing `backend.config` itself)
    so this stays a pure "given a fallback, resolve the real value" function,
    easy to unit test without needing `MASTER_SECRET` set.
    """
    return _cached_threshold(modality, fallback)


def is_calibrated(modality: str) -> bool:
    """Whether a real calibration file exists for `modality` - used by
    `GET /system/health`'s `thresholds_loaded` field."""
    return load_threshold_report(modality, results_dir=RESULTS_DIR) is not None


def clear_cache() -> None:
    """Test hook - mirrors `backend.config.get_settings.cache_clear()`."""
    _cached_threshold.cache_clear()
