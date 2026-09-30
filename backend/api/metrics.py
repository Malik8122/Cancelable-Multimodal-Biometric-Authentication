"""GET /metrics/{modality}: real, already-computed evaluation numbers - never invented here.

Reads `evaluation/results/<modality>_metrics.csv`, the same artifact
the training/evaluation scripts (e.g. `evaluation/voice_metrics.py`) write from
real Kaggle GPU runs. If a modality's file doesn't exist (or a field within
it was never computed), that's returned as missing/absent rather than
fabricated - `available: false` for a whole modality, or a field simply not
present in `metrics` otherwise. This module does zero inference and touches
no model checkpoints; it just formats a CSV as JSON.
"""

from __future__ import annotations

import csv
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.services import _SUPPORTED_MODALITIES
from evaluation.threshold_calibration import load_threshold_report

router = APIRouter()

_RESULTS_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "results"


class ModalityMetricsResponse(BaseModel):
    modality: str
    available: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    #: Whether real protected-template calibration exists for this modality
    #: (evaluation/threshold_calibration.py) - distinct from `available`,
    #: which is about the raw-embedding `metrics` CSV.
    calibrated: bool = False


def _read_metrics_csv(modality: str) -> dict[str, float] | None:
    csv_path = _RESULTS_DIR / f"{modality}_metrics.csv"
    if not csv_path.exists():
        return None

    metrics: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row["metric"]
            if key == "modality":
                continue
            metrics[key] = float(row["value"])
    return metrics


def _calibrated_fields(modality: str) -> dict[str, float]:
    """Real protected-template calibration numbers, under `calibrated_*`
    keys so they can never be confused with the raw-embedding `far`/`frr`/
    `eer`/`auc` above - those are two different score spaces (see
    evaluation/threshold_calibration.py's module docstring), and merging
    them under the same key names would silently conflate the two."""
    report = load_threshold_report(modality, results_dir=_RESULTS_DIR)
    if report is None:
        return {}
    # A threshold set by an operator (see evaluation/results/face_threshold.json) has no measured error rates; report only
    # the fields that were actually measured rather than inventing numbers for the rest.
    fields = {"calibrated_threshold": float(report["threshold"])}
    for source, target in (("far", "calibrated_far"), ("frr", "calibrated_frr"), ("eer", "calibrated_eer"), ("auc", "calibrated_auc")):
        if report.get(source) is not None:
            fields[target] = float(report[source])
    return fields


@router.get("/metrics/{modality}", response_model=ModalityMetricsResponse)
def get_metrics(modality: str) -> ModalityMetricsResponse:
    if modality not in _SUPPORTED_MODALITIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported modality {modality!r}; expected one of {_SUPPORTED_MODALITIES}",
        )

    csv_metrics = _read_metrics_csv(modality)
    calibrated_fields = _calibrated_fields(modality)
    metrics = {**(csv_metrics or {}), **calibrated_fields}

    return ModalityMetricsResponse(
        modality=modality,
        available=csv_metrics is not None,
        metrics=metrics,
        calibrated=bool(calibrated_fields),
    )
