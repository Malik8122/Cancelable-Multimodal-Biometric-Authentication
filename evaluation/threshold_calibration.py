"""Per-modality threshold calibration on **protected-template** Hamming scores.

The audit finding this fixes: `backend/config.py::Settings.match_threshold`
was a single hardcoded 0.9 applied to every modality, never validated against
the score space authentication actually compares (protected-template Hamming
similarity, via `template_protection.matcher.compare`) - only raw-embedding
cosine similarity had ever been measured (e.g.
`evaluation/voice_metrics.py`). `evaluation/privacy_metrics.py`'s
`experiment_protected_far_frr` already computed a single EER point in the
right score space, but only from inside the test suite, never as a saved
report - this module generalizes that into a full threshold sweep and the
loader the backend now actually reads (see `backend/threshold_loader.py`).

Two outputs per modality, both derived from the same sweep so they can never
disagree with each other:

- `evaluation/results/<modality>_threshold.json` - the single EER-optimal
  operating point, in the exact shape `backend/threshold_loader.py` loads.
- `evaluation/results/<modality>_protected_metrics.csv` - the full sweep
  (threshold, accuracy, far, frr, eer, tp, fp, tn, fn), one row per
  threshold from 0.50 to 1.00.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from evaluation.metrics import accuracy_at_threshold, compute_eer, compute_far_frr, confusion_matrix_at_threshold
from evaluation.roc import compute_roc
from template_protection.biohash import DEFAULT_OUTPUT_BITS, generate_template
from template_protection.hkdf_keys import derive_key
from template_protection.matcher import compare

DEFAULT_THRESHOLD_START = 0.50
DEFAULT_THRESHOLD_STOP = 1.00
DEFAULT_THRESHOLD_STEP = 0.01


def sweep_thresholds(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    start: float = DEFAULT_THRESHOLD_START,
    stop: float = DEFAULT_THRESHOLD_STOP,
    step: float = DEFAULT_THRESHOLD_STEP,
) -> list[dict]:
    """One row per threshold: accuracy/FAR/FRR plus the raw TP/FP/TN/FN counts.

    `confusion_matrix_at_threshold` labels impostor=0/genuine=1, so its 2x2
    result is `[[TN, FP], [FN, TP]]` - unpacked explicitly here rather than
    trusting callers to remember that ordering.
    """
    thresholds = np.arange(start, stop + step / 2, step)
    rows = []
    for threshold in thresholds:
        threshold = float(threshold)
        far, frr = compute_far_frr(genuine_scores, impostor_scores, threshold)
        accuracy = accuracy_at_threshold(genuine_scores, impostor_scores, threshold)
        (tn, fp), (fn, tp) = confusion_matrix_at_threshold(genuine_scores, impostor_scores, threshold)
        rows.append(
            {
                "threshold": round(threshold, 4),
                "accuracy": accuracy,
                "far": far,
                "frr": frr,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )
    return rows


def _protected_genuine_impostor_scores(
    embeddings: list[np.ndarray],
    labels: list[str],
    master_secret: bytes | str,
    application_id: str,
    modality: str,
    output_bits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Real protected-template pairwise Hamming scores - mirrors
    `evaluation/privacy_metrics.py::experiment_protected_far_frr`'s pairing
    logic exactly, factored out here so this module and that one can't drift
    apart on what "genuine"/"impostor" means for protected templates."""
    keys = {
        label: derive_key(master_secret, application_id=application_id, user_id=label, modality=modality)
        for label in set(labels)
    }
    templates = [generate_template(embedding, keys[label], output_bits=output_bits) for embedding, label in zip(embeddings, labels)]

    genuine, impostor = [], []
    for (template_a, label_a), (template_b, label_b) in itertools.combinations(zip(templates, labels), 2):
        score = compare(template_a, template_b, metric="hamming")
        (genuine if label_a == label_b else impostor).append(score)
    return np.array(genuine), np.array(impostor)


def calibrate_modality_threshold(
    embeddings: list[np.ndarray],
    labels: list[str],
    master_secret: bytes | str,
    application_id: str,
    modality: str,
    output_bits: int = DEFAULT_OUTPUT_BITS,
    start: float = DEFAULT_THRESHOLD_START,
    stop: float = DEFAULT_THRESHOLD_STOP,
    step: float = DEFAULT_THRESHOLD_STEP,
) -> dict:
    """Full calibration report for one modality: sweep + EER-optimal pick + AUC.

    Requires real, labeled embeddings (multiple samples per identity, several
    identities) - the same input shape `evaluation/experiments.py` and
    `evaluation/privacy_metrics.py` already take. Calibrating on synthetic
    placeholder data (e.g. the offline test fixtures) produces a
    mathematically valid but biometrically meaningless number; this function
    doesn't know or care where `embeddings` came from, so that judgment call
    stays with the caller/script, not hidden in here.
    """
    genuine_scores, impostor_scores = _protected_genuine_impostor_scores(
        embeddings, labels, master_secret, application_id, modality, output_bits
    )
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        raise ValueError(
            f"Need at least one genuine pair and one impostor pair to calibrate {modality}; "
            "make sure the sample set has multiple embeddings per identity and multiple identities."
        )

    sweep = sweep_thresholds(genuine_scores, impostor_scores, start=start, stop=stop, step=step)
    eer, eer_threshold = compute_eer(genuine_scores, impostor_scores)
    roc = compute_roc(genuine_scores, impostor_scores)

    # The EER-optimal *grid* threshold (smallest |FAR-FRR| among the swept
    # values) rather than `eer_threshold` directly - `compute_eer` searches
    # over the actual score values, which may fall between two swept grid
    # points; picking from `sweep` guarantees the selected threshold is one
    # of the rows the CSV also reports, so the JSON and CSV are consistent.
    best_row = min(sweep, key=lambda row: abs(row["far"] - row["frr"]))

    return {
        "modality": modality,
        "threshold": best_row["threshold"],
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far": best_row["far"],
        "frr": best_row["frr"],
        "accuracy": best_row["accuracy"],
        "auc": roc["auc"],
        "num_genuine_pairs": len(genuine_scores),
        "num_impostor_pairs": len(impostor_scores),
        "output_bits": output_bits,
        "sweep": sweep,
    }


def save_threshold_json(report: dict, path: str | Path) -> None:
    """Write the Bug-1 summary shape: `{threshold, eer, far, frr, auc}` (plus
    a few extra provenance fields `backend/threshold_loader.py` ignores but
    that are useful for a human reading the file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "modality": report["modality"],
        "threshold": report["threshold"],
        "eer": report["eer"],
        "far": report["far"],
        "frr": report["frr"],
        "auc": report["auc"],
        "accuracy": report["accuracy"],
        "num_genuine_pairs": report["num_genuine_pairs"],
        "num_impostor_pairs": report["num_impostor_pairs"],
        "output_bits": report["output_bits"],
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_protected_metrics_csv(report: dict, path: str | Path) -> None:
    """Write the Bug-2 full sweep: one row per threshold with accuracy/FAR/FRR/TP/FP/TN/FN.

    `eer` is a single summary statistic (not threshold-dependent), so it's
    repeated on every row rather than only on the EER-optimal one - the
    columns the bug report asked for are per-row, and a reader shouldn't have
    to cross-reference the JSON file to find which row "the" EER belongs to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["threshold", "accuracy", "far", "frr", "eer", "tp", "fp", "tn", "fn"])
        for row in report["sweep"]:
            writer.writerow(
                [row["threshold"], row["accuracy"], row["far"], row["frr"], report["eer"], row["tp"], row["fp"], row["tn"], row["fn"]]
            )


def load_threshold_report(modality: str, results_dir: str | Path = "evaluation/results") -> dict | None:
    """Read `<modality>_threshold.json` back; `None` if it doesn't exist yet.

    This is the read side `backend/threshold_loader.py` calls - kept here
    (not duplicated in `backend/`) so the JSON shape is defined and read in
    exactly one place.
    """
    path = Path(results_dir) / f"{modality}_threshold.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
