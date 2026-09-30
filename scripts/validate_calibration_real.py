"""Validate the BioHash metric calibration on REAL embedding pairs (Part 3 of the IEEE evaluation).

For every enrollment-vs-probe pair of the held-out evaluation sets (LFW faces, VoxCeleb-subset voices): true cosine of the two embeddings -> both templated under the SAME key -> estimated cosine from the
Hamming similarity (template_protection/metric_estimation.py) -> RMSE, MAE, bias, SD, Pearson r, R^2.
Prerequisite: python -m evaluation.ieee.extract_embeddings
Outputs: evaluation/results/calibration_real_validation.csv, calibration_real_binned.csv, figures fig21-fig23.
Run: python -m scripts.validate_calibration_real
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.ieee.common import timed  # noqa: E402
from evaluation.ieee.experiments import part3_calibration_real  # noqa: E402

if __name__ == "__main__":
    with timed("part3_calibration_real"):
        part3_calibration_real()
