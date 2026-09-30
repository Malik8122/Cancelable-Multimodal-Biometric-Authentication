"""Committed evaluation artefacts must be reproducible by the code that claims to produce them."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "evaluation" / "results"


def _committed(name):
    with (RESULTS / name).open(encoding="utf-8") as h:
        return {r["field"]: r["value"] for r in csv.DictReader(h)}


@pytest.mark.parametrize(
    "fn_name, csv_name, fields",
    [
        ("experiment_template_set_revocation", "template_set_revocation.csv",
         ("genuine_similarity_before_mean", "genuine_similarity_after_mean", "revoked_vs_new_similarity_mean")),
        ("experiment_template_set_promotion", "template_set_promotion.csv",
         ("mean_similarity_active_set_1", "mean_similarity_active_set_4", "acceptance_rate")),
    ],
)
def test_template_set_experiments_still_report_hamming_similarity(fn_name, csv_name, fields):
    """These CSV columns are template Hamming similarities; `AuthenticationResult.score` is the fusion-scale
    estimate, so the experiments must read `hamming_similarity` (regression: they silently switched scales)."""
    from evaluation import template_set_experiments as tse

    rerun = getattr(tse, fn_name)()
    committed = _committed(csv_name)
    for field in fields:
        assert float(rerun[field]) == pytest.approx(float(committed[field]), abs=1e-12), field
