"""Template-set experiments (evaluation/template_set_experiments.py) must show the properties they claim."""

from __future__ import annotations

import pytest

from evaluation import template_set_experiments as tse


def test_diversity_templates_in_different_sets_are_unlinkable():
    report = tse.experiment_template_set_diversity(num_users=8)
    assert report["cross_set_similarity_mean"] == pytest.approx(0.5, abs=0.03)
    assert report["same_key_genuine_mean"] > 0.9  # contrast: same key + genuine recapture matches
    assert report["modalities_per_set"] == 2  # face + voice


def test_revocation_moves_all_modalities_and_the_revoked_set_no_longer_matches():
    report = tse.experiment_template_set_revocation(num_users=6)
    assert report["genuine_similarity_before_mean"] > 0.9 and report["genuine_similarity_after_mean"] > 0.9
    assert report["revoked_vs_new_similarity_mean"] == pytest.approx(0.5, abs=0.06)
    assert report["users_with_all_modalities_in_new_set"] == 6


def test_promotion_keeps_authenticating_without_ever_mixing_sets():
    report = tse.experiment_template_set_promotion(num_users=5)
    assert report["acceptance_rate"] == 1.0
    assert report["attempts_mixing_sets"] == 0


def test_exhaustion_is_refused_and_the_active_set_survives():
    report = tse.experiment_template_set_exhaustion()
    assert report["successful_revocations"] == 3
    assert report["exhaustion_refused_http_409"] is True
    assert report["active_set_still_authenticates"] is True and report["final_active_set_version"] == 4


def test_run_all_writes_the_four_csv_reports(tmp_path):
    tse.run_all(tmp_path)
    assert {p.name for p in tmp_path.glob("*.csv")} == {
        "template_set_diversity.csv", "template_set_revocation.csv", "template_set_promotion.csv", "template_set_exhaustion.csv",
    }
