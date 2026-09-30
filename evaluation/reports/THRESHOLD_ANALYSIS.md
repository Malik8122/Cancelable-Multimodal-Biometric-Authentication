# Threshold operating-point analysis

Scope: the deployed decision on 256-bit protected templates, meaning Hamming similarity turned into a calibrated
estimate and then compared with a threshold. Evidence: **REAL DATA**. Nothing here changes a configured threshold.

**Scripts**

- `evaluation/ieee/threshold_analysis.py`: face and voice, full set plus identity-disjoint development/test splits.
- `evaluation/ieee/experiments.py part6`: the wider sweep for all modalities.

**Results**

- `evaluation/results/threshold_sensitivity.csv`: face 0.60–0.90 and voice 0.50–1.00 in steps of 0.01. Each row gives
  FAR, FRR, TAR, accuracy, precision, recall, F1 and the subset EER, for the protected and raw scores, over the
  subsets all / development / test.
- `evaluation/results/threshold_operating_points.csv`: operating points selected on development, evaluated once on test.
- `evaluation/results/threshold_sweep.csv` and `threshold_sweep_eer.csv`: face 0.20–0.95, voice 0.50–1.20.

**Figures** (in `evaluation/figures/`)

| Figure | Shows |
|---|---|
| `threshold_far_frr_tar.png` | FAR, FRR and TAR vs threshold |
| `threshold_roc_det_dev_test.png` | ROC and DET for development and test |
| `fig16_threshold_sensitivity.png` | FAR/FRR, protected vs raw |
| `fig14_roc_protected_templates.png`, `fig15_det_curves.png` | ROC and DET, all modalities |
| `fig25_accuracy_f1_vs_threshold.png` | accuracy and F1 vs threshold |

Accuracy, precision and F1 depend on the genuine:impostor ratio of the protocol (about 1:18 for face), so compare
modalities by FAR, FRR and EER.

## Face (estimated cosine ≥ t)

| | Value |
|---|---|
| Current threshold | **0.80** (TEACHER_REQUESTED_BASELINE) |
| At 0.80, full set | FAR **0.0025 %**, FRR **81.5 %**, TAR 18.5 % (4,523 genuine / 80,900 impostor) |
| EER, full set | **8.50 %** at estimated cosine 0.328 (raw-cosine EER 7.12 %) |
| ROC-AUC | 0.9641 protected, 0.9682 raw |
| Sweep 0.60 → 0.90 | FRR rises from 34.2 % to 97.6 %; FAR falls from 0.28 % to 0 (never above 0.28 %) |

**Trade-off.** The 0.80 threshold is a near-zero-FAR, very high-FRR operating point. Choosing FAR ≤ 1 % on development
(threshold 0.533) gives FAR 0.83 % and FRR 22.9 % on test. Choosing FAR ≤ 0.1 % (threshold 0.677) gives FAR 0.066 %
and FRR 46.9 %. Most of the FRR comes from the model's genuine-score distribution on unconstrained LFW (genuine median
cosine ≈ 0.68), not from template protection.

## Voice (estimated Euclidean distance ≤ t)

| | Value |
|---|---|
| Current threshold | **0.75** (TEACHER_REQUESTED_BASELINE), equivalent to cosine 0.719 |
| At 0.75, full set | FAR **0.26 %**, FRR **23.8 %**, TAR 76.2 % (727 genuine / 17,273 impostor) |
| EER, full set | **2.32 %** at estimated distance 1.005 (raw-distance EER 2.21 %) |
| ROC-AUC | 0.9967 protected, 0.9980 raw |
| Sweep 0.50 → 1.00 | FRR falls from 78.4 % to 3.0 %; FAR rises from 0 to 2.02 % |

**Trade-off.** Voice separates much better than face (EER 2.3 % vs 8.5 %), but the corpus has only 24 test speakers.
The development and test splits have 12 speakers each, and at the same threshold their FRR differs by a factor of two
(15.8 % vs 32.3 %), so any voice operating point carries large uncertainty. The FAR ≤ 1 % point chosen on development
(threshold 0.805) gives FRR 21.5 % on test, with FAR 0.025 %.

## Fusion (face + voice), at the baseline thresholds

These come from the chimeric protocol (`fusion_policy_metrics.csv`, 20 pairings × 24 virtual users):

| Policy | FAR | FRR |
|---|---|---|
| ALL_REQUIRED | 0 | **85.3 %** |
| WEIGHTED | 0 | 59.6 % |
| face only | — | 80.3 % |
| voice only | — | 24.4 % |

Score-level mean(face, voice) reaches EER 1.54 % and AUC 0.9988 (`fusion_score_level_eer.csv`; re-run 2026-09-27 after
fingerprint was removed from the chimeric sampler, which changed the face pairings). Fusing scores
separates well; the high FRR comes from the per-modality thresholds.

## Selection protocol

- An operating point may be proposed only from the **development** split, and it is then evaluated once on the
  **test** split (`threshold_operating_points.csv`).
- The teacher baseline is always reported next to any proposed point.
- The final choice is **not made** here. It needs a FAR target and real deployment users; see
  `BASELINE_THRESHOLD_FINDINGS.md` §4.
