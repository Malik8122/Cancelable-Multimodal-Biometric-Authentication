# Baseline threshold findings

This report covers the teacher-requested thresholds, the experimental analysis of them, and what is (not) selected.
All data is **REAL DATA**: held-out LFW (face) and VoxCeleb1 test speakers (voice). Scores come from the deployed
256-bit protected-template pipeline.

## 1. Teacher-requested baseline (preserved, not optimized)

| Modality | Rule | Threshold | Provenance |
|---|---|---|---|
| Face | estimated cosine ≥ t | 0.80 | `Settings.face_cosine_threshold`; `THRESHOLD_SOURCE = TEACHER_REQUESTED_BASELINE` |
| Voice | estimated Euclidean distance ≤ t | 0.75 | `Settings.voice_euclidean_threshold`; same |

These thresholds were set as project requirements. They were not selected from genuine/impostor data, and no
optimality is claimed for them.

## 2. The 81.5 % / 85.3 % genuine-rejection findings: verified

| Finding | Value | Labelled data? | Protocol and source |
|---|---|---|---|
| Face FRR at estimated cosine ≥ 0.80 | **81.49 %** (FAR 0.0025 %) | yes: 4,523 genuine / 80,900 impostor comparisons | enrollment (first image) vs probe, 1,618 held-out LFW identities (2–19 images each, disjoint from fine-tuning), 50 impostor probes per identity, seed 20260925 (`evaluation/ieee/common.py::SEED`), each probe templated under the target's key. `raw_vs_protected_metrics.csv` (face, protected_template, at_operating_FRR = 0.814946), `threshold_sensitivity.csv` |
| Face-only FRR at 0.80, chimeric users | 80.35 % ± 5.79 % | yes: 1,440 genuine / 33,120 impostor attempts | 20 pairings × 24 virtual users, 3 attempts each. `fusion_policy_metrics.csv` (face_only) |
| Face + voice ALL_REQUIRED FRR | **85.35 %** ± 5.17 % (FAR 0) | yes: same chimeric protocol | `fusion_policy_metrics.csv` (ALL_REQUIRED(face+voice)) |
| Voice FRR at estimated distance ≤ 0.75 | 23.80 % (FAR 0.26 %) | yes: 727 genuine / 17,273 impostor | `raw_vs_protected_metrics.csv` |

The two chimeric rows were re-run on 2026-09-27 after fingerprint was removed from the chimeric sampler (the face
pairings changed); they were 79.86 % and 84.86 % before.

The face FRR is **not caused by template protection**. The exact raw cosine ≥ 0.80 rejects 83.1 % of genuine pairs,
even more than the protected estimate (81.5 %). The model's genuine cosines on unconstrained LFW mostly lie below
0.80. Genuine estimated cosine has mean 0.630 and median 0.677; the exact raw cosine has mean 0.632 and median 0.680.
The threshold sits far on the secure side of the EER point, which is at an estimated cosine of 0.33.

Both findings are kept as baseline results. They are not hidden, and the thresholds were not changed.

## 3. Experimental threshold analysis (does another operating point trade better?)

`evaluation/ieee/threshold_analysis.py` sweeps face from 0.60 to 0.90 and voice from 0.50 to 1.00, in steps of 0.01.
The results are in `threshold_sensitivity.csv`, `threshold_far_frr_tar.png` and `threshold_roc_det_dev_test.png`.

- **No test-set tuning.** Identities are split 50/50 (seed 20260925) into DEVELOPMENT and TEST. Comparisons across the two
  splits are dropped.
- **Face split:** 809 / 809 identities; 2,360 / 20,856 dev and 2,163 / 19,659 test genuine/impostor comparisons.
- **Voice split:** 12 / 12 speakers; 374 / 4,246 dev and 353 / 4,015 test comparisons.
- **Selection:** candidate points are chosen on DEVELOPMENT and each is evaluated **once** on TEST
  (`threshold_operating_points.csv`).

| Modality | Operating point | Selected on | Threshold | TEST FAR | TEST FRR | TEST TAR |
|---|---|---|---|---|---|---|
| Face | teacher baseline | — | 0.800 | 0.000 % | 80.1 % | 19.9 % |
| Face | development EER point | dev | 0.328 | 8.62 % | 8.46 % | 91.5 % |
| Face | development FAR ≤ 1 % | dev | 0.533 | 0.83 % | 22.9 % | 77.1 % |
| Face | development FAR ≤ 0.1 % | dev | 0.677 | 0.066 % | 46.9 % | 53.1 % |
| Voice | teacher baseline | — | 0.750 | 0.000 % | 32.3 % | 67.7 % |
| Voice | development EER point | dev | 0.980 | 1.12 % | 4.53 % | 95.5 % |
| Voice | development FAR ≤ 1 % | dev | 0.805 | 0.025 % | 21.5 % | 78.5 % |
| Voice | development FAR ≤ 0.1 % | dev | 0.702 | 0.000 % | 39.9 % | 60.1 % |

What the table shows:

- **Face.** Lowering the threshold to an FAR target chosen on development cuts FRR a lot. With FAR ≤ 1 %, FRR falls
  from 80 % to 23 %, and development and test agree closely. The cost is roughly a 300-fold higher FAR (0.0025 % → 0.8 %).
- **Voice.** Development and test **disagree**. At the teacher threshold, FRR is 15.8 % on development and 32.3 % on
  test, because each split has only 12 speakers. Any voice operating point is therefore statistically weak on this
  corpus, and its FRR should be quoted with that caveat.
- EER on the full evaluation set (the full-set rows of `threshold_sensitivity.csv`): face 8.50 %, voice 2.32 %.

## 4. Final selected operating point

**None. The deployed thresholds are unchanged** (face 0.80, voice 0.75; `THRESHOLD_SOURCE = TEACHER_REQUESTED_BASELINE`).

Choosing an operating point is a policy decision about acceptable FAR for a critical-infrastructure door. It needs:

1. an agreed FAR target;
2. real enrolled users captured in the deployment conditions (`evaluation/REAL_USER_PROTOCOL.md`), because LFW and
   VoxCeleb are not the deployment population;
3. a validation set that is separate from the reporting set.

If a new threshold is adopted, record it as `THRESHOLD_SOURCE = EXPERIMENTALLY_SELECTED`, together with its
development set, FAR target and held-out result, and keep this baseline section unchanged.
