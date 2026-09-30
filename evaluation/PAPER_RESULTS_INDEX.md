# Paper results index

This is the single source of truth for the Results section. Every number that may appear in the paper is listed
here with its evidence type, protocol and the file and script it comes from. A number that is not listed here should
not be used.

**Evidence types**

| Code | Meaning |
|---|---|
| [A] | REAL DATA |
| [B] | SYNTHETIC |
| [C] | SOFTWARE TEST |
| [D] | CONFIGURATION |
| [E] | DERIVED |
| [F] | NOT AVAILABLE |

**Common protocol (P-FACE, P-VOICE):** enrollment-vs-probe.

- The first sample of an identity is the reference; its other samples are the genuine probes.
- For face, 50 random impostor probes are drawn per identity (seed 20260925, `evaluation/ieee/common.py::SEED`); for voice, all impostors are used.
- Each probe is templated under the **target's** key.
- The template is 256-bit BioHash, compared by Hamming similarity (`evaluation/ieee/protected.py`, asserted bit-identical to production).

**Datasets**

- Face: held-out LFW. These are identities with 2–19 images, disjoint from the 62 fine-tuning identities: 1,618
  references, 4,523 genuine and 80,900 impostor comparisons.
- Voice: VoxCeleb1 test speakers: 24 speakers, 727 genuine and 17,273 impostor comparisons.

"Reproducible" means the listed script regenerates the number from the committed code and the public datasets.

## 1. Configuration

| Metric | Value | Type | Source | Script | Reproducible | Notes |
|---|---|---|---|---|---|---|
| Embedding dimensions | face 512, voice 192 | [D] | `models/*/inference.py` | `tests/test_modality_dimensions.py` | yes | |
| Template length | 256 bits (32 bytes) | [D] | `backend/config.py::template_bits` | — | yes | |
| Template sets per enrollment | 4 (1 ACTIVE, 3 STANDBY) | [D] | `template_pool_size` | — | yes | |
| Face threshold | estimated cosine ≥ 0.80 | [D] | `face_cosine_threshold` | — | yes | TEACHER_REQUESTED_BASELINE, not optimized |
| Voice threshold | estimated Euclidean distance ≤ 0.75 | [D] | `voice_euclidean_threshold` | — | yes | TEACHER_REQUESTED_BASELINE |
| Default fusion | ALL_REQUIRED over the submitted modalities | [D] | `fusion/config.py` | — | yes | |

## 2. Protected-template performance (deployed pipeline)

| Metric | Value | Type | Dataset / protocol | Source | Script | Reproducible |
|---|---|---|---|---|---|---|
| Face EER (protected) | 8.50 % (95 % CI 7.77–8.93) | [A] | LFW held-out, P-FACE | `protected_vs_raw.csv` | `python -m evaluation.scripts.protected_vs_raw` | yes |
| Face ROC-AUC (protected) | 0.9641 | [A] | same | same | same | yes |
| Face at 0.80 | FAR 0.0025 %, FRR 81.5 %, TAR 18.5 % | [A] | same | `raw_vs_protected_metrics.csv`, `threshold_sensitivity.csv` | `evaluation.ieee.experiments 4`, `evaluation.ieee.threshold_analysis` | yes |
| Voice EER (protected) | 2.32 % (95 % CI 1.95–3.03) | [A] | VoxCeleb1 test, P-VOICE | `protected_vs_raw.csv` | same | yes |
| Voice ROC-AUC (protected) | 0.9967 | [A] | same | same | same | yes |
| Voice at 0.75 | FAR 0.26 %, FRR 23.8 %, TAR 76.2 % | [A] | same | `raw_vs_protected_metrics.csv` | same | yes |

## 3. Raw vs protected (identical pairs)

| Metric | Value | Type | Source | Script | Notes |
|---|---|---|---|---|---|
| Face EER raw → protected | 7.12 % → 8.50 % (Δ +1.38 pp) | [A] | `protected_vs_raw.csv` | `evaluation.scripts.protected_vs_raw` | raw is **not** system performance |
| Face AUC raw → protected | 0.9682 → 0.9641 | [A] | same | same | |
| Voice EER raw → protected | 2.21 % → 2.32 % (Δ +0.12 pp) | [A] | same | same | |
| Voice AUC raw → protected | 0.9980 → 0.9967 | [A] | same | same | |

## 4. Thresholds (`evaluation/reports/THRESHOLD_ANALYSIS.md`, `BASELINE_THRESHOLD_FINDINGS.md`)

| Metric | Value | Type | Source | Notes |
|---|---|---|---|---|
| Face sweep 0.60–0.90 (step 0.01) | FRR 34.2 % → 97.6 %, FAR 0.28 % → 0 | [A] | `threshold_sensitivity.csv` | |
| Voice sweep 0.50–1.00 (step 0.01) | FRR 78.4 % → 3.0 %, FAR 0 → 2.02 % | [A] | same | |
| Face, dev-selected FAR ≤ 1 % (t = 0.533), on TEST | FAR 0.83 %, FRR 22.9 % | [A] | `threshold_operating_points.csv` | identity-disjoint development/test split; **not adopted** |
| Voice, dev-selected FAR ≤ 1 % (t = 0.805), on TEST | FAR 0.025 %, FRR 21.5 % | [A] | same | 12 speakers per split, so unstable; **not adopted** |
| Final selected operating point | none (baseline kept) | [D] | `backend/config.py::threshold_source` | |

## 5. Fusion (chimeric users: LFW face × VoxCeleb voice, 20 pairings × 24 virtual users)

Re-run on 2026-09-27 after iris and fingerprint were removed: the chimeric sampler no longer draws fingerprint
identities, so the face pairings (and the face / fusion figures) changed slightly; voice-only is unchanged.

| Metric | Value | Type | Source | Script | Notes |
|---|---|---|---|---|---|
| Face only | FAR 0, FRR 80.3 % ± 5.8 % | [A] | `fusion_policy_metrics.csv` | `evaluation.ieee.experiments 14` | chimeric users assume the modalities are independent |
| Voice only | FAR 0.13 %, FRR 24.4 % ± 2.0 % | [A] | same | same | |
| Face + voice ALL_REQUIRED | FAR 0, FRR 85.3 % ± 5.2 % | [A] | same | same | the baseline finding |
| Face + voice WEIGHTED | FAR 0, FRR 59.6 % ± 6.0 % | [A] | same | same | |
| Score-level mean(face, voice) | EER 1.54 %, AUC 0.9988 | [A] | `fusion_score_level_eer.csv` | same | not a deployed policy |

## 6. Calibration

| Metric | Value | Type | Source | Script | Notes |
|---|---|---|---|---|---|
| Synthetic fit, face D=512 | R² 0.99986, RMSE 0.00142, MAE 0.00108; single-estimate SD 0.050 at cos 0.80 | [B] | `biohash_metric_calibration.json` (`fit_statistics`) | `scripts/calibrate_biohash_metric_mapping.py` | **SYNTHETIC CALIBRATION**, not accuracy |
| Synthetic fit, voice D=192 | R² 0.99990, RMSE 0.00119, MAE 0.00088; SD 0.047 | [B] | same | same | same |
| Real-embedding estimate error, face (all pairs) | RMSE 0.100, MAE 0.080, bias +0.006, R² 0.774 (n 85,423) | [A] | `calibration_real_validation.csv` | `scripts/validate_calibration_real.py` | genuine pairs: RMSE 0.072 |
| Real-embedding estimate error, voice (all pairs) | RMSE 0.093, MAE 0.074, bias +0.004, R² 0.827 (n 18,000) | [A] | same | same | genuine pairs: RMSE 0.046 |

## 7. Revocation, template sets and unlinkability

| Metric | Value | Type | Source | Script | Notes |
|---|---|---|---|---|---|
| Same face embedding under different set keys | Hamming 0.499 ± 0.032 (n 900) | [A] | `revocation_summary.csv` | `evaluation.ieee.experiments 7_9` | |
| Revoked template vs new active (face / voice) | 0.500 / 0.504; acceptance 0 / 0 | [A] | same | same | |
| Genuine before vs after 3 revocations (face) | 0.761 → 0.766 | [A] | same | same | performance is kept after revocation |
| Genuine before vs after 3 revocations (voice) | 0.811 → 0.804 | [A] | same | same | |
| Template-set lifecycle | 80/80 accepted, 3 revocations then HTTP 409 | [B] | `template_set_*.csv` | `evaluation.template_set_experiments` | synthetic embeddings |
| D↔_sys, face (bin 1/64) | 0.018 (null p95 0.013) | [A] | `unlinkability.csv` | `evaluation.ieee.experiments 8` | Gomez-Barrero et al. 2018; small but above the null floor |
| D↔_sys, voice (bin 1/64) | 0.020 (below the null p95) | [A] | same | same | not distinguishable from the null |

## 8. Latency (i5-1235U, 8.3 GB RAM, Windows 11, Python 3.13.1, torch 2.14.0+cpu, no GPU)

| Metric | Value (warm, 100 runs: mean / median / P95 / P99 ms) | Type | Source | Script |
|---|---|---|---|---|
| Face preprocessing (bbox) | 33.0 / 32.2 / 42.5 / 46.0 | [A] | `latency_benchmark.csv` | `scripts/benchmark_latency.py` |
| Face embedding | 38.0 / 39.3 / 44.1 / 45.0 | [A] | same | same |
| Voice preprocessing / embedding | 17.4 / 18.0 / 25.8 / 29.3 and 50.7 / 49.8 / 63.8 / 67.7 | [A] | same | same |
| BioHash face (512→256) | 43.5 / 42.4 / 50.5 / 63.9 | [A] | same | same |
| Hamming / decision / fusion | 0.006 / 0.09 / 0.004 (means) | [A] | same | same |
| DB lookup (2,000 users × 4 sets) | 2.8 / 2.5 / 4.5 / 5.0 | [A] | same | same |
| API face + voice (full request) | 778 / 757 / 845 / 1025 | [A] | same | same |
| Cold start (10 runs, mean) | face 6.0 s, voice 3.4 s | [A] | same | same |

## 8b. Face alignment (`evaluation/reports/FACE_ALIGNMENT_DECISION.md`)

All rows: [A] REAL DATA, held-out LFW, 1,618 identities, 4,523 genuine / 80,900 impostor pairs, identical pairs for
every system. Script: `python -m evaluation.scripts.evaluate_face_alignment`. The deployed pipeline is still bbox.

| Metric | Value | Source | Notes |
|---|---|---|---|
| Protected EER, bbox v2 (E) → aligned v2 (F) | 8.67 % → 7.18 % | `face_alignment_metrics.csv` | controlled pair (same data, split, seed, recipe) |
| Paired ΔEER, F − E | −1.49 pp (95 % CI −1.90 to −0.69), 0 of 300 resamples show no improvement | `face_alignment_paired_bootstrap.csv` | identity bootstrap |
| Raw EER, E → F | 6.99 % → 5.77 % | `face_alignment_metrics.csv` | raw is not system performance |
| TAR at FAR 0.1 % (protected), E → F | 0.585 → 0.632 | same | |
| FRR at 0.80, E → F | 81.2 % → 76.8 % | `face_threshold_comparison.csv` | teacher threshold unchanged |
| Ablation: deployed checkpoint + aligned input | EER 9.96 % (+1.46 pp vs deployed) | `face_alignment_metrics.csv`, bootstrap | **inference-time ablation**; not a final model |
| Eye-line angle, mean / p95 | 2.89° / 7.68° → 0.98° / 2.77° | `face_alignment_quality.csv` | 6,141 images |
| Landmark RMS to template, p95 | 30.9 → 12.1 px | same | |
| Alignment failures | 0 / 6,141 | same | |
| ALL_REQUIRED face+voice FRR, E → F | 83.3 % → 82.9 % | `face_alignment_fusion.csv` | chimeric users |
| 30° rotation EER, E → F | 13.0 % → 6.75 % | `face_alignment_robustness.csv` | 200 + 200 pairs, so ±2–3 pp |
| Preprocessing latency, warm mean, bbox → aligned | 47.2 → 46.7 ms (P99 80.3 → 73.2) | `face_alignment_latency.csv` | 100 warm / 10 cold runs, interleaved |
| Confounded v1 pair (sklearn slice) | D − C = +6.31 pp | bootstrap | **do not report as an alignment effect** |

## 8c. Figures

| Figure | Content | Type | Script |
|---|---|---|---|
| `evaluation/figures/fig03_roc_protected_templates.{png,pdf}` | ROC of protected-template scores: face (EER 8.16 %, AUC 0.9656), voice (2.75 %, 0.9970), face + voice score-level fusion (1.54 %, 0.9988) | [A] chimeric users, 1,440 genuine / 33,120 impostor attempts | `python -m evaluation.scripts.fig03_roc_protected_templates` (asserts the values equal `fusion_score_level_eer.csv`) |

The fusion curves use score-level mean fusion, which is **not** the deployed ALL_REQUIRED rule.

## 9. Software validation

| Metric | Value | Type | Source | Notes |
|---|---|---|---|---|
| Backend tests | 636 passed, 0 failed | [C] | `pytest` (2026-09-25) | software tests are **not** authentication trials |
| Frontend | typecheck 0 errors; lint 0 errors / 25 warnings; build OK | [C] | `tsc -b`, `oxlint`, `vite build` | |

## 10. Not available

| Item | Type |
|---|---|
| Real-user (participant) FAR / FRR | [F]: no data collected (`evaluation/REAL_USER_PROTOCOL.md`) |
| Historical per-epoch curves of the deployed face checkpoint | [F] |
| Completed voice training reproduction | [F]: interrupted after epoch 8 |
| Formal irreversibility proof | [F] |
