# IEEE Evaluation Package

Single source of truth for the paper. Every number here is generated from the files listed; evidence labels: REAL DATA, SYNTHETIC CALIBRATION, SOFTWARE VALIDATION, CONFIGURATION, DERIVED, FUTURE WORK (exactly one per metric).

## Experiments

| experiment | status | outputs | command |
|---|---|---|---|
| Part 1 Repository audit | done | evaluation/REPOSITORY_AUDIT.md | `python -m evaluation.ieee.audit` |
| Part 2 Metric verification | done | evaluation/metric_verification_report.md | `python -m evaluation.ieee.verification` |
| Part 3 Real-embedding calibration validation | done | calibration_real_validation.csv, calibration_real_binned.csv | `python -m scripts.validate_calibration_real` |
| Part 4 Raw vs protected | done | raw_vs_protected_metrics.csv | `python -m evaluation.ieee.experiments 4` |
| Part 5 Real-user harness | harness built + unit-tested; NOT RUN (no consented participants) | evaluation/real_user_evaluation.py | `python -m evaluation.real_user_evaluation --root <folder> --consent-confirmed` |
| Part 6 Threshold sweep | done | threshold_sweep.csv, threshold_sweep_eer.csv | `python -m evaluation.ieee.experiments 6` |
| Part 7 Template-set experiments | re-run (synthetic, matches committed) + real-embedding lifecycle | metric_verification.csv, revocation_*.csv | `python -m evaluation.ieee.experiments 7_9` |
| Part 8 Unlinkability | done | unlinkability.csv | `python -m evaluation.ieee.experiments 8` |
| Part 9 Revocability | done | revocation_per_attempt.csv, revocation_summary.csv | `python -m evaluation.ieee.experiments 7_9` |
| Part 10 Latency | done | latency_benchmark.csv, reports/latency_report.md | `python -m scripts.benchmark_latency` |
| Part 11 Memory | done | memory_benchmark.csv | `python -m scripts.benchmark_memory` |
| Part 12 Storage | done | storage_analysis.csv | `python -m evaluation.ieee.storage` |
| Part 13 Robustness | done (simulated degradations of real probes; pose / physical mic NOT AVAILABLE) | robustness.csv | `python -m evaluation.ieee.robustness` |
| Part 14 Fusion | done (chimeric users) | fusion_policy_metrics.csv, fusion_score_level_eer.csv | `python -m evaluation.ieee.experiments 14` |
| Part 15 Presentation attacks | framework only; IAPMR NOT AVAILABLE (no attack data); APCER/BPCER not applicable (no PAD) | pad_results.csv | `python -m evaluation.ieee.pad_evaluation` |
| Part 16 Security statistics | done | template_security.csv | `python -m evaluation.ieee.experiments 16` |
| Part 17 Software validation | done | software_validation.csv, reports/software_validation.md | `python -m evaluation.ieee.audit software` |
| Part 18 Figures | 26 figures | evaluation/figures/ | `python -m evaluation.ieee.figures (+ the scripts above)` |
| Part 19 Tables | 12 tables | evaluation/tables/ | `python -m evaluation.ieee.reports` |
| Part 20 Claims | done | evaluation/PAPER_CLAIMS.md | `python -m evaluation.ieee.reports` |

## Key findings (generated from the result files)

- **face** [REAL DATA]: EER 7.12% raw -> 8.50% protected (256-bit); at the deployed rule FAR 0.00%, FRR 81.49% (`raw_vs_protected_metrics.csv`).
- **voice** [REAL DATA]: EER 2.21% raw -> 2.32% protected (256-bit); at the deployed rule FAR 0.26%, FRR 23.80% (`raw_vs_protected_metrics.csv`).
- **Face threshold** [REAL DATA]: the protected-template EER point lies at estimated cosine 0.33; the teacher-requested 0.80 sits far into the low-FAR/high-FRR region for this face model (`threshold_sweep.csv`).
- **Voice threshold** [REAL DATA]: protected EER point at estimated distance 1.00 vs the configured 0.75.
- **ALL_REQUIRED(face+voice)** [REAL DATA, chimeric]: FAR 0.00%, FRR 85.35% (`fusion_policy_metrics.csv`).
- **WEIGHTED(face+voice)** [REAL DATA, chimeric]: FAR 0.00%, FRR 59.58% (`fusion_policy_metrics.csv`).
- **OR(face+voice)** [REAL DATA, chimeric]: FAR 0.13%, FRR 19.38% (`fusion_policy_metrics.csv`).
- **Score-level fusion** [REAL DATA, chimeric]: EER face 8.16%, voice 2.75%, mean(face,voice) 1.54% (`fusion_score_level_eer.csv`).
- **Calibration on real embeddings** [REAL DATA]: per-bin error SD matches the synthetic prediction (face [0.7,0.8): real 0.058 vs synthetic 0.055; face [0.8,0.9): real 0.046 vs synthetic 0.043) with near-zero mean error (`calibration_real_binned.csv`).

## Headline metrics (see evaluation/tables/ for full tables with sources)

# Embedding-model performance (raw embeddings)

| modality | score | n_genuine | n_impostor | EER | EER_95CI | ROC_AUC | TAR@FAR=1% | TAR@FAR=0.1% | operating_rule | FAR@op | FRR@op | F1@op | source | evidence_label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| face | raw_cosine | 4523 | 80900 | 7.12% | 6.54% - 7.62% | 0.9682 | 84.79% | 66.88% | raw_cosine >= 0.8 | 0.00% | 83.11% | 0.2890 | evaluation/results/raw_vs_protected_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | raw_distance | 727 | 17273 | 2.21% | 1.91% - 2.88% | 0.9980 | 94.91% | 79.92% | raw_distance <= 0.75 | 0.03% | 24.07% | 0.8591 | evaluation/results/raw_vs_protected_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |


# Protected-template (deployed) performance

| modality | score | n_genuine | n_impostor | EER | EER_95CI | ROC_AUC | TAR@FAR=1% | TAR@FAR=0.1% | operating_rule | FAR@op | FRR@op | F1@op | source | evidence_label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| face | estimated_cosine | 4523 | 80900 | 8.50% | 7.76% - 8.90% | 0.9641 | 77.47% | 53.42% | estimated_cosine >= 0.8 | 0.00% | 81.49% | 0.3122 | evaluation/results/raw_vs_protected_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | estimated_distance | 727 | 17273 | 2.32% | 1.94% - 3.03% | 0.9967 | 91.75% | 69.60% | estimated_distance <= 0.75 | 0.26% | 23.80% | 0.8356 | evaluation/results/raw_vs_protected_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |


# Calibration: synthetic fit and real-embedding validation

| modality | data | max_fit_residual_hamming | coefficients | RMSE_cosine | MAE_cosine | R2 | pearson_r | bias | source | evidence_label |
|---|---|---|---|---|---|---|---|---|---|---|
| face | synthetic pairs (19,600) | 0.00378 | 0.8817, 0.0274, -0.0214 | n/a | n/a | n/a | n/a | n/a | evaluation/results/biohash_metric_calibration.json (python -m scripts.calibrate_biohash_metric_mapping) | SYNTHETIC CALIBRATION |
| voice | synthetic pairs (19,600) | 0.00306 | 0.8855, 0.0191, -0.0178 | n/a | n/a | n/a | n/a | n/a | evaluation/results/biohash_metric_calibration.json (python -m scripts.calibrate_biohash_metric_mapping) | SYNTHETIC CALIBRATION |
| face | real pairs, all (n=85423) | n/a | n/a | 0.1004 | 0.0799 | 0.7735 | 0.8945 | 0.0057 | evaluation/results/calibration_real_validation.csv (python -m scripts.validate_calibration_real) | REAL DATA |
| face | real pairs, genuine (n=4523) | n/a | n/a | 0.0717 | 0.0554 | 0.8731 | 0.9391 | -0.0022 | evaluation/results/calibration_real_validation.csv (python -m scripts.validate_calibration_real) | REAL DATA |
| face | real pairs, in_calibrated_range (n=76916) | n/a | n/a | 0.0999 | 0.0795 | 0.7536 | 0.8852 | 0.0098 | evaluation/results/calibration_real_validation.csv (python -m scripts.validate_calibration_real) | REAL DATA |
| voice | real pairs, all (n=18000) | n/a | n/a | 0.0932 | 0.0736 | 0.8269 | 0.9175 | 0.0036 | evaluation/results/calibration_real_validation.csv (python -m scripts.validate_calibration_real) | REAL DATA |
| voice | real pairs, genuine (n=727) | n/a | n/a | 0.0463 | 0.0352 | 0.8435 | 0.9224 | 0.0040 | evaluation/results/calibration_real_validation.csv (python -m scripts.validate_calibration_real) | REAL DATA |
| voice | real pairs, in_calibrated_range (n=16305) | n/a | n/a | 0.0920 | 0.0727 | 0.8141 | 0.9118 | 0.0063 | evaluation/results/calibration_real_validation.csv (python -m scripts.validate_calibration_real) | REAL DATA |


# Fusion policies (chimeric virtual users from real data; mean ± SD over pairings)

| policy | modalities | FAR | FRR | accuracy | F1_pooled | TP/FN/FP/TN | protocol | source | evidence_label |
|---|---|---|---|---|---|---|---|---|---|
| face_only | face | 0.00% ± 0.00% | 80.35% ± 5.79% | 96.65% | 0.3285 | 283/1157/0/33120 | 20 chimeric pairings x 24 virtual users | evaluation/results/fusion_policy_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice_only | voice | 0.13% ± 0.08% | 24.38% ± 2.04% | 98.86% | 0.8465 | 1089/351/44/33076 | 20 chimeric pairings x 24 virtual users | evaluation/results/fusion_policy_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| ALL_REQUIRED(face+voice) | face+voice | 0.00% ± 0.00% | 85.35% ± 5.17% | 96.44% | 0.2556 | 211/1229/0/33120 | 20 chimeric pairings x 24 virtual users | evaluation/results/fusion_policy_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| WEIGHTED(face+voice) | face+voice | 0.00% ± 0.00% | 59.58% ± 5.98% | 97.52% | 0.5757 | 582/858/0/33120 | 20 chimeric pairings x 24 virtual users | evaluation/results/fusion_policy_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| OR(face+voice) | face+voice | 0.13% ± 0.08% | 19.38% ± 2.94% | 99.07% | 0.8779 | 1161/279/44/33076 | 20 chimeric pairings x 24 virtual users | evaluation/results/fusion_policy_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| MAJORITY(face+voice)=AND | face+voice | 0.00% ± 0.00% | 85.35% ± 5.17% | 96.44% | 0.2556 | 211/1229/0/33120 | 20 chimeric pairings x 24 virtual users | evaluation/results/fusion_policy_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |


## Figures

| file | label | caption |
|---|---|---|
| `evaluation/figures/fig01_system_architecture.png` | CONFIGURATION | System architecture of the implemented cancelable multimodal authentication system. |
| `evaluation/figures/fig02_enrollment_workflow.png` | CONFIGURATION | Enrollment workflow: name, guided face poses, two voice recordings, four keyed template sets. |
| `evaluation/figures/fig03_authentication_workflow.png` | CONFIGURATION | Authentication workflow; the stored template is compared only by Hamming similarity. |
| `evaluation/figures/fig04_face_pipeline.png` | CONFIGURATION | Face pipeline as implemented. |
| `evaluation/figures/fig05_voice_pipeline.png` | CONFIGURATION | Voice pipeline as implemented. |
| `evaluation/figures/fig07_biohash_generation.png` | CONFIGURATION | BioHash generation (template_protection/): three HKDF-SHA256 seeds drive projection, quantization and permutation. |
| `evaluation/figures/fig08_template_lifecycle.png` | CONFIGURATION | Template-set lifecycle: whole multimodal sets are revoked and promoted together. |
| `evaluation/figures/fig09_fusion_architecture.png` | CONFIGURATION | Score conversion and fusion: every modality enters fusion on one higher-is-better scale. |
| `evaluation/figures/fig10_calibration_curve.png` | SYNTHETIC CALIBRATION | Offline calibration on 19,600 synthetic pairs per modality through the real BioHash: mean ± SD and fitted cubic-in-angle model. |
| `evaluation/figures/fig11_calibration_residuals.png` | SYNTHETIC CALIBRATION | Synthetic calibration: model residuals and single-estimate standard deviation vs cosine. |
| `evaluation/figures/fig12_roc_face_raw_vs_protected.png` | REAL DATA | ROC of the face pipeline on real data: raw embedding cosine vs 256-bit protected-template Hamming similarity. |
| `evaluation/figures/fig13_roc_voice_raw_vs_protected.png` | REAL DATA | ROC of the voice pipeline on real data: raw embedding cosine vs 256-bit protected-template Hamming similarity. |
| `evaluation/figures/fig14_roc_protected_templates.png` | REAL DATA | ROC of all modalities, protected templates (solid) vs raw embeddings (dotted), real data. |
| `evaluation/figures/fig15_det_curves.png` | REAL DATA | DET curves (normal-deviate axes), protected (solid) vs raw (dotted), real data. |
| `evaluation/figures/fig16_threshold_sensitivity.png` | REAL DATA | FAR and FRR vs decision threshold (protected: solid; exact raw metric: dotted); dot = protected EER. |
| `evaluation/figures/fig17_latency_distribution.png` | REAL DATA | Warm latency per pipeline stage on the evaluation machine (median; whiskers min to p95). |
| `evaluation/figures/fig18_template_revocation_histogram.png` | REAL DATA | Revocation on real embeddings through the shipped template-set code: genuine similarity before/after revocation and the revoked template vs the new active template. |
| `evaluation/figures/fig19_unlinkability_histogram.png` | REAL DATA | Unlinkability (Gomez-Barrero et al. 2018): mated vs non-mated cross-key scores on real embeddings, local D<-> (line), D_sys and its permutation null floor (histogram estimate, bin 4/256). |
| `evaluation/figures/fig20_storage_comparison.png` | SOFTWARE VALIDATION | Template payload vs raw embedding size vs measured per-user database growth (SQLite, 1000 users). |
| `evaluation/figures/fig21_calibration_real_scatter.png` | REAL DATA | Estimated cosine (from the 256-bit template Hamming similarity) vs true embedding cosine on real pairs; dashed = identity. |
| `evaluation/figures/fig22_calibration_real_error_vs_cosine.png` | REAL DATA | Estimation error by true-cosine bin on real pairs (mean ± SD) against the SD predicted by the synthetic calibration. |
| `evaluation/figures/fig23_calibration_real_residual_hist.png` | REAL DATA | Histogram of estimation residuals on real pairs (clamped estimates excluded). |
| `evaluation/figures/fig25_accuracy_f1_vs_threshold.png` | REAL DATA | Accuracy and F1 of the protected-template decision vs threshold (depend on the genuine:impostor ratio of the protocol). |
| `evaluation/figures/fig26_fusion_roc.png` | REAL DATA | ROC of single protected modalities vs equal-weight mean fusion score, chimeric virtual users from real data. |
| `evaluation/figures/fig27_robustness_frr_far.png` | REAL DATA | FRR (bars) and FAR (ticks) of the deployed protected-template rule under degradations of real probes. |
| `evaluation/figures/fig28_memory_benchmark.png` | REAL DATA | Peak process memory per scenario (fresh process each, mean of 3 runs). |

## Tables

- `evaluation/tables/table01_system_configuration.md`
- `evaluation/tables/table02_dataset_summary.md`
- `evaluation/tables/table03_threshold_configuration.md`
- `evaluation/tables/table04_calibration_statistics.md`
- `evaluation/tables/table05_model_performance_raw.md`
- `evaluation/tables/table06_protected_template_performance.md`
- `evaluation/tables/table07_fusion_performance.md`
- `evaluation/tables/table08_latency.md`
- `evaluation/tables/table09_storage.md`
- `evaluation/tables/table10_security_properties.md`
- `evaluation/tables/table11_software_validation.md`
- `evaluation/tables/table12_limitations.md`

## Unavailable metrics

| metric | reason | what would produce it |
|---|---|---|
| FAR/FRR/EER of real users of the deployed system | no consented participant data | `evaluation/real_user_evaluation.py` |
| IAPMR (presentation attacks) | no attack data | `evaluation/ieee/pad_evaluation.py` with data/pad/ |
| APCER / BPCER | no PAD subsystem exists | would require implementing PAD |
| Head-pose robustness | cannot be simulated faithfully | real multi-pose captures |
| Physical-microphone robustness | needs re-recordings | real captures on several devices |
| Demographic breakdown | no demographic labels | consented study with self-reported attributes |
| Test coverage | pytest-cov not installed | `pip install pytest-cov` then `pytest --cov` |
| Frontend unit tests | no frontend test suite | add vitest |
| Face metrics of the committed face_metrics.csv | no pair data committed | superseded by Part 4 held-out LFW evaluation |

## Hardware

- os: Windows 11 (10.0.26200)
- python: 3.13.1
- cpu_logical_cores: 12
- ram_gb: 8.3
- processor: 12th Gen Intel(R) Core(TM) i5-1235U
- torch: 2.14.0+cpu
- gpu: none (CPU only)
- torch_threads: 10

## Runtime

| timestamp | step | seconds |
|---|---|---|
| 2026-09-24 22:43:36 | part18_diagrams | 7.8 |
| 2026-09-24 22:44:07 | part18_diagrams | 6.4 |
| 2026-09-24 22:44:32 | part18_diagrams | 9.6 |
| 2026-09-24 22:46:14 | part1_audit | 14.1 |
| 2026-09-24 22:47:06 | part12_storage | 1069.0 |
| 2026-09-24 22:54:15 | part2_verification | 86.9 |
| 2026-09-24 23:02:39 | part2_verification | 91.6 |
| 2026-09-24 23:17:46 | part3 | 255.2 |
| 2026-09-24 23:18:03 | part4 | 17.2 |
| 2026-09-24 23:18:05 | part6 | 1.3 |
| 2026-09-24 23:23:38 | part8 | 253.2 |
| 2026-09-24 23:28:32 | part16 | 294.2 |
| 2026-09-24 23:29:30 | part14 | 58.3 |
| 2026-09-24 23:30:47 | part7_9 | 77.0 |
| 2026-09-24 23:34:11 | part8 | 187.2 |
| 2026-09-24 23:34:11 | part6 | 0.7 |
| 2026-09-24 23:40:59 | part8 | 355.0 |
| 2026-09-24 23:42:54 | part14 | 114.8 |
| 2026-09-24 23:48:25 | part13_robustness_face | 832.9 |
| 2026-09-25 00:35:55 | part1_audit | 21.2 |
| 2026-09-25 00:55:49 | part13_robustness_voice | 407.8 |
| 2026-09-25 01:22:41 | part17_software | 291.4 |
| 2026-09-25 01:28:44 | part12_storage | 333.2 |
| 2026-09-25 01:29:03 | part1_audit | 3.3 |
| 2026-09-25 01:29:05 | part18_diagrams | 1.5 |

## Reproducibility

1. `git lfs install && git lfs pull` (real checkpoints; the pipelines refuse to run on mock embedders).
2. `pip install -r requirements.txt`; place a Kaggle API token in `~/.kaggle/kaggle.json`.
3. Download the data into the gitignored `data/` folder: LFW via scikit-learn (`fetch_lfw_people(data_home='data/lfw')` downloads the archive), `gaurav41/voxceleb1-audio-wav-files-for-india-celebrity` -> `data/voxceleb_subset` (Kaggle API).
4. `python -m evaluation.ieee.run_all` (or the per-step commands above). Seeds are fixed (`evaluation/ieee/common.py::SEED`, split seed 42).
5. Evaluation keys derive from a fixed evaluation-only secret (`EVAL_SECRET`), never a deployment `MASTER_SECRET`.
6. Raw data and embedding caches stay in `data/` (gitignored); result files contain scores only.

## Generated files

- `evaluation/reports/hardware.json`
- `evaluation/reports/latency_report.md`
- `evaluation/reports/pytest_junit.xml`
- `evaluation/reports/runtime_log.csv`
- `evaluation/reports/software_validation.md`
- `evaluation/results/.gitkeep`
- `evaluation/results/api_endpoints.csv`
- `evaluation/results/biohash_metric_calibration.json`
- `evaluation/results/calibration_real_binned.csv`
- `evaluation/results/calibration_real_validation.csv`
- `evaluation/results/configuration.csv`
- `evaluation/results/dataset_audit.csv`
- `evaluation/results/db_schema.csv`
- `evaluation/results/face_metrics.csv`
- `evaluation/results/fusion_policy_metrics.csv`
- `evaluation/results/fusion_score_level_eer.csv`
- `evaluation/results/latency_benchmark.csv`
- `evaluation/results/memory_benchmark.csv`
- `evaluation/results/metric_verification.csv`
- `evaluation/results/pad_results.csv`
- `evaluation/results/raw_vs_protected_metrics.csv`
- `evaluation/results/revocation_per_attempt.csv`
- `evaluation/results/revocation_summary.csv`
- `evaluation/results/robustness.csv`
- `evaluation/results/software_validation.csv`
- `evaluation/results/storage_analysis.csv`
- `evaluation/results/template_security.csv`
- `evaluation/results/template_set_diversity.csv`
- `evaluation/results/template_set_exhaustion.csv`
- `evaluation/results/template_set_promotion.csv`
- `evaluation/results/template_set_revocation.csv`
- `evaluation/results/threshold_sweep.csv`
- `evaluation/results/threshold_sweep_eer.csv`
- `evaluation/results/unlinkability.csv`
- `evaluation/results/voice_confusion_matrix.csv`
- `evaluation/results/voice_metrics.csv`
- `evaluation/results/voice_roc.csv`
- `evaluation/tables/table01_system_configuration.csv`
- `evaluation/tables/table01_system_configuration.md`
- `evaluation/tables/table02_dataset_summary.csv`
- `evaluation/tables/table02_dataset_summary.md`
- `evaluation/tables/table03_threshold_configuration.csv`
- `evaluation/tables/table03_threshold_configuration.md`
- `evaluation/tables/table04_calibration_statistics.csv`
- `evaluation/tables/table04_calibration_statistics.md`
- `evaluation/tables/table05_model_performance_raw.csv`
- `evaluation/tables/table05_model_performance_raw.md`
- `evaluation/tables/table06_protected_template_performance.csv`
- `evaluation/tables/table06_protected_template_performance.md`
- `evaluation/tables/table07_fusion_performance.csv`
- `evaluation/tables/table07_fusion_performance.md`
- `evaluation/tables/table08_latency.csv`
- `evaluation/tables/table08_latency.md`
- `evaluation/tables/table09_storage.csv`
- `evaluation/tables/table09_storage.md`
- `evaluation/tables/table10_security_properties.csv`
- `evaluation/tables/table10_security_properties.md`
- `evaluation/tables/table11_software_validation.csv`
- `evaluation/tables/table11_software_validation.md`
- `evaluation/tables/table12_limitations.csv`
- `evaluation/tables/table12_limitations.md`
