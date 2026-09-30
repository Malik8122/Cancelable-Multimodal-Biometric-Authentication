# Master project audit

> **Modality change (2026-09-27).** Iris and fingerprint have been removed from the system: the implemented
> modalities are **face and voice**, and the `AT_LEAST_TWO` fusion policy (which needed three modalities) is gone.
> This document is a dated record. Its iris / fingerprint material (models, datasets, results, thresholds,
> three-modality fusion) is kept for history only and no longer describes the code. Current state: `README.md`,
> `docs/ROADMAP.md`; current numbers: `evaluation/PAPER_RESULTS_INDEX.md`.

Audited 2026-09-25 at `main` @ `873c13e` plus uncommitted evaluation work. Where sources conflict, this order decides:

1. code
2. checkpoints the code loads
3. reproducible artifacts
4. configuration
5. documentation
6. older reports

Line numbers drift, so functions are named instead. Evidence labels used throughout: [A] REAL DATA, [B] SYNTHETIC,
[C] SOFTWARE TEST, [D] CONFIGURATION, [E] DERIVED, [F] FUTURE / NOT AVAILABLE.

## 1. Architecture

```
capture (browser)
  -> FastAPI (backend/api/*)
  -> decode_biometric_sample (backend/utils.py)
  -> per-modality ModalityService (backend/services/base_service.py)
       preprocess -> embed -> L2 -> BioHash 256 bits (template_protection/biohash.py) under an HKDF key
       -> Hamming similarity vs the ACTIVE template set (template_protection/matcher.py)
       -> calibrated estimate + threshold (backend/services/modality_metrics.py::decide)
  -> fusion policy (fusion/policy.py)
  -> ACCESS_GRANTED / ACCESS_DENIED / ENROLLMENT_REQUIRED
  -> audit log (backend/database/models.py::AuditLog)
```

- The frontend is React/Vite (`frontend/src`).
- The database is SQLite through SQLAlchemy.
- Four template sets are created per enrollment (T1 ACTIVE, T2–T4 STANDBY), each under its own `key_version`.

## 2. Modality scope (from the code, not the README)

| Modality | API (`backend/services/__init__.py::_SUPPORTED_MODALITIES`) | Checkpoint | Enrollment UI | Authentication UI | Status |
|---|---|---|---|---|---|
| Face | yes | `models/face/saved/face_embedder.pt` | `RegisterPage` (5-pose guided) | `AuthenticatePage` | **deployed** |
| Voice | yes | `models/voice/saved/voice_embedder.pt` | yes (2 recordings) | yes | **deployed** |
| Fingerprint | yes | `models/fingerprint/saved/fingerprint_embedder.pt` | yes (image upload) | yes | **deployed, optional** |
| Iris | yes (API accepts it) | **none**, so it uses a mock embedder | not offered | not offered | **not implemented**; no result exists |

**Is fingerprint mandatory?** No.

- `authenticate_samples` authenticates exactly the modalities the user submits. `AuthenticatePage` preselects nothing.
- ALL_REQUIRED (the default) requires every **submitted** modality to pass.
- AT_LEAST_TWO is only valid with exactly three submitted modalities, so it is the one policy that needs fingerprint.
- Buildings cannot require modalities: `backend/buildings.py` rejects `required_modalities`.

## 3. Models and dimensions

| Modality | Model | Init / training | Embedding | Verified in |
|---|---|---|---|---|
| Face | InceptionResnetV1 (facenet-pytorch) | VGGFace2-pretrained; fine-tuned on LFW (62 identities with ≥ 20 images) with an ArcFace head | **512** | `models/face/inference.py::FACE_EMBEDDING_DIM` |
| Voice | ECAPA-TDNN (SpeechBrain module) | **trained from scratch**; no pretrained speaker weights (`docs/VOICE_MODEL.md`) | **192** | `models/voice/inference.py::VOICE_EMBEDDING_DIM` |
| Fingerprint | ResNet50 (ImageNet) + head | fine-tuned on SOCOFing, subject-disjoint | **512** | `models/fingerprint/inference.py::FINGERPRINT_EMBEDDING_DIM` |
| Iris | ResNet18 + head | not trained | 256 (mock) | `models/iris/inference.py` |

## 4. Preprocessing

- **Face, deployed:** `FacePreprocessor` / `FacePreprocessorBaseline`, which is MTCNN plus a **bounding-box crop** resized
  to 160×160, with `(x−127.5)/128` in the embedder.
  - The landmarks are used only for the enrollment quality gates.
  - An optional similarity-alignment mode is `FacePreprocessorAligned` (`FACE_ALIGNMENT=similarity`, alias `FACE_ALIGNED`).
  - Details: `docs/FACE_ALIGNMENT_AUDIT.md`, `evaluation/reports/FACE_ALIGNMENT_DECISION.md`.
- **Face quality gates** (enrollment, measured before alignment), from `embeddings/pipelines.py::_evaluate_face_quality`:

  | Gate | Condition that rejects |
  |---|---|
  | confidence | < 0.90 |
  | single face | more than one face |
  | sharpness | < 25 (Laplacian variance of the crop to be embedded) |
  | size ratio | < 0.15 |
  | centre offset | > 0.35 |
  | roll | > 20° |
  | yaw ratio | > 0.20 |
  | valid poses | fewer than 3 of 5 |

  Statuses: NO_FACE, MULTIPLE_FACES, LOW_CONFIDENCE, BLURRY, TOO_SMALL, OFF_CENTER, TOO_ANGLED (the task's
  POSE_INVALID), and in aligned mode LANDMARK_FAILURE and ALIGNMENT_FAILED.
- **Voice:** 16 kHz log-mel front end (`preprocessing/voice.py`), with a recording-quality and consistency gate at
  enrollment (`backend/services/recording_quality.py`).
- **Fingerprint:** CLAHE, ridge normalization and Gabor enhancement (`preprocessing/fingerprint.py`).

## 5. Training records

- Face and fingerprint were trained on Kaggle notebooks, voice by the training script.
- Their historical per-epoch values are **NOT AVAILABLE**, except where they are quoted in documents (`training/TRAINING_SUMMARY.md`).
- Reproduction runs with full records (`config.json`, `training_log.csv`, `environment.json`, `checkpoint_manifest.json`) are
  under `training/<modality>/runs/`. The runner is `training/run_training.py` and the recorder is `training/recorder.py`.
- Face reproduced: COMPLETED. The controlled face alignment pair (v2) is under `training/face/bbox_fullframe_v2` and
  `training/face/aligned_v2`.
- Voice reproduction: INTERRUPTED after epoch 8, because the process ended with an earlier session. It has no
  `run_status` COMPLETED.

## 6. BioHash and matching

- The template is `generate_template(embedding, key, 256)`:
  1. L2 normalization.
  2. A key-seeded Haar-orthonormal projection. For voice (192 < 256) this uses two blocks.
  3. Quantization against key-seeded thresholds with a jitter of 0.5·σ.
  4. A key-seeded permutation.
  5. Storage as packed bits in `ProtectedTemplate.protected_template`.
- Keys are HKDF-SHA256 from `MASTER_SECRET`, `application_id`, `user_id`, `modality` and `key_version`
  (`template_protection/hkdf_keys.py`).
- Comparison is always a Hamming similarity, the fraction of equal bits (`template_protection/matcher.py`).
- The runtime template length is **256 bits** (`Settings.template_bits`).

## 7. Calibration and thresholds

| Modality | Metric (calibrated **estimate** from the Hamming similarity) | Rule | Value | Source |
|---|---|---|---|---|
| Face | estimated cosine | ≥ | **0.80** | `Settings.face_cosine_threshold` [D], teacher-requested |
| Voice | estimated Euclidean distance (unit vectors) | ≤ | **0.75** | `Settings.voice_euclidean_threshold` [D], teacher-requested |
| Fingerprint | Hamming similarity | ≥ | **0.90** | `Settings.match_threshold` fallback [D]. No `evaluation/results/fingerprint_threshold.json` exists, so `backend/threshold_loader.py` logs that it is uncalibrated |

- `Settings.threshold_source = TEACHER_REQUESTED_BASELINE` records this provenance; it was added in this audit.
- Calibration curves live in `evaluation/results/biohash_metric_calibration.json` [B]. They are fitted on synthetic
  unit-vector pairs through the real transform (`scripts/calibrate_biohash_metric_mapping.py`).
- The curves are validated on real embeddings in `evaluation/results/calibration_real_validation.csv` [A].

## 8. Fusion

- The per-modality `fusion_score` is on the estimated-cosine scale:
  - face: the estimate itself;
  - voice: cos = 1 − d²/2;
  - fingerprint: through its calibration curve.
- The fused score is the mean of the submitted modalities' scores.
- Policies:
  - ALL_REQUIRED (default): every submitted modality must pass.
  - WEIGHTED: the fused score must reach the threshold, and any submitted score below the floor vetoes.
  - AT_LEAST_TWO: needs exactly three submitted modalities, at least two of which pass.

## 9. Database and security

- **Stored:** in `backend/database/models.py`:

  | Table | Contents |
  |---|---|
  | `users` | id, optional display name |
  | `protected_templates` | template bytes, key_version, set version/status, ACTIVE/STANDBY/REVOKED timestamps |
  | `master_secret_fingerprints` | an HKDF fingerprint of the secret, not the secret itself |
  | `audit_logs` | scores, thresholds, decision, policy, template/key versions |
  | `enrollment_events` | enrollment outcomes |
- **Not stored:** raw images, audio, embeddings or landmarks.
  - Enrollment centroid embeddings are transient.
  - Checked with `evaluation/results/db_schema.csv` [D] and `storage_analysis.csv`.
- `MASTER_SECRET` has no default; the process fails fast if it is missing.
- No logging call formats the secret (grep of `backend/`).
- Per-modality scores leave the API only when `DEBUG_SCORES=true`.
- Claims **not** made: irreversible, unhackable, secure against all attacks, zero privacy risk. `docs/PRIVACY_AND_SECURITY.md`
  states the limits.

## 10. Evaluation and testing (where the evidence lives)

| Topic | Result file(s) | Script | Label |
|---|---|---|---|
| Raw vs protected, identical pairs | `protected_vs_raw.csv`, `raw_vs_protected_metrics.csv` | `evaluation/scripts/protected_vs_raw.py`, `evaluation/ieee/experiments.py part4` | [A] |
| Threshold sweep + dev/test operating points | `threshold_sensitivity.csv`, `threshold_operating_points.csv` | `evaluation/ieee/threshold_analysis.py` | [A] |
| Synthetic calibration | `biohash_metric_calibration.json` | `scripts/calibrate_biohash_metric_mapping.py` | [B] |
| Real-embedding calibration | `calibration_real_validation.csv`, `calibration_real_binned.csv` | `scripts/validate_calibration_real.py` | [A] |
| Fusion | `fusion_policy_metrics.csv`, `fusion_score_level_eer.csv` | `evaluation/ieee/experiments.py part14` | [A] (chimeric) |
| Revocation / template sets | `revocation_summary.csv`, `template_set_*.csv` | `experiments.py part7_9`, `evaluation/template_set_experiments.py` | [A] / [B] |
| Unlinkability | `unlinkability.csv` | `experiments.py part8` (D↔ / D_sys, Gomez-Barrero et al. 2018) | [A] |
| Latency / memory | `latency_benchmark.csv`, `memory_benchmark.csv`, `face_alignment_latency.csv` | `scripts/benchmark_latency.py`, `benchmark_memory.py` | [A] (hardware-specific) |
| Face alignment | `face_alignment_*.csv` | `evaluation/scripts/evaluate_face_alignment.py` | [A] |
| Real users | none | `evaluation/real_user_evaluation.py`, `evaluation/REAL_USER_PROTOCOL.md` | [F] |
| Software tests | `software_validation.csv` | `pytest` | [C] |

Software tests are not biometric trials. The test count is software validation only and is re-measured in
`docs/FINAL_PROJECT_VALIDATION.md`.

## 11. Findings

| # | FILE | FUNCTION | CURRENT BEHAVIOR | EXPECTED BEHAVIOR | ACTION |
|---|---|---|---|---|---|
| 1 | `scripts/calibrate_biohash_metric_mapping.py` | `MODALITY_DIMS` | fingerprint fitted at D=256 | D=512 (the model's dimension) | **Fixed.** Refitted: face and voice bit-identical, fingerprint curve Δ ≤ 0.00056; tests added |
| 2 | `evaluation/template_set_experiments.py` | `_MODALITY_DIMS` | fingerprint 256 | 512 | **Fixed**; re-run, conclusions unchanged |
| 3 | `docs/PROJECT_REPORT.md` | §8.2 | old fingerprint EER 44.5 % / AUC 0.579 presented as the result | current EER 30.77 % / AUC 0.7644 | **Fixed**: history kept, marked superseded, banner added (`docs/FINGERPRINT_DOCUMENTATION_AUDIT.md`) |
| 4 | `docs/ARCHITECTURE.md`, `docs/COMPUTER_VISION.md`, `docs/PROJECT_FUNDAMENTALS.md`, `docs/PROJECT_REPORT.md` | face preprocessing | claimed landmark-based geometric alignment | bounding-box crop (deployed); alignment optional | **Fixed** |
| 5 | `docs/TEMPLATE_PROTECTION.md`, `docs/COMPUTER_VISION.md` | dimensions | fingerprint "256-d" | 512 | **Fixed** |
| 6 | `docs/PROJECT_FUNDAMENTALS.md` | face rejection rules | "only NO_FACE or BLURRY" | 7 gates plus the valid-pose rule | **Fixed** |
| 7 | `docs/PROJECT_REPORT.md`, `README.md` | scope | "face, iris and fingerprint" | face, voice, fingerprint deployed; iris mock-only | PROJECT_REPORT: banner. README: already says iris is mock-only; scope line updated |
| 8 | `preprocessing/face.py` | `FacePreprocessor.detect_and_align` (baseline) | non-finite landmarks would give NaN roll/yaw, which pass the `>` gates (MTCNN never returns NaN landmarks with a box, so this is not observed) | reject | **Not changed** (baseline preserved). The aligned mode rejects with LANDMARK_FAILURE. Documented |
| 9 | `backend/config.py` | thresholds | no provenance recorded | provenance label | **Added** `threshold_source = TEACHER_REQUESTED_BASELINE` |
| 10 | `preprocessing/face.py` | aligned mode | one failure code for missing and degenerate landmarks | distinct codes | **Added** `LandmarkFailure` / LANDMARK_FAILURE |
| 11 | `evaluation/real_user_evaluation.py` | consent/deletion | one global `--consent-confirmed` flag | per-participant consent, withdrawal, deletion | **Added** `consent.csv`, `--withdraw`, `--delete-captures`, protocol check |
| 12 | `evaluation/ieee/experiments.py` | `_plot_unlinkability` | invalid escape `\l` (SyntaxWarning) | raw string | **Fixed** |
| 13 | fingerprint threshold | `backend/threshold_loader.py` | 0.90 fallback, uncalibrated | a data-selected threshold (future) | Documented [F] |
| 14 | `.gitignore` | checkpoints | `training/face/<variant>/runs/*/checkpoints/*.pt` (about 110 MB each) not ignored | ignored | **Fixed** |
| 15 | Face at 0.80 | `decide` | FRR 81.5 % (protected, held-out LFW) | — | **Preserved** as a baseline finding (`evaluation/reports/BASELINE_THRESHOLD_FINDINGS.md`) |

## 12. Known missing evidence ([F] NOT AVAILABLE)

- Real-user (participant) results: no data has been collected.
- Historical per-epoch training curves for the deployed face and fingerprint checkpoints.
- A completed reproduction run of voice training (interrupted after epoch 8).
- Iris: no model, no result.
- Formal irreversibility or security proofs, and attack evaluations beyond the ones in `template_security.csv` / `pad_results.csv`.
- A data-selected fingerprint threshold.
- Real face+voice captures from the same person. All fusion results use chimeric pairings of LFW faces with
  VoxCeleb voices.
