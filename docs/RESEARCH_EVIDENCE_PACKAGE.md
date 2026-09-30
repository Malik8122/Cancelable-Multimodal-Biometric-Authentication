# Research Evidence Package

> **Modality change (2026-09-27).** Iris and fingerprint have been removed from the system: the implemented
> modalities are **face and voice**, and the `AT_LEAST_TWO` fusion policy (which needed three modalities) is gone.
> This document is a dated record. Its iris / fingerprint material (models, datasets, results, thresholds,
> three-modality fusion) is kept for history only and no longer describes the code. Current state: `README.md`,
> `docs/ROADMAP.md`; current numbers: `evaluation/PAPER_RESULTS_INDEX.md`.

**Project:** Cancelable Multimodal Biometric Authentication for Critical Infrastructure
**Audited state:** `main` @ `873c13e` ("Implement multimodal biometric updates"), 2026-09-24
**Purpose:** the verified facts, numbers, equations and limitations the capstone paper may use, each with its source.
This is not the paper.

> **Snapshot (2026-09-24).** It is superseded where it conflicts with `evaluation/PAPER_RESULTS_INDEX.md` and
> `docs/FINAL_PROJECT_VALIDATION.md` (2026-09-25). Three items it lists as open are now resolved: the fingerprint D = 256
> calibration (refitted at 512), face alignment (implemented and evaluated), and the software-test count (now 636).

Every number below carries one of these evidence labels. Never mix them in the paper.

| Label | Meaning |
|---|---|
| **[A] REAL-DATA RESULT** | Measured on real biometric data (public datasets or real captures) |
| **[B] SYNTHETIC CALIBRATION / SIMULATION** | Measured on synthetic vectors through the real code |
| **[C] SOFTWARE TEST** | Software correctness evidence, not biometric performance |
| **[D] CONFIGURATION** | A value set in code or config, not measured |
| **[E] DERIVATION** | Follows mathematically from [B]/[D] |
| **[F] FUTURE** | Proposed, not done |

> Three distinctions that must survive into the paper:
> 559 passing backend tests ≠ 559 authentication trials.
> The synthetic BioHash calibration curve ≠ real-user biometric performance.
> The face 0.80 / voice 0.75 thresholds are **teacher-requested configuration values**, not experimentally optimized.

---

## 1. Executive research summary

- **What exists.** A working web system (FastAPI backend + React frontend) that enrolls face and voice (and optionally
  fingerprint), transforms each embedding into a **256-bit keyed BioHash template**, stores **only** those templates,
  and authenticates by **Hamming comparison** of templates. Per-modality decisions are expressed as a calibrated
  **estimate** of face cosine similarity (≥ 0.80) and voice Euclidean distance (≤ 0.75). Decisions are fused under an
  `ALL_REQUIRED` policy. Templates are revocable through a pool of 4 template sets. Users register with a display name
  that is kept separate from the internal ID.
- **Strongest legitimate numeric evidence.**
  1. **[A]** Voice embedding model on a held-out VoxCeleb1-subset split (raw embeddings, not templates): EER 2.29 %,
     AUC 0.9967 over 281,625 pairs.
  2. **[B]** The Hamming→cosine calibration fits with R² ≥ 0.99986 and a cosine-space RMSE of 0.004–0.005.
  3. **[B]** Template-set experiments on synthetic embeddings: cross-key similarity 0.500 ± 0.029 (unlinkable-looking),
     revocation drops the old template to 0.504, 80/80 promotions accepted.
  4. **[C]** 559/559 backend tests pass.
- **What does not exist.** No labelled genuine/impostor evaluation of the **deployed protected-template system** on real
  users. So **FAR, FRR, EER and ROC-AUC of the actual system cannot be established from the current repository.** There
  is also no formal security evaluation and no latency benchmark. The only timing data is incidental audit-log latency.
- **Paper readiness:** suitable today for a **system-design / implementation paper** with honest limitations. **Not**
  suitable for any claim of authentication accuracy of the deployed system (see §18).

---

## 2. Repository architecture (as implemented)

```
Browser (React, frontend/)  --multipart-->  FastAPI (backend/main.py)
   Register: POST /users -> /enroll/face (5 poses) -> /enroll (voice x2)
   Verify:   POST /authenticate/fusion (any enrolled subset)
                     |
     backend/services/authentication.py::authenticate_samples
                     |  per modality
     backend/services/base_service.py::ModalityService
        embed (embeddings/pipelines.py) -> BioHash (template_protection/biohash.py)
        -> Hamming compare (template_protection/matcher.py)
        -> decision (backend/services/modality_metrics.py)
                     |
     fusion/policy.py::evaluate_fusion_policy (ALL_REQUIRED default)
                     |
     SQLite (backend/database/models.py): users, protected_templates, audit_logs, enrollment_events
```

### 2.1 Face pipeline

| Stage | File / function | Input → output | Parameters | Label |
|---|---|---|---|---|
| Capture | `frontend/src/components/capture/GuidedFaceCapture.tsx` | webcam frame → PNG | 5 poses: front, left, right, up, down | [D] |
| Detection + crop | `preprocessing/face.py::FacePreprocessor` (MTCNN, facenet-pytorch) | RGB → 160×160 crop | `image_size=160, margin=0, post_process=True` | [D] |
| Enrollment quality gates | `preprocessing/face.py`, `embeddings/pipelines.py` | crop → VALID/BLURRY/… | Laplacian sharpness ≥ 25.0; detection prob ≥ 0.90; face-size ratio ≥ 0.15; centre offset ≤ 0.35; roll ≤ 20°; yaw ratio ≤ 0.20; ≥ 3 valid of 5 (`MIN_VALID_POSES`) | [D] |
| Embedding | `models/face/inference.py::FaceEmbedder` (InceptionResnetV1) | 160×160×3 → ℝ⁵¹² | checkpoint `models/face/saved/face_embedder.pt` (Git LFS) | [D] |
| Normalization | `models/common/base_embedder.py::_l2_normalize` | → unit vector | ‖e‖₂ = 1 | [D] |
| Enrollment centroid | `embeddings/centroid.py::centroid_embedding` | valid pose embeddings → one unit vector | normalize(mean(eᵢ)) | [D] |
| BioHash | `template_protection/biohash.py::generate_template` | ℝ⁵¹² → {0,1}²⁵⁶ | see §2.3 | [D] |
| Compare + decide | `matcher.compare`, `modality_metrics.decide` | two templates → ĉ, match | ĉ ≥ 0.80 | [D] |

Model provenance (documented, not re-verifiable from committed artefacts): InceptionResnetV1 pretrained on VGGFace2 and
fine-tuned with an ArcFace head on LFW (62 identities, 3,023 images, `min_faces_per_person=20`)
(`docs/PROJECT_REPORT.md` §8.2).

### 2.2 Voice pipeline

| Stage | File / function | Input → output | Parameters | Label |
|---|---|---|---|---|
| Capture | `frontend/.../VoiceCapture.tsx`, `useWavRecorder.ts` | mic → WAV | 4–5 s, auto-stop at 5 s; phrase "Security authentication for government access." | [D] |
| Preprocessing | `preprocessing/voice.py` | waveform → 80 × frames log-mel | 16 kHz resample, mono, energy VAD (ratio 0.02), RMS normalization (target 0.1), 4.0 s segment, n_fft 400, hop 160, 80 mel | [D] |
| Embedding | `models/voice/inference.py::VoiceEmbedder` (ECAPA-TDNN, `models/voice/model.py`) | log-mel → ℝ¹⁹² | checkpoint `voice_embedder.pt`; trained with ArcFace m = 0.5, s = 30 (`models/voice/saved/training_config.json`) | [D] |
| Normalization | `base_embedder._l2_normalize` | → unit vector | ‖e‖₂ = 1 (verified in code; this is what makes d = √(2−2cos) valid) | [D] |
| Enrollment consistency | `backend/services/recording_quality.py` | two recordings → band | **exact** cosine of two in-memory embeddings: EXCELLENT ≥ 0.85, GOOD ≥ 0.75, FAIR ≥ 0.60 (needs confirmation), POOR < 0.60 (rejected); the template is built from recording 1 | [D] |
| BioHash | same as face | ℝ¹⁹² → {0,1}²⁵⁶ | 256 > 192, so the projection uses two orthonormal blocks (192 + 64 rows), not mutually orthogonal (`transform.build_orthonormal_projection`) | [D] |
| Compare + decide | `modality_metrics.decide` | → d̂, match | d̂ ≤ 0.75 | [D] |

### 2.3 Cancelable transform (BioHash) — `template_protection/`

1. **Key material** (`hkdf_keys.py::derive_key`): salt = SHA-256(`application_id|user_id|modality|key_version`); three
   HKDF-SHA256 expansions of `MASTER_SECRET` with distinct `info` labels → 32-byte `projection_seed`,
   `permutation_seed`, `threshold_seed`.
2. **Normalize:** x ← x/‖x‖.
3. **Project** (`transform.py`): W ∈ ℝ^{256×D}, rows orthonormal within blocks of ≤ D (QR of a seeded Gaussian, i.e.
   Haar-random); p = Wx, each pᵢ ∈ [−1, 1].
4. **Quantize:** bᵢ = 1[pᵢ > tᵢ], tᵢ ~ N(0, (0.5·std(p))²), seeded by `threshold_seed`.
5. **Permute:** a keyed permutation π of the 256 bits.
6. **Store:** `pack_bits`, 256 bits = **32 bytes** (verified: `length(protected_template)=32` for every row in the local DB).

`TEMPLATE_FORMAT_VERSION = 1`. The library default is `DEFAULT_OUTPUT_BITS = 128`, but runtime always passes
`Settings.template_bits = 256`; the stored row keeps its own `output_bits`.

### 2.4 Comparison, decision and fusion

- `matcher.compare(a, b, "hamming")` = fraction of equal bits (1.0 on constant-time exact equality).
- `modality_metrics.decide` converts that into the modality's metric and decision (§8).
- `authentication.authenticate_samples`: every submitted modality is authenticated against the **ACTIVE** template set
  (never STANDBY/REVOKED, and never sets mixed), then fused by `fusion/policy.py` (§8).

---

## 3. Complete numerical evidence sheet (traceability)

Format: **name → value → source → interpretation**.

### 3.1 Configuration [D]
- Face embedding dimension → 512 → `models/face/inference.py::FACE_EMBEDDING_DIM` → BioHash input size
- Voice embedding dimension → 192 → `models/voice/inference.py::VOICE_EMBEDDING_DIM` → BioHash input size
- Fingerprint embedding dimension → **512** → `models/fingerprint/inference.py::FINGERPRINT_EMBEDDING_DIM` (since `fc9fd47`) → optional modality. **Note:** the calibration script calibrates fingerprint at D = 256 (`MODALITY_DIMS`), a mismatch (see §15 #8)
- Template length → 256 bits → `backend/config.py::Settings.template_bits` → stored template size
- Stored template size → 32 bytes → local DB `protected_templates.protected_template` → 256/8
- Template sets per user → 4 → `Settings.template_pool_size` → 1 ACTIVE + 3 STANDBY
- Face threshold → 0.80 (estimated cosine, higher = better) → `Settings.face_cosine_threshold` → teacher-requested
- Voice threshold → 0.75 (estimated Euclidean distance, lower = better) → `Settings.voice_euclidean_threshold` → teacher-requested
- Fingerprint threshold → 0.90 (Hamming similarity) → `Settings.match_threshold` fallback (`backend/threshold_loader.py`) → inherited default
- Default fusion policy → ALL_REQUIRED → `fusion/config.py::DEFAULT_FUSION_POLICY`
- Fusion weights → equal (1.0 each, renormalized) → `fusion/score_fusion.py::resolve_normalized_weights`
- WEIGHTED veto floor → 0.0 (estimated-cosine scale) → `fusion/config.py::DEFAULT_WEIGHTED_FLOOR`
- HKDF → SHA-256, 32-byte seeds → `template_protection/hkdf_keys.py`
- Quantization jitter → σ = 0.5·std(p) → `transform.quantize`
- Face input size → 160×160 → `preprocessing/face.py::FACE_INPUT_SIZE`
- Voice audio → 16 kHz, 4.0 s, 80 mel, n_fft 400, hop 160 → `preprocessing/voice.py`
- Upload limit → 5,000,000 bytes → `Settings.max_upload_size_bytes`
- Display name → 1–64 chars; letters, spaces and ' ’ - .; must start with a letter → `backend/display_names.py`

### 3.2 Derived thresholds [E] (from the calibration curve; computed in this audit with `modality_metrics.decide` / `MetricCurve`)
- Face ĉ ≥ 0.80 ⇔ Hamming similarity ≥ **0.81759** ⇔ differing bits ≤ **46** of 256
- Voice d̂ ≤ 0.75 ⇔ ĉ ≥ 1 − 0.75²/2 = **0.71875** ⇔ Hamming similarity ≥ **0.78229** ⇔ differing bits ≤ **55** of 256
- Fingerprint h ≥ 0.90 ⇔ differing bits ≤ **25** ⇔ fusion-scale ĉ ≥ 0.9384
- Face+voice fusion threshold = (0.80 + 0.71875)/2 = **0.759375** (informational under ALL_REQUIRED)

### 3.3 Synthetic calibration [B] — `evaluation/results/biohash_metric_calibration.json`
See §4 for the full table.

### 3.4 Real-dataset model evaluations [A] — raw embeddings, NOT the protected-template system
- **Voice** (`evaluation/results/voice_metrics.csv`, `voice_confusion_matrix.csv`, `voice_roc.csv`; VoxCeleb1 Indian-celebrity subset, Kaggle, held-out split):
  - test utterances: 751; pairs: 281,625 = C(751, 2) (verified); genuine 13,101; impostor 268,524
  - EER 0.022890 at cosine threshold 0.32214; AUC 0.996724 (recomputed here by trapezoid over 4,303 ROC points; matches the CSV's 0.9967238)
  - at the EER threshold: TP 12,801, FN 300, FP 6,144, TN 262,380 → accuracy 0.977119, FAR 0.022881, FRR 0.022899,
    precision 0.675693, recall 0.977101, F1 0.798914 (every value recomputed from the confusion matrix and matching the CSV)
  - number of speakers: **NOT AVAILABLE** (printed at kernel run time, not committed)
- **Face** (`evaluation/results/face_metrics.csv`; LFW, 62 identities, 3,023 images per `docs/PROJECT_REPORT.md`):
  accuracy@EER 0.990, EER 0.010, AUC 0.999 (3 decimals, copied from a training-log line). Pair counts, ROC data, split
  protocol and whether the evaluation identities were disjoint from the training identities: **NOT AVAILABLE**. Treat as
  indicative only; likely optimistic.
- **Fingerprint** (`fingerprint_metrics.csv`; SOCOFing): 900 samples, 404,550 pairs = C(900, 2) (verified), EER 0.3077,
  AUC 0.7644 (recomputed 0.764447). Weak model and outside the face+voice scope.

### 3.5 Real captures through the deployed system — local audit log (`biometric.db`, gitignored, NOT in the repo)
These are **unlabelled**: `authenticated` is the system's decision at the time, not ground truth.

| Regime | Period | Scale | Attempts | Internal IDs | Face score stats | Voice score stats |
|---|---|---|---|---|---|---|
| R0: Hamming, threshold 0.90 | 2026-09-17 | Hamming similarity | 2 | 1 | n=2: 0.832, 0.902 | n=2: 0.805, 0.844 |
| R1: Hamming, threshold 0.80 | 2026-09-22 → 09-24 08:10 UTC | Hamming similarity | 50 | 6 | n=50: min 0.4609, Q1 0.7148, median 0.7871, Q3 0.8633, max 0.9375, mean 0.7635, sd 0.1276 | n=15: min 0.5312, Q1 0.7676, median 0.7852, Q3 0.8340, max 0.9102, mean 0.7865, sd 0.0882 |
| R2: new metric, real camera | 2026-09-24 15:32, 15:43 UTC | estimated cosine | 2 | 1 | 0.8921, 0.8858 (both granted) | — |
| R2: synthetic smoke test | 2026-09-24 15:25–15:37 UTC | fusion scale | 5 | 1 | — | 0.9985 ×3, −0.200 (clamp) ×2; synthetic Chrome microphone, **not a person** |

Decisions in R1: face-only 14 granted / 21 denied; face+voice 3 granted / 12 denied.
**Do not infer the number of people from the internal IDs.** One person may hold several IDs, and several people may
have used one.

### 3.6 Documented but not re-verifiable real observations
- `docs/AUTHENTICATION_RELIABILITY_REPORT.md`: user `OPERATOR-UWKAMM`, face 0.754 / 0.766 / 0.820 and voice
  0.797 / 0.848 / 0.879 (Hamming, asserted genuine); later "fifteen logged genuine face attempts scored 0.738–0.859
  (mean 0.784; 5 of 15 ≥ 0.80)". That audit log is **not** in the current database (0 rows for that ID). The deleted
  `face_threshold.json` quoted **0.738–0.820**, which disagrees with 0.738–0.859. **Do not cite either range as a
  verified result.**

### 3.7 Latency (incidental, from the audit log) — see §13
### 3.8 Software [C] — see §6

---

## 4. Calibration analysis [B] (synthetic, not biometric performance)

**Source:** `scripts/calibrate_biohash_metric_mapping.py` → `evaluation/results/biohash_metric_calibration.json`;
runtime `template_protection/metric_estimation.py`; decisions `backend/services/modality_metrics.py`;
tests `tests/test_modality_metrics.py`.

**Procedure.**
- **Input:** pairs of synthetic unit vectors (a, b) with prescribed cosine c, passed through the real BioHash
  transform. The fast path is asserted bit-identical to `generate_template`.
- **Grid:** 49 cosines from −0.2 to 1.0 in steps of 0.025.
- **Keys:** 50 keys × 8 pairs per key per grid point = **400 pairs per point = 19,600 pairs per modality**, seed 0,
  256 bits.
- **Why synthetic pairs are valid:** the projection is Haar-random and the jitter is scale-relative, so the transform is
  rotation-invariant. The distribution of Hamming similarity therefore depends only on (c, D, N). This is a **property
  of the transform**. It is **not** a claim about real embeddings, and it has not been validated on real embedding
  pairs (see §18, item 2).

**Model** (fitted by least squares, no intercept by construction; h(1) = 1 exactly):

$$1-h(c)=\frac{p_1\theta+p_2\theta^2+p_3\theta^3}{\pi},\qquad \theta=\arccos c$$

Inversion: ĉ = h⁻¹(observed h), tabulated over 2,001 points and clamped to [−0.2, 1.0].

| Statistic | Face (D=512) | Voice (D=192) | Fingerprint (calibrated at D=256; **the real model is 512-d**, see §15 #8) | How computed |
|---|---|---|---|---|
| p₁, p₂, p₃ | 0.881745, 0.027410, −0.021372 | 0.885522, 0.019056, −0.017816 | 0.884612, 0.022578, −0.019118 | JSON `coefficients` |
| Intercept | none (constrained h(1)=1) | none | none | model form |
| Grid points / pairs | 49 / 19,600 | 49 / 19,600 | 49 / 19,600 | JSON |
| Fit RMSE (Hamming space, fit vs 49 means) | 0.00142 | 0.00119 | 0.00116 | this audit |
| Fit MAE (Hamming space) | 0.00108 | 0.00088 | 0.00093 | this audit |
| Max abs residual (Hamming space) | 0.00379 | 0.00306 | 0.00296 | JSON `max_fit_residual` (matches) |
| Min abs residual | 0.000000 (at c=1) | 0.000000 | 0.000000 | this audit |
| R² (Hamming space) | 0.999865 | 0.999905 | 0.999909 | this audit |
| RMSE of mean-curve inversion (cosine space) | 0.00455 | 0.00403 | 0.00387 | this audit |
| MAE (cosine space) | 0.00331 | 0.00278 | 0.00297 | this audit |
| Max abs error (cosine space) | 0.0142 | 0.0118 | 0.0101 | this audit |
| **Single-estimate SD** at c = 0.90 / 0.80 / 0.725 / 0.60 | 0.032 / 0.050 / 0.063 / 0.077 | 0.031 / 0.047 / 0.057 / 0.067 | 0.030 / 0.047 / 0.054 / 0.069 | JSON `cosine_estimate_std` |
| Largest single-estimate SD (any c) | 0.114 | 0.106 | 0.101 | JSON |
| h at c = 0 (unrelated, same key) | 0.5643 | 0.5673 | 0.5638 | JSON `hamming_similarity_mean` |
| h at clamp floor c = −0.2 | 0.5131 (fit) | 0.5130 | 0.5123 | model |

**Interpretation:**
- The **model fit** is near-perfect (R² ≈ 0.9999).
- The **per-attempt uncertainty** is what matters for decisions: about ±0.05 in cosine at the face threshold. That is
  inherent to 256-bit quantization. It is not a fit error.

**Extrapolation risk:**
- Observed h below about 0.513 clamps to ĉ = −0.2 (d̂ = 1.549). The smoke test's wrong voice (h = 0.496) hit this
  clamp, so the displayed "−0.20" is a floor, not a measurement.
- Under a **shared** key, unrelated embeddings give h ≈ 0.56, not 0.50, because the quantization thresholds are shared.

**Cross-check with independent earlier data [B]:** `AUTHENTICATION_RELIABILITY_REPORT.md` (25 pairs per row) gives
h(0.80) = 0.814 (face) and 0.816 (voice), against 0.8186 / 0.8172 here. h(0.00) = 0.547 / 0.558 against 0.564 / 0.567.
The two agree within sampling error.

---

## 5. Biometric evaluation analysis

| Question | Answer |
|---|---|
| Labelled genuine/impostor trials of the deployed protected-template system | **NOT AVAILABLE FROM CURRENT REPOSITORY** |
| FAR / FRR / EER / ROC-AUC / TAR of the deployed system (face, voice, fusion) | **NOT AVAILABLE** |
| Accuracy / precision / recall / F1 of the deployed system | **NOT AVAILABLE** |
| Number of real users | **NOT AVAILABLE** (7 internal IDs in the audit log ≠ people) |
| Enrollment samples per user | By design: face 5 captures (≥ 3 valid) → 1 centroid; voice 2 recordings → template from recording 1 [D] |
| Real attempts (unlabelled) | 54 non-synthetic audit rows (R0 2 + R1 50 + R2 2): 37 face-only, 17 face+voice, 0 voice-only |
| Raw-embedding model performance | Voice [A]: EER 2.29 %, AUC 0.9967 (§3.4). Face [A]: EER 0.010, AUC 0.999 (indicative only) |
| Protected-template separability (128 vs 256 bit) | Documented only, raw data not committed: face AUC raw 0.935 → 256-bit 0.909 (EER 19.1 %); voice 0.997 → 0.986 (EER 5.6 %), from "20 synthetic identities × 5 samples, real checkpoints" (`AUTHENTICATION_RELIABILITY_REPORT.md`) |

Required sentence for the paper: *"Formal biometric performance metrics such as FAR, FRR, EER, and ROC-AUC cannot be
established for the deployed protected-template system from the current repository."*

---

## 6. Software evaluation [C]

| Item | Result | Source |
|---|---|---|
| Backend tests | **559 collected, 559 passed, 0 failed** (on the exact committed content, before `873c13e`) | `python -m pytest` |
| New tests in this update | 47 (`test_modality_metrics.py` 23, `test_display_names.py` 24) | collect-only |
| Frontend typecheck | `tsc -b`: 0 errors | `npm` scripts |
| Frontend lint | oxlint: 0 errors, 25 warnings (in code that predates this update) | `npm run lint` |
| Frontend build | `vite build`: success | `npm run build` |
| Frontend unit tests | none exist in the project | `package.json` |
| API smoke tests | POST /users 201; empty name 422; the same name twice gives distinct IDs; CORS preflight 200 | live server on a temporary DB |
| Browser test | Headless Chrome with **synthetic** camera and microphone: name validation, registration, camera stream (640×480) → backend NO_FACE (correct), voice enrollment ("Excellent", 4 sets), voice grant/deny, name shown only on grant. **No real face or voice was captured by the auditor.** The audit log shows 2 real face authentications under the new metric by someone else (§3.5). | smoke scripts |

Largest test groups: `test_flexible_auth` 40, `test_voice_enrollment_quality` 36, `test_template_sets` 26,
`test_display_names` 24, `test_modality_metrics` 23, `test_key_continuity` 18, `test_face_quality_gating` 18,
`test_transform` 17, `test_voice_preprocessing` 17, `test_security_validation` 15, `test_audit_logging` 15 (54 files in total).
Most integration tests use **stub embedders** (MTCNN cannot detect faces in synthetic images), so they validate the
wiring, not recognition accuracy.

---

## 7. Dataset audit

| Modality | Dataset | Public | Subjects | Samples | Split | License | Evidence |
|---|---|---|---|---|---|---|---|
| Face (training) | LFW via `sklearn.fetch_lfw_people`, `min_faces_per_person=20` | yes | 62 identities | 3,023 images | NOT AVAILABLE (val_acc 0.913 reported) | UMass non-commercial research | `docs/DATASETS.md`, `PROJECT_REPORT.md` §8.2 |
| Voice (training / eval) | VoxCeleb1 subset (Indian celebrities), Kaggle `gaurav41/...` | yes (subset mirror) | NOT AVAILABLE | 751 test utterances | 70/15/15 (`training_config.json`) | DbCL-1.0 as declared by the mirror; not verified against VGG | `docs/DATASETS.md`, `voice_metrics.csv` |
| Fingerprint | SOCOFing | yes | 600 (dataset) | 6,000 (dataset); 900 evaluated | NOT AVAILABLE | non-commercial academic | `docs/DATASETS.md` |
| Iris | CASIA-Iris-Thousand mirror | licence-gated | — | — | — | **No completed checkpoint; not part of the system** | `PROJECT_REPORT.md` §8.4 |
| Real users | none formally collected | — | NOT AVAILABLE | 54 unlabelled attempts (local only) | — | no consent protocol documented | §3.5 |

Demographics: **NOT AVAILABLE** for all datasets. Augmentation: voice config lists augmentations but
`augmentation_enabled: false`.

---

## 8. Mathematical formulation (only what the code implements)

**Normalization** (every embedder): x̂ = x / ‖x‖₂

**Cosine and Euclidean distance.** For unit vectors, cos(x, y) = x·y and d(x, y) = ‖x − y‖₂ = √(2 − 2cos θ). This
identity is valid here because all embeddings are L2-normalized (`base_embedder.py`) and BioHash re-normalizes.

**BioHash** (§2.3): b = π(1[W x̂ > t]), with W ∈ ℝ^{256×D}, t ~ N(0, (0.5·std(Wx̂))²) and (W, t, π) = HKDF(MASTER_SECRET, context).

**Hamming** (N = 256):
- H(a, b) = Σᵢ 1[aᵢ ≠ bᵢ]
- h(a, b) = 1 − H/N (the implemented "Hamming similarity")

**Calibrated estimates** [B/E]:
- ĉ = h⁻¹(h), with h(c) = 1 − (p₁θ + p₂θ² + p₃θ³)/π, clamped to [−0.2, 1]
- d̂ = √(2 − 2ĉ)

**Decisions:**
- face: ĉ ≥ τ_f = 0.80
- voice: d̂ ≤ τ_v = 0.75
- fingerprint: h ≥ 0.90

**Fusion-scale scores** (higher = better):
- s_face = ĉ_face
- s_voice = 1 − d̂²/2 (= ĉ_voice)
- s_fp = ĉ_fp
- thresholds: t_face = 0.80, t_voice = 1 − τ_v²/2 = 0.71875, t_fp = ĉ(0.90) = 0.9384

**Fusion** over the submitted set M:
- S = (1/|M|) Σ_{m∈M} s_m (equal weights)
- T = (1/|M|) Σ t_m

**Policy** (`fusion/policy.py`):
- ALL_REQUIRED: accept ⇔ ∀m: match_m. S is reported but does not decide.
- WEIGHTED: accept ⇔ S ≥ T ∧ ∀m: s_m ≥ 0.0.
- AT_LEAST_TWO: exactly 3 modalities submitted; accept ⇔ at least 2 match.

**Directions:** cosine ↑ better · Euclidean ↓ better · fusion ↑ better · Hamming distance ↓ better (Hamming similarity ↑ better).

---

## 9. Proposed tables

**Table 1 — System configuration [D]:**

| Parameter | Value |
|---|---|
| Modalities | face and voice (fingerprint optional) |
| Face model / dimension | InceptionResnetV1 (VGGFace2 pretrained, LFW fine-tuned), 512 |
| Voice model / dimension | ECAPA-TDNN (ArcFace-trained, VoxCeleb1 subset), 192 |
| Normalization | L2 |
| Template | BioHash, 256 bits (32 B), HKDF-SHA256 keyed |
| Template sets per user | 4 (1 ACTIVE) |
| Face threshold | estimated cosine ≥ 0.80 |
| Voice threshold | estimated Euclidean distance ≤ 0.75 |
| Fusion | equal-weight mean, ALL_REQUIRED |

**Table 2 — Biometric metrics:**

| Modality | Metric | Direction | Threshold | Hamming equivalent | Runtime representation | Threshold origin |
|---|---|---|---|---|---|---|
| Face | cosine (**estimate**) | ↑ | 0.80 | h ≥ 0.8176 (H ≤ 46) | ĉ from template Hamming | teacher-requested |
| Voice | Euclidean (**estimate**) | ↓ | 0.75 | h ≥ 0.7823 (H ≤ 55) | d̂ = √(2−2ĉ) | teacher-requested |
| Fingerprint | Hamming similarity | ↑ | 0.90 | H ≤ 25 | measured h | inherited default |
| Fusion | mean of fusion-scale scores | ↑ | 0.759375 (face+voice) | — | S | derived; not decisive under ALL_REQUIRED |

**Table 3 — Calibration results [B]:** the §4 table.

**Table 4 — Authentication results:** **NOT AVAILABLE** for the deployed system. A real-dataset table of the embedding
models (voice [A], face [A], clearly labelled "raw embedding, not protected template") is possible now.

**Table 5 — Software validation [C]:** the §6 table.

**Table 6 — Storage / template characteristics [D/E]:**

| Item | Value |
|---|---|
| Per template | 32 bytes plus row metadata (21 columns) |
| Per user, face + voice, 4 sets | 4 × 2 × 32 B = 256 B of template payload; revoked rows are retained with status REVOKED |
| Also stored | display name, internal ID, key/template/set versions, audit rows with per-modality scores, a hash fingerprint of `MASTER_SECRET` |
| Not stored | raw images, raw audio, embeddings |

**Table 7 — Comparison with existing approaches:** qualitative only (design features: modality set, template protection
used, revocability). **No numeric comparison**, because there is no comparable evaluation of the deployed system.

---

## 10. Proposed figures

| # | Title | x / y | Data | Exists? | Generate now? |
|---|---|---|---|---|---|
| 1 | System architecture | — | §2 | yes | yes |
| 2 | Face / voice pipelines | — | §2.1–2.2 | yes | yes |
| 3 | BioHash generation | — | §2.3 | yes | yes |
| 4 | Fusion and decision | — | §8 | yes | yes |
| 5 | Calibration: embedding cosine vs Hamming similarity (means ± SD, fitted curve; face and voice) | cos / h | `biohash_metric_calibration.json` | yes | yes |
| 6 | Voice: Hamming similarity vs estimated Euclidean distance, with the 0.75 line | h / d̂ | same JSON + d = √(2−2c) | yes | yes |
| 6b | Single-estimate SD vs cosine | cos / SD | JSON `cosine_estimate_std` | yes | yes |
| 7 | Score distributions, genuine vs impostor (deployed system) | score / density | **NOT AVAILABLE** | no | no |
| 8 | ROC / DET of the deployed system | FAR / TAR | **NOT AVAILABLE** | no | no |
| 8a | ROC of the voice **embedding model** (raw) | FPR / TPR | `voice_roc.csv` | yes | yes, if labelled "raw embedding" |
| 9 | Template-set unlinkability histogram | h | only summary statistics in `template_set_diversity.csv`; per-pair data needs a rerun of `evaluation.template_set_experiments` | partial | after a rerun |

---

## 11. Claims audit

**A. Safe to claim**
- The system uses 256-bit keyed BioHash cancelable templates (HKDF-SHA256 keys, orthonormal projection, keyed
  quantization and permutation).
- Raw face images, raw voice audio and embeddings are not stored. Only templates plus metadata are.
- Face and voice are fused under an ALL_REQUIRED policy with equal-weight score fusion.
- Templates are revocable through a pool of 4 keyed template sets. On synthetic embeddings, templates of the same
  embedding under different keys give h = 0.500 ± 0.029 (360 pairs, 20 users). Revocation drops the revoked set's
  similarity to 0.504 [B].
- Display names are separate from internal identifiers.
- 559 automated backend tests pass [C].
- The voice embedding model achieves EER 2.29 % / AUC 0.9967 on a held-out VoxCeleb1-subset split, **at the raw
  embedding level** [A].

**B. Claims that need qualification**
- "Face similarity is measured by cosine similarity": say that it is a **calibrated estimate derived from the template
  Hamming comparison**, with SD ≈ 0.05 near the threshold.
- "Voice uses Euclidean distance": same wording, as an **estimated** distance.
- "The calibration is accurate (R² 0.9999)": say this is **fit quality on synthetic vectors**, justified by the
  transform's rotation invariance, and not validated on real embedding pairs.
- "Unlinkability / revocability": say this was **observed on synthetic embeddings**, with no formal unlinkability
  metric.
- "Face EER 0.010": say it is a training-log summary on LFW, the protocol is unverified, and it is raw-embedding level.
- The 128→256-bit improvement: documented, but its data is not committed.

**C. Must not claim**
- Any accuracy, FAR, FRR, EER, AUC or TAR **of the deployed system**.
- "Exact cosine similarity" or "exact Euclidean distance" at runtime.
- That the 0.80 / 0.75 thresholds are optimal, calibrated or validated.
- Irreversibility, "secure against attacks", "zero privacy risk", spoof or deepfake resistance.
- Any number of test users or subjects in real-user trials.
- Suitability or certification for critical infrastructure. Present it as design motivation only.
- That 559 tests represent biometric trials.

---

## 12. Limitations

1. No labelled real-user evaluation of the protected-template system; FAR/FRR/EER unknown.
2. Calibration is synthetic. The rotation-invariance argument is sound, but it has not been validated on real embedding
   pairs.
3. Single-attempt estimate uncertainty is about ±0.05 in cosine at the face threshold, a 256-bit quantization effect.
4. Thresholds are teacher-requested, not tuned. The face rule (h ≥ 0.818) is stricter than the previous 0.80 rule, and
   past real face scores clustered near 0.78–0.87.
5. No presentation-attack / liveness detection; spoofing and deepfake robustness not evaluated.
6. No formal security analysis (stolen-key scenario, inversion or hill-climbing attacks). BioHash security relies on
   the secrecy of `MASTER_SECRET`, and literature shows BioHash degrades when the token or key is compromised.
7. Voice: a 192-d embedding into 256 bits uses non-orthogonal projection blocks. Training used a limited subset
   (speaker count unknown).
8. Face model evaluation is closed-set LFW with an unverified protocol; fingerprint is weak (EER 30.8 %).
9. No demographic or bias analysis; no robustness study (lighting, camera, microphone, noise).
10. No latency or throughput benchmark; only incidental, cold-start-dominated audit latencies on unknown hardware.
11. Several docs are stale (§15).

---

## 13. System performance

- **Benchmarks:** none in the repository. There is no timing of the embedding, BioHash or comparison stages.
- **Incidental end-to-end latency** from `audit_logs.latency_ms`: server-side, from request receipt to decision,
  including model load on cold start. Hardware is **NOT AVAILABLE**.
  - R1, 50 real attempts: median 1,808 ms, IQR 1,191–3,071 ms, mean 3,945 ms, SD 6,633 ms, p95 19,329 ms,
    min 316 ms, max 30,696 ms.
  - R2, 2 real face attempts: 399 ms and 1,269 ms.
  - Multi-modality requests deliberately release model caches (`authentication.release_modality_cache`, for 512 MB
    hosts), which inflates latency.
- **Throughput, memory, CPU:** **NOT AVAILABLE**.
- **Storage:** see Table 6 (legitimately computed).

---

## 14. Research contribution analysis

**Implemented contributions (defensible):**
1. An end-to-end, open, reproducible multimodal (face + voice) authentication system in which **only** 256-bit
   cancelable templates are stored.
2. **Metric-specific decision rules on protected templates.** Face cosine and voice Euclidean decisions are expressed
   through an offline-calibrated mapping from template Hamming similarity, and fused on one direction-consistent scale.
   This is **an engineering method, and it needs literature positioning**: SimHash/random-hyperplane angle estimation
   is classical, so claim the *application and calibration within a keyed BioHash*, not the idea itself.
3. Revocation of whole multimodal credentials through keyed template-set pools.
4. A usability-oriented registration design that separates display name from identifier.

**Novelty claims to avoid** unless a literature search supports them: "first", "novel fusion", "novel BioHash".
Suggested wording: "we implement and evaluate…", "we adapt…", "to our knowledge, few open implementations…" (only if
supported by the search).

---

## 15. Documentation vs implementation discrepancies

| # | Document | Says | Implementation / data says |
|---|---|---|---|
| 1 | `docs/PROJECT_REPORT.md` §8.2 | fingerprint EER 0.445, AUC 0.579 | `fingerprint_metrics.csv`: EER 0.3077, AUC 0.7644 (later checkpoint, commit 3a0bfeb) |
| 2 | `docs/PROJECT_REPORT.md` | 24 tests; Phase 2/3 "not yet implemented" | 559 tests; template protection, fusion and UI implemented |
| 3 | README "The problem" | face, **iris**, fingerprint | the application uses face, voice and fingerprint; iris has no completed checkpoint |
| 4 | Real-user face range | 0.738–0.820 (deleted `face_threshold.json` note) | the same report elsewhere says 0.738–0.859 over 15 attempts; the source log is not in the repo |
| 5 | `PROJECT_DOCUMENTATION.md`, `VIVA_TEMPLATE_PROTECTION_PREP.md` | Hamming 0.80 thresholds | superseded (banner added in `873c13e`) |
| 6 | `template_protection/biohash.py` | `DEFAULT_OUTPUT_BITS = 128` | runtime uses 256 (`Settings.template_bits`); cosmetic, but state 256 in the paper |
| 8 | `scripts/calibrate_biohash_metric_mapping.py::MODALITY_DIMS`, `evaluation/template_set_experiments.py` docstring | fingerprint 256-d | the fingerprint embedder is 512-d. Only fingerprint's fusion-scale score is affected (its decision is h ≥ 0.90); the D=512 face curve is nearly identical. Fix: set 512, rerun the calibration (~15 s), rerun tests |
| 7 | `docs/BIOMETRIC_METRICS.md` | "about ±0.05 near the thresholds" | 0.050 at c = 0.80; 0.057–0.063 at c ≈ 0.72 (voice threshold); still accurate as "about" |

---

## 16. Recommended paper structure (IEEE)

1. Abstract
2. Introduction (motivation, unimodal limits, template-protection need, contributions)
3. Related work (§17 categories)
4. Threat model and design goals (privacy of stored templates, revocability; out of scope: liveness)
5. System architecture
6. Face and voice pipelines
7. Cancelable protection (BioHash, HKDF, template sets)
8. Metric-specific decisions on protected templates and calibration
9. Fusion and decision policy
10. Implementation and user workflow (display names)
11. Experimental setup: A calibration, B embedding-model evaluation, C template-set experiments, D software testing,
    E real-user evaluation (only if done)
12. Results
13. Discussion
14. Security and privacy considerations
15. Limitations
16. Conclusion and future work
17. References

### Abstract ingredients (verified status in brackets)
- **Problem:** stored biometric templates are irrevocable [context].
- **System:** face + voice, 256-bit BioHash, HKDF keys [D].
- **Decisions and fusion:** estimated cosine ≥ 0.80 and estimated distance ≤ 0.75, fused under ALL_REQUIRED [D].
- **Calibration:** R² ≥ 0.9999 on 19,600 synthetic pairs per modality, single-estimate SD ≈ 0.05 [B].
- **Voice model:** EER 2.29 % at the raw-embedding level [A].
- **Unlinkability:** cross-key similarity 0.500 on synthetic embeddings [B].
- **Software:** 559 tests [C].
- **Limitation:** no deployed-system FAR/FRR [statement].

---

## 17. Recommended references

These are cited from knowledge. **Verify every entry (DOI or URL) before submission.** None were fetched during this
audit.

| Category | Reference | Why |
|---|---|---|
| Biometrics foundations | A. K. Jain, A. Ross, S. Prabhakar, "An introduction to biometric recognition," IEEE TCSVT 14(1):4–20, 2004, doi:10.1109/TCSVT.2003.818349 | framing |
| Multimodal / fusion | A. Ross, A. K. Jain, "Information fusion in biometrics," Pattern Recognition Letters 24(13):2115–2125, 2003, doi:10.1016/S0167-8655(03)00079-5 | fusion levels |
| Fusion | A. Jain, K. Nandakumar, A. Ross, "Score normalization in multimodal biometric systems," Pattern Recognition 38(12):2270–2285, 2005, doi:10.1016/j.patcog.2005.01.012 | score normalization and sum rule |
| Fusion | J. Kittler et al., "On combining classifiers," IEEE TPAMI 20(3):226–239, 1998, doi:10.1109/34.667881 | sum rule |
| Fusion | K. Nandakumar et al., "Likelihood ratio-based biometric score fusion," IEEE TPAMI 30(2):342–347, 2008, doi:10.1109/TPAMI.2007.70796 | alternative fusion (future work) |
| Multibiometrics | A. Ross, K. Nandakumar, A. K. Jain, *Handbook of Multibiometrics*, Springer, 2006, doi:10.1007/0-387-33123-9 | background |
| Cancelable biometrics | N. K. Ratha, J. H. Connell, R. M. Bolle, "Enhancing security and privacy in biometrics-based authentication systems," IBM Systems Journal 40(3):614–634, 2001, doi:10.1147/sj.403.0614 | origin of cancelable biometrics |
| Cancelable biometrics | N. K. Ratha et al., "Generating cancelable fingerprint templates," IEEE TPAMI 29(4):561–572, 2007, doi:10.1109/TPAMI.2007.1004 | transforms |
| Cancelable review | V. M. Patel, N. K. Ratha, R. Chellappa, "Cancelable biometrics: A review," IEEE Signal Processing Magazine 32(5):54–65, 2015, doi:10.1109/MSP.2015.2434151 | survey |
| Template protection | A. K. Jain, K. Nandakumar, A. Nagar, "Biometric template security," EURASIP J. Adv. Signal Process. 2008:579416, doi:10.1155/2008/579416 | requirements (revocability, non-invertibility) |
| Template protection | K. Nandakumar, A. K. Jain, "Biometric template protection: Bridging the performance gap between theory and practice," IEEE SPM 32(5):88–100, 2015, doi:10.1109/MSP.2015.2427849 | performance/security trade-off |
| Template protection survey | C. Rathgeb, A. Uhl, "A survey on biometric cryptosystems and cancelable biometrics," EURASIP J. Info. Security 2011:3, doi:10.1186/1687-417X-2011-3 | survey |
| BioHash | A. T. B. Jin, D. N. C. Ling, A. Goh, "Biohashing: two factor authentication featuring fingerprint data and tokenised random numbers," Pattern Recognition 37(11):2245–2255, 2004, doi:10.1016/j.patcog.2004.04.011 | original BioHash |
| BioHash analysis | A. Kong et al., "An analysis of BioHashing and its variants," Pattern Recognition 39(7):1359–1368, 2006, doi:10.1016/j.patcog.2005.10.025 | stolen-token weakness (limitation) |
| BioHash | A. B. J. Teoh, Y. W. Kuan, S. Lee, "Cancellable biometrics and annotations on BioHash," Pattern Recognition 41(6):2034–2044, 2008, doi:10.1016/j.patcog.2007.12.002 | BioHash properties |
| BioHash | A. Lumini, L. Nanni, "An improved BioHashing for human authentication," Pattern Recognition 40(3):1057–1065, 2007, doi:10.1016/j.patcog.2006.05.030 | variants |
| Unlinkability | M. Gomez-Barrero et al., "General framework to evaluate unlinkability in biometric template protection systems," IEEE TIFS 13(6):1406–1420, 2018, doi:10.1109/TIFS.2017.2788000 | metric for future work |
| Binary hashing / angle | M. S. Charikar, "Similarity estimation techniques from rounding algorithms," STOC 2002, doi:10.1145/509907.509965 | P(bit differs) = θ/π (basis of the calibration) |
| Random hyperplane | M. X. Goemans, D. P. Williamson, JACM 42(6):1115–1145, 1995, doi:10.1145/227683.227684 | same principle |
| Face detection | K. Zhang et al., "Joint face detection and alignment using multitask cascaded convolutional networks," IEEE SPL 23(10):1499–1503, 2016, doi:10.1109/LSP.2016.2603342 | MTCNN |
| Face embedding | F. Schroff, D. Kalenichenko, J. Philbin, "FaceNet," CVPR 2015, doi:10.1109/CVPR.2015.7298682 | embedding and L2 metric |
| Face backbone | C. Szegedy et al., "Inception-v4, Inception-ResNet…," AAAI 2017, doi:10.1609/aaai.v31i1.11231 | InceptionResNet |
| Face training data | Q. Cao et al., "VGGFace2," IEEE FG 2018, doi:10.1109/FG.2018.00020 | pretraining |
| Face loss | J. Deng et al., "ArcFace," CVPR 2019, doi:10.1109/CVPR.2019.00482 | loss used |
| Face dataset | G. B. Huang et al., "Labeled Faces in the Wild," UMass TR 07-49, 2007, http://vis-www.cs.umass.edu/lfw/ | dataset |
| Speaker embedding | B. Desplanques, J. Thienpondt, K. Demuynck, "ECAPA-TDNN," Interspeech 2020, doi:10.21437/Interspeech.2020-2650 | voice model |
| Speaker embedding | D. Snyder et al., "X-vectors," ICASSP 2018, doi:10.1109/ICASSP.2018.8461375 | background |
| Speaker dataset | A. Nagrani, J. S. Chung, A. Zisserman, "VoxCeleb," Interspeech 2017, doi:10.21437/Interspeech.2017-950 | dataset |
| Toolkit | M. Ravanelli et al., "SpeechBrain," arXiv:2106.04624, 2021 | implementation |
| Spoofing (voice) | M. Todisco et al., "ASVspoof 2019," Interspeech 2019, doi:10.21437/Interspeech.2019-2249 | limitation / future |
| Spoofing (general) | S. Marcel et al. (eds.), *Handbook of Biometric Anti-Spoofing*, 2nd ed., Springer, 2019, doi:10.1007/978-3-319-92627-8 | PAD |
| Standards | ISO/IEC 24745:2022 (biometric information protection); ISO/IEC 30107-3 (PAD testing); ISO/IEC 19795-1 (performance testing) | evaluation protocol |
| Key derivation | H. Krawczyk, P. Eronen, RFC 5869 (HKDF), 2010, https://www.rfc-editor.org/rfc/rfc5869 | key derivation |
| Identity guidelines | NIST SP 800-63B, doi:10.6028/NIST.SP.800-63b | biometric authenticator requirements |
| Critical infrastructure | NIST SP 800-82 Rev. 3, 2023, doi:10.6028/NIST.SP.800-82r3 | context only |
| Bias | P. Grother, M. Ngan, K. Hanaoka, NISTIR 8280, 2019, doi:10.6028/NIST.IR.8280 | demographic-effects limitation |
| Fingerprint dataset | Y. I. Shehu et al., "SOCOFing," arXiv:1807.10609, 2018 | dataset |

---

## 18. Missing experiments / data and paper-readiness assessment

**Readiness:** a system/implementation paper can be written now if it stays within §11 A–B. It is **not ready** for
claims of authentication performance. Priority 1–2 below would move it to an evaluation paper.

# WHAT WE STILL NEED TO MEASURE

**1. Real-user genuine/impostor evaluation of the deployed protected-template system (highest priority).**
- **Why:** it is the only way to report FAR/FRR/EER/ROC of what was built.
- **Data:** consented participants (document consent and data deletion). Each person enrolls once: 5 face poses and
  2 voice recordings. Then ≥ 10 genuine face and ≥ 10 genuine voice attempts across 2 sessions on different days.
  Recommended ≥ 30 people (minimum 20).
- **Protocol:**
  - Genuine: each person's own attempts against their own template.
  - Impostor: each person's attempts against every other person's template **under the target's key**, which is what
    the real system does (`scripts/calibrate_protected_thresholds.py` documents this pairing).
  - Record the Hamming similarity per comparison and never store raw embeddings beyond the offline research harness.
  - Volume: 30 × 10 = 300 genuine and 30 × 10 × 29 = 8,700 impostor comparisons per modality.
- **Metrics:** FAR, FRR, TAR at 0.80/0.75 and at EER; EER; ROC-AUC; the same for face-only, voice-only and fusion
  (ALL_REQUIRED and WEIGHTED); bootstrap 95 % CIs.
- **Graphs:** genuine/impostor histograms (Hamming and estimated metric), ROC, DET, FAR/FRR vs threshold.
- **Tables:** a per-modality and per-fusion results table.
- **Files:** `evaluation/real_user_evaluation.py` → `evaluation/results/real_user_scores.csv` (anonymized IDs, modality,
  genuine flag, h, ĉ/d̂) and `real_user_metrics.csv`.

**2. Real-embedding validation of the calibration (can be done now, with no participants).**
- **Why:** it confirms the synthetic rotation-invariance assumption on real embeddings.
- **Data:** LFW (face, 512-d, via the repo checkpoint) and the VoxCeleb subset (voice, 192-d): all pairs, or ≥ 20,000
  sampled pairs, with the true cosine computed offline.
- **Protocol:** template both embeddings under the same random key and compare the true cosine with ĉ from
  `metric_estimation`.
- **Metrics:** RMSE, MAE, bias and SD of (ĉ − c) binned by c; the same for d̂.
- **Graphs:** ĉ vs c scatter with the identity line; error vs c.
- **Files:** `scripts/validate_calibration_real.py` → `evaluation/results/calibration_real_validation.csv`.

**3. Protected-template vs raw-embedding separability on public datasets (can be done now).**
- **Why:** it quantifies the accuracy cost of template protection.
- **Data:** LFW and the VoxCeleb subset test split.
- **Metrics:** EER and AUC for raw cosine vs 256-bit h (same key for genuine, target key for impostors).
- **Graphs:** ROC overlay.
- **Files:** `evaluation/results/{face,voice}_protected_vs_raw.csv`. This also puts the currently doc-only 128-vs-256
  table on committed data.

**4. Threshold sensitivity.**
- **What:** FAR/FRR as the face threshold varies over 0.70–0.90 and the voice threshold over 0.55–1.0 (from #1 or #3).
- **Output:** curves plus a table at the teacher values. Save to `evaluation/results/threshold_sensitivity.csv`.

**5. Latency benchmark.**
- **Protocol:** 100 warm and 10 cold runs per stage (preprocess, embed, BioHash, compare, full request), recording the
  hardware (CPU, RAM, OS) and single vs multi-modality requests.
- **Metrics:** mean, median, SD, p95.
- **Script:** `scripts/benchmark_latency.py` → `evaluation/results/latency_benchmark.csv`.

**6. Unlinkability and revocability with real embeddings.**
- **Protocol:** Gomez-Barrero D↔ / D_sys on mated vs non-mated cross-key template pairs from #3. Revoked-vs-new
  similarity distribution.
- **Output:** `evaluation/results/unlinkability_real.csv` plus a histogram.

**7. Presentation-attack testing (ISO/IEC 30107-3).**
- **Attacks:** printed photo, screen replay and video for face; replayed recording and TTS/voice-clone for voice;
  ≥ 10 attacks per species per participant subset.
- **Metrics:** APCER, BPCER.
- **Output:** `evaluation/results/pad_results.csv`. Expect poor results, since no liveness check exists; report
  honestly.

**8. Stolen-key scenario.**
- **Protocol:** an attacker holding `MASTER_SECRET` or the user's key: impostor Hamming distribution when the attacker
  uses the target key (the #1 impostor protocol already models this), plus the known BioHash stolen-token analyses.
- **Output:** a discussion plus the #1 numbers.

**9. Robustness.**
- **Face:** lighting levels and camera distance.
- **Voice:** SNR 20/10/5 dB noise and two microphones.
- **Metric:** genuine-score shift and FRR at the thresholds.
- **Output:** `evaluation/results/robustness.csv`.

**10. Demographic analysis.**
- Only with explicit consent and a sufficient sample; otherwise state it as a limitation.

**Also:** commit the face evaluation's pair-level data (ROC CSV), the split protocol and the speaker count for voice.
Keep the data-deletion and consent record with the study.
