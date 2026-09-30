# Paper Source of Truth

> **Modality change (2026-09-27).** Iris and fingerprint have been removed from the system: the implemented
> modalities are **face and voice**, and the `AT_LEAST_TWO` fusion policy (which needed three modalities) is gone.
> This document is a dated record. Its iris / fingerprint material (models, datasets, results, thresholds,
> three-modality fusion) is kept for history only and no longer describes the code. Current state: `README.md`,
> `docs/ROADMAP.md`; current numbers: `evaluation/PAPER_RESULTS_INDEX.md`.

**Project:** Cancelable Multimodal Biometric Authentication for Critical Infrastructure
**Repository state:** `main` @ `873c13e` plus the uncommitted evaluation package in `evaluation/ieee/` (2026-09-25)
**Purpose:** the only reference for writing the IEEE paper. It is documentation, not paper text.
**Updated:** 2026-09-25 after the master audit (`docs/MASTER_PROJECT_AUDIT.md`). Every paper number is indexed in `evaluation/PAPER_RESULTS_INDEX.md`, and the status of each change is in §16.

**Rules used in this document**
- Every statement cites the source file / function it was read from. Where code and documentation disagree, the **code wins**, and the disagreement is listed.
- Numbers come only from code, configuration, committed result files, or the reproducible evaluation package (`evaluation/IEEE_EVALUATION_PACKAGE.md`). Anything else is **NOT AVAILABLE**.
- Evidence labels: **REAL DATA** · **SYNTHETIC** · **SOFTWARE TEST** · **CONFIGURATION** · **DERIVED** · **FUTURE**.

---

## 1. Project overview

**Final title:** *Cancelable Multimodal Biometric Authentication for Critical Infrastructure* (`README.md`).

**Problem statement.** A biometric characteristic cannot be changed after a breach. A conventional system that stores face/voice/fingerprint templates (raw images or unprotected embeddings) turns a database compromise into a permanent identity exposure. Combining several modalities improves verification reliability, but it multiplies the amount of sensitive biometric data held unless storage is designed around that risk (`README.md` "The problem").

**What the system does.** It enrolls face and voice (fingerprint optional) and maps each embedding through a keyed, cancelable transform (BioHash) to a 256-bit template. It stores **only** those templates and authenticates by comparing templates with Hamming similarity. Each modality's decision is expressed as a calibrated estimate of the embedding metric (face: cosine, voice: Euclidean). Decisions are fused under a configurable policy (default: every presented factor must pass). A user's credential is a pool of four keyed template sets, which can be revoked and replaced (`backend/services/base_service.py`, `template_protection/`, `fusion/`, `backend/database/crud.py`).

**Motivation for critical infrastructure.** Access to high-assurance facilities (the demo models buildings with clearance levels, `config/buildings.json`) needs:
- strong identity assurance, which motivates multiple factors;
- revocable credentials, because a leaked credential must be replaceable;
- no stored raw biometric data, which limits breach impact.

*Scope note:* buildings are session context only; they impose no biometric policy (`backend/buildings.py`, `docs/MULTI_TEMPLATE_ARCHITECTURE.md`). No claim of suitability for real critical-infrastructure deployment is supported (§14).

**Why multimodal.**
- A user unable to present one factor can present another; the user chooses among enrolled factors (`backend/services/authentication.py`).
- An attacker must defeat every presented modality under `ALL_REQUIRED` (`fusion/config.py`).
- Measured: score-level fusion of face+voice gives EER **1.67 %**, against 2.75 % for the best single modality, on chimeric users (**REAL DATA**, `evaluation/results/fusion_score_level_eer.csv`).

**Why cancelable biometrics.**
- Stored templates are keyed transforms, so a compromised template can be revoked. The same biometric under a new key yields an unrelated template: the revoked template vs the new one gives mean Hamming similarity **0.500** and 0 % acceptance (**REAL DATA**, `evaluation/results/revocation_summary.csv`).
- Templates of the same person under different application keys are not linkable beyond the estimator's noise floor for voice and fingerprint (face: marginally above it; §10).

---

## 2. Complete system architecture

```
Browser (React SPA: frontend/src)
  |  HTTPS multipart (PNG/JPEG/BMP images, WAV audio); JSON for /users
  v
FastAPI backend (backend/main.py; routes in backend/api/*.py)
  |  decode + validate upload (backend/utils.py::decode_biometric_sample; <= 5,000,000 bytes)
  v
Preprocessing (preprocessing/face.py | voice.py | fingerprint.py)
  v
Embedding model (models/*/inference.py via embeddings/pipelines.py) -> L2-normalized float vector
  v
BioHash template protection (template_protection/biohash.py::generate_template, keys from hkdf_keys.py)
  v
Protected template storage (backend/database/crud.py::save_modality_templates -> SQLite BLOB, 32 bytes)
  v
Matching (template_protection/matcher.py::compare, Hamming similarity)
  v
Per-modality decision (backend/services/modality_metrics.py::decide)
  v
Fusion (fusion/policy.py::evaluate_fusion_policy)
  v
Decision + audit row (backend/services/authentication.py; backend/utils.py::record_authentication_audit)
```

| Stage | Source file · function | Input | Output | Dimension / type |
|---|---|---|---|---|
| Capture | `frontend/src/components/capture/GuidedFaceCapture.tsx`, `VoiceCapture.tsx`, `FingerprintCapture.tsx` | camera / microphone / file | PNG / WAV / image file | — |
| API | `backend/api/fusion.py`, `enroll.py`, `user.py` | multipart form | JSON decision | — |
| Decode | `backend/utils.py::decode_biometric_sample` | upload | `np.ndarray` RGB uint8, or `(waveform float, sample_rate)` | H×W×3 / 1-D |
| Face preprocess | `preprocessing/face.py::FacePreprocessor.preprocess` | RGB image | aligned crop | 160×160×3 float |
| Voice preprocess | `preprocessing/voice.py::VoicePreprocessor.preprocess` | waveform, rate | log-mel | 80 × frames (4.0 s; hop 160) |
| Fingerprint preprocess | `preprocessing/fingerprint.py::FingerprintPreprocessor.preprocess` | RGB image | enhanced, ImageNet-normalized | 224×224×3 float32 |
| Embedding | `models/{face,voice,fingerprint}/inference.py::*Embedder.extract_embedding` | preprocessed array | unit vector | 512 / 192 / 512, float32 |
| Keys | `template_protection/hkdf_keys.py::derive_key` | MASTER_SECRET, app, user, modality, key_version | 3 × 32-byte seeds | bytes |
| Template | `template_protection/biohash.py::generate_template` | embedding + key | bit vector | 256 × uint8 {0,1} |
| Store | `backend/database/crud.py::save_modality_templates` via `base_service.py::ModalityService._store` | 4 templates / modality | rows; `pack_bits` → 32 B BLOB | 256 bits |
| Match | `template_protection/matcher.py::compare(metric="hamming")` | two bit vectors | fraction of equal bits h ∈ [0,1] | float |
| Decide | `backend/services/modality_metrics.py::decide` | h, template bits | estimate, match, fusion score | float, bool |
| Fuse | `fusion/policy.py::evaluate_fusion_policy` | per-modality scores + matches | `FusionDecision` | bool, float |
| Respond | `backend/services/authentication.py::to_public_response` | outcome | `AuthenticationDecision` (+ display name on success) | JSON |

**API endpoints** (26, extracted from the live app, `evaluation/results/api_endpoints.csv`):
- **Enrollment:** `POST /enroll`, `/enroll/face/check-pose`
- **Authentication:** `POST /authenticate`, `/authenticate/fusion`, `/verify/{face,iris,fingerprint,voice}`
- **Template sets:** `POST /revoke-template`, `GET /templates/{user_id}`, `POST /templates/{user_id}/activate/{version}`, `/replenish`, `/generate`
- **Users:** `POST /users`, `POST /user/{user_id}/display-name`, `GET|DELETE /user/{user_id}`, `GET /user/{user_id}/enrollment-status`
- **Buildings:** `GET /buildings`, `/building/{id}`
- **Metrics:** `GET /metrics/{modality}`
- **Audit:** `GET /audit/system`, `GET|DELETE /audit/{user_id}`
- **Health:** `GET /system/health`, `/health`

---

## 3. Modality pipelines

### 3.1 Face

**Model** (`models/face/inference.py`; training `kaggle_kernels/face_training/face-embedding-training.ipynb`)

| Item | Value | Source |
|---|---|---|
| Architecture | InceptionResnetV1 (facenet-pytorch) | `models/face/inference.py::FaceEmbedder._build_model` |
| Pretrained backbone | VGGFace2 (`pretrained='vggface2'`) | notebook cell 10 |
| Fine-tuning dataset | LFW, `fetch_lfw_people(min_faces_per_person=20, color=True, funneled=True)` | notebook cell 4 |
| Loss | ArcFace (`ArcMarginProduct`, s = 30.0, m = 0.50) + CrossEntropy | `models/common/arcface.py`; notebook cell 10 |
| Trainable layers | `block8`, `last_linear`, `last_bn` (rest frozen) + ArcFace head | notebook cell 10 |
| Embedding size | 512 | `FACE_EMBEDDING_DIM` |
| Normalization | L2 (`_l2_normalize`) | `models/common/base_embedder.py` |
| Checkpoint | `models/face/saved/face_embedder.pt` (+ `.h5` export), Git LFS, ~112 MB | notebook cell 14; `.gitattributes` |

**Kaggle training**

| Item | Value | Source |
|---|---|---|
| Notebook | `kaggle_kernels/face_training/face-embedding-training.ipynb` (Colab twin `notebooks/01_face_training_and_testing.ipynb`) | — |
| Epochs | 10 | cell 12 `NUM_EPOCHS = 10` |
| Optimizer | Adam, lr = 1e-4 | cell 10 |
| Batch size | 32 | cell 12 |
| Scheduler | none | cell 12 |
| Split | per-identity 70 / 15 / 15, seed 42 | cell 8 |
| Input normalization | (x − 127.5) / 128 | cell 12 |
| Augmentations | none | cell 12 |
| Hardware | Kaggle kernel with GPU enabled (`kernel-metadata.json`); `docs/PROJECT_REPORT.md` §8.3 reports the completed runs fell back to **CPU** | — |
| Final epoch | train_loss 0.1378, train_acc 0.976, **val_acc 0.913** (10/10) | `docs/PROJECT_REPORT.md` §8.2 (training log; not re-verifiable) |
| Training duration | NOT AVAILABLE | — |
| Test metrics (identities overlap training) | accuracy@EER 0.990, EER 0.010, AUC 0.999 | `evaluation/results/face_metrics.csv` (NOT VERIFIABLE, no pair data) |
| Held-out-identity evaluation | EER **7.12 %** raw / **8.50 %** protected (1,618 LFW identities with 2–19 images, disjoint from fine-tuning) | `evaluation/results/raw_vs_protected_metrics.csv` (**REAL DATA**) |

**Preprocessing** (`preprocessing/face.py`)

| Item | Value |
|---|---|
| Detector | MTCNN (facenet-pytorch), `image_size=160, margin=0, post_process=True` (`_get_detector`) |
| Crop | 160×160 (`FACE_INPUT_SIZE`) |
| Alignment | **Deployed default (`FACE_ALIGNMENT=bbox`, FACE_BASELINE):** bounding-box crop and resize; the 5-point landmarks are only quality signals. **Optional (`FACE_ALIGNMENT=similarity`, FACE_ALIGNED):** 5-landmark Umeyama similarity transform onto `ALIGNMENT_TEMPLATE_160` (`FacePreprocessorAligned`), IMPLEMENTED + EVALUATED, not deployed; see `evaluation/reports/FACE_ALIGNMENT_DECISION.md` |
| Blur detection | Laplacian variance ≥ 25.0 (`BLUR_MIN_SHARPNESS`) |
| Detection confidence | ≥ 0.90 (`MIN_DETECTION_CONFIDENCE`) |
| Face size | ≥ 0.15 of frame (`MIN_FACE_SIZE_RATIO`) |
| Centring | offset ≤ 0.35 (`MAX_CENTER_OFFSET`) |
| Pose | roll ≤ 20° (`MAX_ROLL_DEGREES`), yaw ratio ≤ 0.20 (`MAX_YAW_RATIO`) |
| Multi-face | low-confidence detections ignored when counting faces (commit `ec7e0e1`); `MultipleFacesDetected` |

**Enrollment** (`backend/api/enroll.py` → `ModalityService.enroll_poses`)
1. The UI guides 5 captures (front, left, right, up, down; `backend/services/face_enrollment.py::FACE_POSES`). Each capture can be checked live via `POST /enroll/face/check-pose`.
2. Each capture goes through detection, the quality gates and embedding (`embeddings/pipelines.py::FacePipeline.embed_poses`). Rejected poses are reported.
3. At least 3 valid poses are required (`MIN_VALID_POSES`); otherwise `FaceCaptureRejected` is raised and nothing is stored.
4. Centroid = normalize(mean of valid embeddings) (`embeddings/centroid.py`). The per-pose embeddings are discarded.
5. For each of the 4 template sets: derive an HKDF key (key_version v…v+3), run BioHash, and store the 256-bit template (`_store`).

**Authentication** (`ModalityService.authenticate`)
1. Embed the live capture.
2. Load the claimed user's ACTIVE-set face row and derive the key for its key_version.
3. BioHash the live embedding, compare Hamming similarity h with the stored template, and pass through `security_validation.validate_authentication`.
4. `decide`: estimated cosine ĉ ≥ 0.80.

### 3.2 Voice

**Model** (`models/voice/model.py`, `models/voice/inference.py`, `models/voice/config.py`)

| Item | Value | Source |
|---|---|---|
| Architecture | SpeechBrain `ECAPA_TDNN(input_size=80, lin_neurons=192)` | `models/voice/model.py::VoiceEmbeddingNet` |
| Pretraining | none: fine-tuned "from scratch" on the subset | `docs/VOICE_MODEL.md` ("a from-scratch fine-tune") |
| Input features | 80-bin log-mel, stacked to 3 identical channels to satisfy `BaseEmbedder` | `models/voice/inference.py` docstring |
| Embedding | 192, L2-normalized | `VOICE_EMBEDDING_DIM`; `base_embedder.py` |
| ArcFace | margin 0.50, scale 30.0 | `VoiceConfig.arcface_margin / arcface_scale` |
| Checkpoint | `models/voice/saved/voice_embedder.pt` (~25 MB, LFS) + `training_config.json` | — |

**Kaggle training**

| Item | Value | Source |
|---|---|---|
| Notebook | `kaggle_kernels/voice_training/voice-embedding-training.ipynb` | — |
| Dataset | Kaggle `gaurav41/voxceleb1-audio-wav-files-for-india-celebrity` | `kernel-metadata.json` |
| Speakers / utterances | **24 / 4,857** | `docs/VOICE_MODEL.md`; re-counted in this evaluation |
| Split | per-speaker 70/15/15, seed 42: **3,389 / 717 / 751** (speakers overlap across splits: closed-set) | `models/voice/dataset.py::VoxCelebDataset`; `docs/VOICE_MODEL.md` |
| Epochs | 30 (all completed) | `VoiceConfig.num_epochs`; `docs/VOICE_MODEL.md` |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 | `models/voice/train.py`; `VoiceConfig` |
| Scheduler | CosineAnnealingLR(T_max = 30) | `models/voice/train.py` |
| Batch size | 64 | `VoiceConfig` |
| Early stopping | patience 5 on validation loss; best = `voice_embedder_best.pt` | `models/voice/train.py` |
| Mixed precision | configured; active only on CUDA (the run used CPU) | `train.py` GradScaler |
| Augmentations | **disabled** (`augmentation_enabled: false`); configured options: noise std 0.005, speed 0.9/1.0/1.1, time mask 10 frames, frequency mask 8 bins, gain ±6 dB | `training_config.json` |
| Hardware / duration | CPU fallback (the Kaggle P100 was incompatible), "a little under 8 hours" | `docs/VOICE_MODEL.md` |
| Test EER | **2.29 %** (accuracy@EER 97.71 %, AUC 0.9967, 281,625 pairs) | `evaluation/results/voice_metrics.csv` (verified consistent; §12) |
| EER during training | NOT AVAILABLE (validation used loss, not EER) | `models/voice/train.py` |

**Audio preprocessing** (`preprocessing/voice.py`, `VoiceConfig`)

| Parameter | Value |
|---|---|
| Resample | 16,000 Hz (`TARGET_SAMPLE_RATE`), mono |
| VAD | energy based, threshold ratio 0.02 (frame 400, hop 160, `_trim_silence`) |
| Loudness | RMS normalized to 0.1 (`_normalize_loudness`) |
| Segment | 4.0 s (`CLIP_SECONDS`); inference = deterministic centre crop, training = random crop |
| Features | log-mel, n_fft 400 (25 ms), hop 160 (10 ms), 80 mel bins (`_log_mel_filterbank`) |
| Capture (UI) | 4–5 s, auto-stop at 5 s; phrase "Security authentication for government access." (`VoiceCapture.tsx`) |

**Enrollment** (`ModalityService.enroll_confirmed`)
1. Two recordings are embedded.
2. The **exact** cosine between the two in-memory embeddings sets the quality band: EXCELLENT ≥ 0.85, GOOD ≥ 0.75, FAIR ≥ 0.60 (stored only if the user accepts it), POOR < 0.60 (rejected) (`backend/services/recording_quality.py`).
3. Templates are generated from recording 1 into the 4 sets.

**Authentication:** as for face, with `decide`: estimated Euclidean distance d̂ ≤ 0.75.

### 3.3 Fingerprint (optional modality)

| Item | Value | Source |
|---|---|---|
| Architecture | ResNet50 (ImageNet1K_V2) + projection head 2048→1024→BN→ReLU→Dropout(0.3)→512, L2 | `models/fingerprint/model.py` |
| Frozen layers | conv1, bn1, layer1 | `FingerprintConfig.frozen_backbone_layers` |
| Embedding | **512** | `FINGERPRINT_EMBEDDING_DIM` |
| Loss | ArcFace m 0.5, s 64.0, label smoothing 0.1 + hard-negative hinge (margin 0.3, weight 0.1, from epoch 10) | `FingerprintConfig` |
| Optimizer / schedule | AdamW lr 3e-4, wd 1e-4, betas (0.9, 0.999), grad clip 1.0; 3-epoch warmup + cosine to 1e-6; max 30 epochs | `FingerprintConfig`; `models/fingerprint/train.py` |
| Batching | P/K sampler: 16 identities × 4 samples | `FingerprintConfig`; `models/fingerprint/sampler.py` |
| Early stopping | patience 8 on validation **EER**, at least 10 epochs | `train.py` |
| Split | subject-disjoint 70/15/15, seed 42 (600 subjects → 90 test subjects) | `models/fingerprint/dataset.py::subject_disjoint_split` |
| Augmentation | Affine (rot ±12°, translate ±5 %, scale 0.9–1.1, p 0.8), brightness/contrast 0.15 (p 0.5), Gaussian noise (p 0.2), motion blur (p 0.15), elastic (p 0.2), grid distortion (p 0.2), random crop 200 (p 0.5), resize 224 | `dataset.py::build_augmentation_pipeline` |
| Preprocessing | grayscale → CLAHE → ridge normalization → Gaussian 3×3 → Gabor bank (8 orientations, 15×15, σ 4, λ 10, γ 0.5) → min-max → 224×224 → 3-channel → ImageNet normalization | `preprocessing/fingerprint.py::enhance` |
| Dataset | SOCOFing Real images, Kaggle `ruizgara/socofing` | `kernel-metadata.json` |
| Checkpoint | `models/fingerprint/saved/fingerprint_embedder.pt` | commit `3a0bfeb` |
| Test metrics (subject-level protocol, 900 images) | accuracy 69.23 %, EER 30.77 %, AUC 0.7644 | `fingerprint_metrics.csv` (reproduced; §12) |
| Epochs actually run / best epoch / hardware | NOT AVAILABLE | — |

**Enrollment / authentication:** a single image each (`ModalityService.enroll`); decision h ≥ 0.90 (fallback threshold).

**Documentation vs code.** `docs/PROJECT_REPORT.md` §7–8 describes an **older** fingerprint model: layer4-only fine-tuning, 12 epochs, EER 0.445 / AUC 0.579. The code and checkpoint were replaced in `fc9fd47` / `3a0bfeb`, so use the table above. The earlier protocol also treats *different fingers of the same subject* as genuine. The IEEE evaluation instead uses finger-level genuine pairs (a Real image vs SOCOFing's altered impressions): EER 4.74 % raw / 9.45 % protected.

---

## 4. Datasets

**Face**

| Field | Value |
|---|---|
| Dataset | LFW (funneled) via `sklearn.datasets.fetch_lfw_people` |
| Purpose | fine-tuning (identities with ≥ 20 images); held-out evaluation (identities with 2–19 images) |
| Identities / images | fine-tuning: 62 / 3,023 (`docs/PROJECT_REPORT.md` §8.2); evaluation: 1,618 / 6,141 (`evaluation/results/dataset_audit.csv`) |
| Train / val / test | 70/15/15 per identity (cell 8); exact counts printed at run time: NOT AVAILABLE |
| License | UMass, non-commercial research (`docs/DATASETS.md`) |
| Source | vis-www.cs.umass.edu/lfw (sklearn download) |
| Preprocessing | MTCNN 160×160 |

**Voice**

| Field | Value |
|---|---|
| Dataset | `gaurav41/voxceleb1-audio-wav-files-for-india-celebrity` (VoxCeleb1 Indian-celebrity subset) |
| Purpose | training + closed-set evaluation |
| Speakers / utterances | 24 / 4,857 |
| Train / val / test | 3,389 / 717 / 751 |
| License | DbCL-1.0 as declared by the mirror; not verified against VGG (`docs/DATASETS.md`) |
| Preprocessing | §3.2 |

**Fingerprint**

| Field | Value |
|---|---|
| Dataset | `ruizgara/socofing` (SOCOFing) |
| Purpose | training (Real); evaluation (Real + Altered-Easy/Medium/Hard) |
| Subjects / images | 600 / 6,000 Real (10 fingers each) + altered versions |
| Train / val / test subjects | 420 / 90 / 90 (subject-disjoint, seed 42; `round(0.7·600)` etc.) |
| License | non-commercial academic (Shehu et al., arXiv:1807.10609) |
| Preprocessing | §3.3 |

**Iris:** CASIA-Iris-Thousand mirror; no completed checkpoint, not part of the evaluated system (`docs/PROJECT_REPORT.md` §8.4).

**Demographics:** NOT AVAILABLE for every dataset.

---

## 5. Model training details (summary)

| | Face | Voice | Fingerprint |
|---|---|---|---|
| Backbone | InceptionResnetV1 (VGGFace2) | ECAPA-TDNN | ResNet50 (ImageNet) |
| Loss | ArcFace s 30, m 0.5 | ArcFace s 30, m 0.5 | ArcFace s 64, m 0.5, LS 0.1 + hard-negative |
| Optimizer | Adam 1e-4 | AdamW 1e-3, wd 1e-4 | AdamW 3e-4, wd 1e-4 |
| LR schedule | none | cosine (T = 30) | 3-epoch warmup + cosine → 1e-6 |
| Batch | 32 | 64 | 16 × 4 (P/K) |
| Epochs | 10 | 30 (patience 5) | ≤ 30 (patience 8, min 10) |
| Hardware | Kaggle; CPU fallback reported | CPU, ~8 h | NOT AVAILABLE |
| Duration | NOT AVAILABLE | ~8 h | NOT AVAILABLE |
| Best checkpoint metric | val_acc 0.913 (final epoch) | best validation loss (value NOT AVAILABLE); test EER 2.29 % | lowest validation EER (value NOT AVAILABLE); test EER 30.77 % (subject protocol) |

---

## 6. BioHash template protection

**Key derivation** (`template_protection/hkdf_keys.py::derive_key`):

$$\text{salt}=\mathrm{SHA256}(\text{app}\,\|\,\text{user}\,\|\,\text{modality}\,\|\,v),\qquad s_\ast=\mathrm{HKDF\text{-}SHA256}(\text{MASTER\_SECRET},\ \text{salt},\ \text{info}_\ast,\ 32\ \text{bytes})$$

The three `info` labels, `template_protection/projection`, `…/permutation` and `…/threshold`, give three independent seeds s_P, s_π, s_T. v is the `key_version`.

**Transform** (`template_protection/transform.py`, orchestrated by `biohash.py::generate_template`), for an embedding x ∈ ℝ^D and N = 256 output bits:
1. Normalize: x̂ = x / ‖x‖₂.
2. **Projection matrix:** a Gaussian G ∈ ℝ^{D×b} seeded from s_P, then QR decomposition G = QR, with W_block = Qᵀ (orthonormal rows; Haar-random). Blocks of b ≤ D rows are stacked until N rows exist. Face and fingerprint (D = 512) need 1 block; voice (D = 192) needs 2 blocks (192 + 64), which are not orthogonal to each other (`build_orthonormal_projection`).
3. **Projection:** p = W x̂ ∈ [−1, 1]^N.
4. **Quantization:** b_i = 1[p_i > t_i], with t_i ∼ 𝒩(0, (0.5·σ(p))²) seeded by s_T (`quantize`).
5. **Permutation:** c = π(b), with π a random permutation seeded by s_π (`apply_permutation`).
6. **Bit packing:** `utils.pack_bits` gives 32 bytes, stored in `protected_templates.protected_template`, with `output_bits = 256` and `template_version = 1` (`TEMPLATE_FORMAT_VERSION`).

**Comparison:** h(a, b) = (1/N) Σ 1[a_i = b_i]; Hamming distance H = N(1 − h). There is a constant-time exact-equality shortcut (`matcher.compare`).

**Why 256 bits.** It was raised from 128 after a separability experiment: 128 bits lost about 7 EER points against raw embeddings, and 256 recovered most of it (face AUC 0.861 → 0.909; voice 0.959 → 0.986). This experiment is documented in `docs/AUTHENTICATION_RELIABILITY_REPORT.md` "128 vs 256-bit" and `backend/config.py::template_bits`; its raw data is **not committed** (documented only). 256 ≤ 512 keeps the face projection fully orthonormal.

**Template pool and lifecycle** (`backend/database/crud.py`; `docs/MULTI_TEMPLATE_ARCHITECTURE.md`)
- Enrollment writes `template_pool_size = 4` sets (`backend/config.py`). Set 1 is **ACTIVE** and sets 2–4 are **STANDBY**, each under its own key_version.
- A set holds one template per enrolled modality. Enrolling a further modality adds it to every live set.
- `revoke_active_set_and_promote`: ACTIVE → **REVOKED** and the oldest STANDBY → ACTIVE, for all modalities in one transaction. With no STANDBY left, `TemplatePoolExhaustedError` (HTTP 409) is raised and re-enrollment is required.
- Authentication reads only the ACTIVE set and never mixes sets (`authentication.py`: a `SecurityValidationError` is raised on a set mismatch).
- Revoke / activate / generate require a fresh capture of every modality in the ACTIVE set (`backend/services/template_sets.py::authorize_with_active_set`; 403 otherwise).

---

## 7. Fusion strategy

**Per-modality decision and score conversion** (`backend/services/modality_metrics.py::decide`)

| Modality | Metric (estimate from h) | Match rule | Direction | Fusion score s | Fusion threshold t |
|---|---|---|---|---|---|
| Face | ĉ = h⁻¹(h) (calibrated cosine) | ĉ ≥ 0.80 | ↑ | s = ĉ | 0.80 |
| Voice | d̂ = √(2 − 2ĉ) (calibrated Euclidean) | d̂ ≤ 0.75 | ↓ | s = 1 − d̂²/2 (= ĉ) | 1 − 0.75²/2 = 0.71875 |
| Fingerprint | h (measured) | h ≥ 0.90 | ↑ | s = ĉ_fp | ĉ_fp(0.90) = 0.9384 |

Calibration: 1 − h(c) = (p₁θ + p₂θ² + p₃θ³)/π, with θ = arccos c; the inverse is tabulated and clamped to [−0.2, 1] (`template_protection/metric_estimation.py`). A raw distance is never averaged with a similarity.

**Weights** (`fusion/score_fusion.py`): w_m = 1 for each present modality, renormalized so that Σ w_m = 1.
- Fused score: S = Σ_{m∈M} w_m s_m.
- Fused threshold: T = (1/|M|) Σ t_m (`authentication.py`). For face + voice, T = 0.759375.

**Policies** (`fusion/policy.py`; default `ALL_REQUIRED`, `fusion/config.py`)
- **ALL_REQUIRED:** accept ⇔ ∀m ∈ M: match_m. S is reported but does not decide.
- **WEIGHTED:** accept ⇔ S ≥ T and ∀m: s_m ≥ 0.0 (`DEFAULT_WEIGHTED_FLOOR`, chance level on the cosine scale).
- **AT_LEAST_TWO:** valid only with exactly 3 modalities; accept ⇔ at least 2 match.

A submitted modality that is not enrolled returns ENROLLMENT_REQUIRED (HTTP 409); nothing is evaluated (`authentication.py`).

---

## 8. Thresholds

| Modality | Configured | Hamming equivalent | Max differing bits (of 256) | Origin | Source |
|---|---|---|---|---|---|
| Face | est. cosine ≥ 0.80 | h ≥ 0.8176 | 46 | **teacher-requested**; not optimized | `backend/config.py::face_cosine_threshold` |
| Voice | est. distance ≤ 0.75 | h ≥ 0.7823 | 55 | **teacher-requested**; not optimized | `backend/config.py::voice_euclidean_threshold` |
| Fingerprint | h ≥ 0.90 | h ≥ 0.90 | 25 | **inherited** default `match_threshold` | `backend/config.py`; `backend/threshold_loader.py` |
| Fusion (face + voice) | T = 0.759375 | — | — | **derived**; only decisive under WEIGHTED | `modality_metrics.py` |

The equivalents are **DERIVED** from the calibration curve (`decide(...).hamming_threshold`).

Measured consequences (**REAL DATA**, `evaluation/results/raw_vs_protected_metrics.csv`, `threshold_sweep_eer.csv`):
- **Face:** FAR 0.0025 %, FRR **81.5 %**. The EER point lies at ĉ ≈ 0.33.
- **Voice:** FAR 0.26 %, FRR 23.8 %. The EER point lies at d̂ ≈ 1.00.
- **Fingerprint:** FAR 16.2 %, FRR 5.8 %.

History: face and voice previously used 0.80 on raw Hamming similarity (operator-specified JSON files, removed in `873c13e`).

Provenance label: `Settings.threshold_source = TEACHER_REQUESTED_BASELINE` (CONFIGURATION). Sweep and operating points: face 0.60–0.90 and voice 0.50–1.00, step 0.01. Candidate points are selected on an identity-disjoint DEVELOPMENT split and evaluated once on TEST (`threshold_sensitivity.csv`, `threshold_operating_points.csv`; `evaluation/reports/THRESHOLD_ANALYSIS.md`). Example: face FAR ≤ 1 % selected on development gives FAR 0.83 % and FRR 22.9 % on test. **No threshold was changed.**

---

## 9. Database design

SQLAlchemy models in `backend/database/models.py`; the full column list is in `evaluation/results/db_schema.csv`.

| Table | Key fields | Notes |
|---|---|---|
| `users` | `id` (PK, internal ID `USER-xxxxxxxxxxxx` or legacy), `username` (**display name**, nullable), `created_at` | the name is a label only |
| `protected_templates` | `template_id` (PK), `user_id` (FK → users.id), `modality`, `application_id`, `template_version`, `key_version`, `output_bits`, **`protected_template` (BLOB, 32 B)**, `is_active`, `template_status`, `template_set_version`, `template_set_status`, `template_group_id`, `template_index`, activation / revocation timestamps and reason (21 columns) | one row per (set, modality) |
| `audit_logs` | `audit_id`, `timestamp`, `user_id`, `building_id`, `modality_list`, `similarity_scores`, `thresholds_used`, `fusion_score`, `fusion_policy`, `fusion_similarity`, `authenticated`, `authentication_state`, `latency_ms`, `template_versions`, `key_versions`, per-modality similarity columns, submitted / enrolled / authenticated modalities (24 columns) | stores scores, not biometric data |
| `enrollment_events` | `event_id`, `user_id`, `application_id`, `modality`, `outcome`, `created_at` | ENROLLED / ENROLLMENT_INCONSISTENT etc. |
| `master_secret_fingerprint` | `id`, `fingerprint` (hash of MASTER_SECRET), `created_at` | key-continuity check (`backend/key_continuity.py`) |

- **Stored:** 256-bit templates, key and set versions, display name, audit scores.
- **Never stored:** images, audio, embeddings, keys, MASTER_SECRET (`base_service.py` deletes embeddings after use; `docs/PRIVACY_AND_SECURITY.md`).
- **Audit scores:** rows before `873c13e` hold Hamming similarities; later rows hold fusion-scale scores (`AuthenticationResult.score`).
- **Measured storage** (**SOFTWARE TEST**, `evaluation/results/storage_analysis.csv`): 1,000 users with face + voice in 4 sets take 4.08 MB, about 3.96 KB per user. Of that, 256 B is template payload. An audit row is about 500 B.

---

## 10. Security design

**Threat model (implemented scope).** The design protects against disclosure of the template database: templates are keyed, revocable and not directly usable as biometric data. It does **not** address:
- presentation attacks (no liveness or PAD);
- an attacker who holds both the templates and MASTER_SECRET;
- a compromised capture client or server at the time of capture.

**Privacy guarantees (implemented):**
- No raw data or embeddings are persisted.
- Per-modality scores are returned only with `DEBUG_SCORES=true` (`backend/config.py::debug_scores`; `authentication.py::to_public_response`).
- The display name is returned only on ACCESS_GRANTED.
- CORS is restricted to configured origins (`backend/main.py::_resolve_cors_origins`).
- Uploads are validated for type and size (`Settings.allowed_*`, `max_upload_size_bytes`).

**Keys:**
- One root `MASTER_SECRET` (required; the server refuses to start without it).
- A per-(application_id, user_id, modality, key_version) key via HKDF-SHA256 (§6).
- **Application ID** separation: the same person in two applications gets unrelated keys.
- **Key versioning:** every template set uses a fresh key_version, and versions are never reused (`crud.next_key_version`).
- **MASTER_SECRET continuity:** a stored hash detects a changed secret at startup (`backend/key_continuity.py`, `scripts/rotate_master_secret.py`).

**Revocation flow.**
1. `POST /revoke-template` with fresh captures of every ACTIVE-set modality (`authorize_with_active_set`).
2. `revoke_active_set_and_promote`.
3. The next STANDBY set becomes ACTIVE.
4. With the pool exhausted: HTTP 409, then re-enroll.

Measured (**REAL DATA**, `revocation_summary.csv`):
- The revoked template against the new ACTIVE template gives h = 0.500 ± 0.030, accepted 0 % (face n = 150, voice n = 24).
- Genuine acceptance is unchanged after revocation.

**Cross-key unlinkability** (**REAL DATA**, `evaluation/results/unlinkability.csv`; Gomez-Barrero et al. 2018, histogram estimator, bin 4/256, with a permutation null floor):

| Modality | D_sys | Null-floor mean | Null-floor p95 |
|---|---|---|---|
| Face | 0.018 | 0.010 | 0.013 (marginally exceeded) |
| Voice | 0.020 | 0.026 | 0.040 |
| Fingerprint | 0.014 | 0.012 | 0.017 |

**Template statistics** (**REAL DATA**, `template_security.csv`; face shown):

| Statistic | Value |
|---|---|
| Fraction of ones | 0.5003 |
| Per-bit entropy | 0.9996 bits |
| Mean \|bit correlation\| | 0.0198 (independence expectation 0.0198) |
| Cross-user cross-key HD | 0.4998 |
| Daugman DoF | 253.5 (**DERIVED**) |
| Cross-key collisions at the decision threshold | 0 |

**User ID vs display name:** the internal ID is generated server-side (`crud.create_named_user`) and is the only identifier used for keys and storage. The name is validated separately (`backend/display_names.py`), duplicates are allowed, and legacy users are shown as "User <short id>".

---

## 11. Implementation details

| Layer | Technology (versions from `requirements.txt` / `frontend/package.json`) |
|---|---|
| API framework | FastAPI ≥ 0.110, Uvicorn ≥ 0.27, python-multipart |
| Backend config | pydantic 2 / pydantic-settings (`backend/config.py`) |
| Database | SQLite via SQLAlchemy 2 (Postgres driver present; `database_url` configurable) |
| ML | PyTorch ≥ 2.2, torchvision, torchaudio, facenet-pytorch ≥ 2.5.3, SpeechBrain ≥ 1.0, OpenCV, scikit-learn, albumentations |
| Crypto | `cryptography` ≥ 42 (HKDF-SHA256), `hashlib` SHA-256 |
| Frontend | React 19, TypeScript 6, Vite 8, Tailwind CSS 4, react-router 7, motion |
| Deployment | backend on Render (`backend/render.yaml`, free plan, persistent SQLite disk `DATABASE_PATH`); frontend on Vercel (`frontend/vercel.json`) |
| Model storage | Git LFS (`.gitattributes`: *.pt, *.h5) |

---

## 12. Experimental evidence inventory

| Artefact | Content | Label | Source |
|---|---|---|---|
| `evaluation/results/face_metrics.csv` | acc 0.990, EER 0.010, AUC 0.999 (training-identity test) | REAL DATA (NOT VERIFIABLE) | face notebook |
| `voice_metrics.csv`, `voice_roc.csv`, `voice_confusion_matrix.csv` | EER 2.29 %, AUC 0.9967, 13,101 genuine / 268,524 impostor pairs | REAL DATA (all values verified; EER threshold drifted, §14) | voice notebook |
| `fingerprint_*.csv` (4 files) | EER 30.77 %, AUC 0.7644, subject protocol | REAL DATA (verified) | fingerprint notebook |
| `biohash_metric_calibration.json` | 3 × 19,600 synthetic pairs; fit R² ≥ 0.99986 | SYNTHETIC | `scripts/calibrate_biohash_metric_mapping.py` |
| `calibration_real_validation.csv`, `calibration_real_binned.csv` | real-pair estimate error; per-bin SD matches synthetic (e.g. face [0.8, 0.9): 0.046 vs 0.043) | REAL DATA | `scripts/validate_calibration_real.py` |
| `raw_vs_protected_metrics.csv` | EER, AUC, TAR@FAR, operating points, bootstrap CIs | REAL DATA | `evaluation/ieee/experiments.py` Part 4 |
| `threshold_sweep*.csv` | FAR/FRR/TAR/accuracy/F1 vs threshold | REAL DATA | Part 6 |
| `template_set_*.csv` (4) | diversity / revocation / promotion / exhaustion | SYNTHETIC (re-run matches exactly) | `evaluation/template_set_experiments.py` |
| `revocation_*.csv` | real-embedding lifecycle through the shipped code | REAL DATA | Parts 7/9 |
| `unlinkability.csv` | D_sys + null floor | REAL DATA | Part 8 |
| `fusion_policy_metrics.csv`, `fusion_score_level_eer.csv` | 11 policies, chimeric users (20 pairings × 24) | REAL DATA (chimeric) | Part 14 |
| `template_security.csv` | bit balance, entropy, correlation, DoF, collisions | REAL DATA / DERIVED | Part 16 |
| `robustness.csv` | 28 degradation conditions | REAL DATA (simulated degradations) / FUTURE (pose, microphone) | `evaluation/ieee/robustness.py` |
| `latency_benchmark.csv` | per-stage warm (100) and cold (10) runs; API face+voice median 757 ms | REAL DATA (one CPU laptop) | `scripts/benchmark_latency.py` |
| `memory_benchmark.csv` | peak 692 MB for a face+voice request | REAL DATA | `scripts/benchmark_memory.py` |
| `storage_analysis.csv` | ~3.96 KB per user | SOFTWARE TEST / DERIVED | `evaluation/ieee/storage.py` |
| `software_validation.csv` | 636/636 backend tests (2026-09-25); tsc 0 errors; lint 0 errors / 25 warnings; build OK | SOFTWARE TEST | `evaluation/ieee/audit.py software` |
| `metric_verification.csv` | 70 match, 1 discrepancy, 3 not verifiable | DERIVED | `evaluation/ieee/verification.py` |
| `pad_results.csv` | IAPMR NOT AVAILABLE | FUTURE | `evaluation/ieee/pad_evaluation.py` |
| `configuration.csv`, `api_endpoints.csv`, `db_schema.csv`, `dataset_audit.csv` | extracted facts | CONFIGURATION / REAL DATA | `evaluation/ieee/audit.py` |
| Real-user FAR/FRR | — | FUTURE (harness `evaluation/real_user_evaluation.py`) | — |

Local-only (gitignored `biometric.db`): 54 unlabelled real attempts from 7 internal IDs. **Not usable for error rates** (no ground truth).

---

## 13. Figures available

All in `evaluation/figures/` at 300 dpi, matplotlib only. Labels and captions are in `evaluation/figures/figure_index.csv`.

| Figures | Content | Data / script |
|---|---|---|
| fig01–fig09 | architecture, enrollment, authentication, face / voice / fingerprint pipelines, BioHash, template lifecycle, fusion | `evaluation/ieee/figures.py` (from code) |
| fig10, fig11 | calibration curve, residuals and uncertainty | `biohash_metric_calibration.json` |
| fig21–fig23 | real-pair calibration scatter, error vs cosine, residual histogram | `calibration_real_*.csv` |
| fig12, fig13, fig24, fig14 | ROC raw vs protected per modality; all modalities | `raw_vs_protected_metrics.csv` (Part 4) |
| fig15 | DET curves | Part 4 |
| fig16, fig25 | FAR/FRR vs threshold; accuracy / F1 vs threshold | `threshold_sweep.csv` |
| fig18 | revocation histogram | `revocation_per_attempt.csv` |
| fig19 | unlinkability histograms with D↔ | `unlinkability.csv` |
| fig26 | fusion ROC | Part 14 |
| fig27 | robustness FRR / FAR | `robustness.csv` |
| fig17, fig28 | latency per stage; memory | benchmark CSVs |
| fig20 | storage comparison | `storage_analysis.csv` |
| Confusion matrices | voice and fingerprint (committed CSVs); per-policy TP/FN/FP/TN | `*_confusion_matrix.csv`, `fusion_policy_metrics.csv` (no PNG yet) |
| Score distributions (deployed system, real users) | — | NOT AVAILABLE |

---

## 14. Limitations (from code and results only)

1. **No liveness / presentation-attack detection** anywhere in the pipeline. IAPMR is unmeasured, and APCER/BPCER are not applicable (`evaluation/results/pad_results.csv`).
2. **No labelled real-user study of the deployed system,** so real-user FAR/FRR are unknown. The harness exists (`evaluation/real_user_evaluation.py`).
3. **Teacher thresholds produce a very high face FRR** (81.5 %; ALL_REQUIRED face + voice 84.9 %) on the evaluation data (§8).
4. **Voice is closed-set** (24 speakers, shared across splits). The speaker count is known (24), but only the Indian-celebrity subset is represented.
5. **Face alignment:** the deployed face path uses the bounding box only. Landmark alignment is implemented and, with a retrained checkpoint, lowers the protected EER from 8.67 % to 7.18 % (paired Δ −1.49 pp, CI [−1.90, −0.69]). It is not deployed, because switching needs the new checkpoint and re-enrollment. With the **old** checkpoint it is worse (+1.46 pp). See `FACE_ALIGNMENT_DECISION.md`.
6. **Fingerprint model is weak** under the subject protocol (EER 30.8 %). Its embeddings are compressed into cosine 0.8–1.0, which gives FAR 16 % at h ≥ 0.90. Fingerprint genuine probes are synthetic alterations of one impression.
7. **Calibration mismatch (RESOLVED 2026-09-25):** the fingerprint curve had been fitted at D = 256. It is now refitted at D = 512. Face and voice curves are bit-identical, and the fingerprint curve moved by at most 0.00056 in Hamming. Fusion results were regenerated and are identical (`docs/FINGERPRINT_DOCUMENTATION_AUDIT.md`).
8. **Estimates are not exact metrics.** The per-attempt SD is about 0.05 in cosine near the thresholds, rising to about 0.1 for impostor-level pairs.
9. **Chimeric fusion evaluation:** it assumes modality independence and has only 24 virtual users per pairing.
10. **No demographic evaluation;** no robustness data for head pose or a physical microphone change.
11. **No throughput / concurrency benchmark;** latency was measured on a single CPU laptop. Multi-modality requests release and reload models by design (for a 512 MB host), which adds about 470 ms (`authentication.py::release_modality_cache`).
12. **No formal cryptographic or attack evaluation** (inversion, hill-climbing, stolen-key); BioHash security depends on MASTER_SECRET secrecy.
13. **Face test metrics in `face_metrics.csv`** come from identities used in fine-tuning and cannot be verified.
14. **Voice EER threshold drift:** the committed evaluation has an EER threshold of 0.322, but re-extraction gives 0.437. EER and AUC reproduce, which points to a later preprocessing change (`c7d1c7e`).
15. **Training records are incomplete:** the notebooks contain no saved outputs; fingerprint epochs and hardware are NOT AVAILABLE.
16. **The iris modality is not implemented end-to-end** (no checkpoint).

---

## 15. Viva notes

**Why BioHash?**
- It is a well-studied cancelable transform: a keyed random projection followed by binarization.
- It gives fixed-length binary templates that compare cheaply (Hamming comparison takes 0.01 ms, measured).
- It is revocable by changing the key.
- Security rests on key secrecy, not on a hardness proof (`template_protection/biohash.py` docstring).

**Why HKDF?**
- It derives independent, reproducible per-(application, user, modality, version) seeds from one secret.
- There is no per-user key storage, and revocation just increments the version.
- Separate `info` labels keep the projection, threshold and permutation seeds independent (`hkdf_keys.py`).

**Why Hamming?**
- It is the natural metric for binary templates, and the only comparison possible, because embeddings are never stored.
- For a random-hyperplane-style projection, the expected fraction of differing bits grows monotonically with the angle between the embeddings. This is the basis of the calibration.

**Why cosine for face?**
- The face model is trained with an angular-margin loss (ArcFace) on L2-normalized embeddings, so angle is the native similarity.
- It is the teacher-requested metric.
- At runtime it is a **calibrated estimate** from h, validated on real LFW pairs.

**Why Euclidean for voice?**
- It was requested as a distinct metric.
- For unit vectors, d = √(2 − 2 cos θ), so it carries the same angular information with the opposite direction (lower is better).
- It is converted back to s = 1 − d²/2 for fusion, so a distance is never averaged with a similarity.

**Why 256 bits?**
- The 128-bit templates measurably lost separability, and 256 recovered most of it (documented experiment).
- 256 ≤ 512 keeps the face projection orthonormal.
- Measured cost of protection at 256 bits: face +1.4 EER points, voice +0.1.

**Why ALL_REQUIRED?**
- It fixes a found flaw: a weighted average let a strong modality hide a failed one (`fusion/config.py` docstring).
- Each presented factor must pass on its own, which is AND semantics.
- The cost is a higher FRR (84.9 % with the teacher thresholds), whereas score-level fusion reached EER 1.67 %.

**Why template revocation / pools?**
- A compromised credential must be replaceable without re-capture.
- The 4 pre-generated keyed sets allow instant promotion.
- Whole sets move together, so modalities never mix keys.
- Measured: a revoked template does not match the new one (h ≈ 0.50, 0 % accepted).

**Why estimates rather than exact metrics?**
- Storing embeddings to compute exact metrics would defeat cancelability.
- The Hamming-to-cosine curve depends only on the transform (rotation invariance), so it can be calibrated offline.
- On real embeddings, the error SD matches the synthetic prediction.

**Why a display name separate from the ID?** The ID drives keys and storage and must be unique and stable. Names can collide, and they are shown only after successful authentication.

**What does this system not protect against?** Presentation attacks, leakage of MASTER_SECRET together with the templates, and a compromised capture device (§10).

---

## 16. Status of the 2026-09-25 changes

| Item | IMPLEMENTED | EVALUATED | Evidence / notes |
|---|---|---|---|
| Face 5-landmark similarity alignment (`FacePreprocessorAligned`, FACE_ALIGNED mode) | yes | yes (REAL DATA, held-out LFW) | `face_alignment_*.csv`; default remains FACE_BASELINE |
| FACE_BASELINE preserved (`FacePreprocessorBaseline`) | yes | yes (bit-identical output, and system A reproduces the deployed FRR 0.814946 exactly) | `tests/test_face_alignment.py` |
| LANDMARK_FAILURE / ALIGNMENT_FAILED statuses | yes | SOFTWARE TEST | `tests/test_face_alignment.py`, `tests/test_final_system_workflow.py` |
| Controlled alignment training pair (`face_bbox_fullframe_v2`, `face_aligned_v2`) | yes | COMPLETED | `training/face/ALIGNMENT_RUNS.md` |
| Aligned checkpoint deployed | no | — | NOT_AVAILABLE in production: needs installation plus re-enrollment |
| Fingerprint calibration at D = 512 | yes | SYNTHETIC CALIBRATION (R² 0.99993) | `biohash_metric_calibration.json` |
| Threshold sweep + development/test operating points | yes | yes (REAL DATA) | `threshold_sensitivity.csv`, `threshold_operating_points.csv` |
| THRESHOLD_SOURCE label | yes | CONFIGURATION | `backend/config.py` |
| Raw vs protected on identical pairs | yes | yes | `protected_vs_raw.csv` |
| Real-user consent / withdrawal / protocol check | yes | SOFTWARE TEST | `evaluation/REAL_USER_PROTOCOL.md` |
| Real-user results | harness only | **NOT_AVAILABLE** | no participants were recruited |
| Voice training reproduction | started | **NOT_AVAILABLE** (interrupted after epoch 8) | `training/voice/runs/` |
| Protected-template ROC figure (fig 3) | yes | yes (chimeric users) | `evaluation/scripts/fig03_roc_protected_templates.py` |

