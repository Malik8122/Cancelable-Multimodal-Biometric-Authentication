# Cancelable Multimodal Biometric Authentication for Critical Infrastructure
## Comprehensive Technical Project Documentation

> **Modality change (2026-09-27).** Iris and fingerprint have been removed from the system: the implemented
> modalities are **face and voice**, and the `AT_LEAST_TWO` fusion policy (which needed three modalities) is gone.
> This document is a dated record. Its iris / fingerprint material (models, datasets, results, thresholds,
> three-modality fusion) is kept for history only and no longer describes the code. Current state: `README.md`,
> `docs/ROADMAP.md`; current numbers: `evaluation/PAPER_RESULTS_INDEX.md`.

> **Update - superseded parts.** This document is a snapshot of commit `5c67be1`. Since then, face and voice are decided in their own metrics: face on a calibrated **estimate** of cosine similarity (>= 0.80), voice on a calibrated **estimate** of Euclidean distance (<= 0.75, lower is better), both derived from the unchanged 256-bit BioHash Hamming comparison. `evaluation/results/face_threshold.json` and `voice_threshold.json` (0.80 on Hamming similarity), described below, were removed, and registration now asks for a display name. See [BIOMETRIC_METRICS.md](BIOMETRIC_METRICS.md) for the current behaviour; everything else here is unchanged.

**Document status:** describes the project **as implemented in the repository at commit `5c67be1`** ("Finalize biometric enrollment and authentication UX"), branch `main`. This is documentation of the actual system, not the original proposal. Every claim below is either a direct citation from source code/tests/docs (marked with file paths), a measured result from a committed artifact, or explicitly labeled as an inference, limitation, or future-work item. Where something could not be verified in the repository, this document says so explicitly rather than guessing.

**Labels used throughout:**
- **FACT** — directly verifiable from source code, config, or committed data.
- **MEASURED RESULT** — a number produced by an actual run and committed to the repo (a CSV/JSON in `evaluation/results/`, or a documented real observation).
- **INFERENCE** — reasonably concluded from code behavior but not stated explicitly anywhere.
- **CURRENT LIMITATION** — something the system does not do today.
- **FUTURE WORK / PLANNED** — discussed in project docs as a next step, not implemented.

---

## Table of Contents

1. [Project Overview](#part-1--project-overview)
2. [Complete System Architecture](#part-2--complete-system-architecture)
3. [Datasets](#part-3--datasets)
4. [Model Training](#part-4--model-training)
5. [Face Pipeline](#part-5--face-pipeline)
6. [Face Quality Gating](#part-6--face-quality-gating)
7. [Fingerprint Pipeline](#part-7--fingerprint-pipeline)
8. [Voice Pipeline](#part-8--voice-pipeline)
9. [Template Protection](#part-9--template-protection)
10. [MASTER_SECRET Key Continuity](#part-10--master_secret-key-continuity)
11. [Enrollment](#part-11--enrollment)
12. [Authentication](#part-12--authentication)
13. [Fusion](#part-13--fusion)
14. [Fusion Diagnostics](#part-14--fusion-diagnostics)
15. [Database](#part-15--database)
16. [Security and Privacy](#part-16--security-and-privacy)
17. [Debugging / Major Problems Found](#part-17--debugging--major-problems-found)
18. [Actual Experimental Results](#part-18--actual-experimental-results)
19. [Testing](#part-19--testing)
20. [Current Frontend User Flow](#part-20--current-frontend-user-flow)
21. [Repository Structure](#part-21--repository-structure)
22. [How to Run on a New Laptop](#part-22--how-to-run-on-a-new-laptop)
23. [What Has Not Been Implemented](#part-23--what-has-not-been-implemented)
24. [Research/Evaluation Readiness](#part-24--researchevaluation-readiness)
25. [Current Status](#part-25--current-status)

---

## PART 1 — Project Overview

### What the project is

A capstone system implementing **multimodal biometric authentication** (face, fingerprint, voice) combined with **cancelable, revocable biometric template protection**, exposed through a FastAPI backend and a React/Vite frontend. FACT, per `README.md:1-20` and the actual implemented code.

### Problem statement (as stated in the project's own docs)

Traditional biometric systems store templates derived directly from a raw biometric characteristic. Unlike a password, a face/fingerprint/voice cannot be changed after a breach, so a compromised biometric database is a **permanent** exposure, not a recoverable one. `README.md:24-30`.

### Motivation / proposed solution

Combine multimodal recognition (reliability from more than one factor) **with** a cancelable/revocable template-protection layer, so the system never stores a raw image or a raw, unprotected embedding — only a protected, one-way-transformed template is persisted and compared. `README.md:32-44`.

```text
Biometric input -> feature extraction -> embedding -> cancelable
transformation -> protected template -> authentication
```

### Target use case

A facility/checkpoint-style access-control demo ("Critical Infrastructure" framing) — the frontend presents a "campus" of buildings, each with its own authentication checkpoint. `frontend/src/pages/LandingPage.tsx`, `frontend/src/pages/CheckpointPage.tsx`. The building is **context only**: it carries no biometric policy of its own — which factors to present is the user's choice. `backend/api/authenticate.py` docstring, `backend/services/authentication.py:1-20`.

### The modalities actually implemented and offered

| Modality | Offered in the deployed web app? | Evidence |
|---|---|---|
| Face | Yes | `frontend/src/config/buildings.ts` `FACTORS`, `backend/services/face_service.py` |
| Fingerprint | Yes | Same |
| Voice | Yes | Same |
| Iris | **No** — implemented as scaffolding only, never trained, not exposed in the UI | `docs/PROJECT_FUNDAMENTALS.md:152-156`: "Iris is not offered in the web app"; `models/iris/saved/` contains only `.gitkeep`, no checkpoint (verified by directory listing) |

**FACT:** the frontend's own selectable-factor list (`frontend/src/config/buildings.ts`) offers exactly face, fingerprint, and voice. Iris exists in the backend/model code (`models/iris/`, `preprocessing/iris.py`, `backend/services/iris_service.py`) and is reachable via the raw API, but is not surfaced anywhere in the frontend UI and has no trained checkpoint, so it runs in a non-biometric mock-mode fallback if ever invoked.

### Multimodal fusion — authentication objective

The system fuses whichever modalities the user chooses to present in one session into a single access decision (`ACCESS_GRANTED` / `ACCESS_DENIED`), never per-modality decisions leaking to the client in production. `backend/services/authentication.py:1-20`, `fusion/`.

### What this project explicitly does NOT claim or implement

- **No deepfake detection.** Not implemented. `docs/PRIVACY_AND_SECURITY.md:59-65` states plainly: "Nothing in this API cryptographically binds an `/authenticate` request to a live capture (e.g. no liveness detection, no challenge-response) — a captured *image* could be replayed... Documented here rather than silently out of scope." **FUTURE WORK / PLANNED** (on hold, per explicit project instruction during this development phase).
- **No liveness / presentation-attack detection.** Same citation. **FUTURE WORK / PLANNED.**
- **No claim of cryptographic one-wayness, regulatory compliance, or "100% security."** `docs/PRIVACY_AND_SECURITY.md:33-43`, `docs/TEMPLATE_PROTECTION.md:183-214` state this explicitly and repeatedly.

This document does not describe deepfake/liveness detection anywhere below as implemented, because it is not.

---

## PART 2 — Complete System Architecture

### End-to-end flow (as implemented)

```text
 FRONTEND (React + Vite, frontend/src/)
     |  canvas.toBlob() capture (face/fingerprint image, or WAV recording)
     v
 HTTP API call (frontend/src/api/client.ts)
     |  multipart/form-data: user_id, application_id, modality-specific fields
     v
 BACKEND (FastAPI, backend/)
     |  backend/api/{enroll,authenticate,verify,fusion}.py
     v
 decode_biometric_sample()  (backend/utils.py)
     |  BGR->RGB via cv2 for images; WAV decode for voice
     v
 PREPROCESSING  (preprocessing/{face,fingerprint,voice,iris}.py)
     |  modality-specific: detect/align (face), CLAHE+Gabor (fingerprint),
     |  resample+VAD+mel (voice), Hough+Daugman (iris)
     v
 FEATURE EXTRACTION / EMBEDDING  (models/{face,fingerprint,voice,iris}/inference.py)
     |  BaseEmbedder.extract_embedding() -> fixed-length, L2-normalized vector
     v
 [ENROLLMENT ONLY] face: centroid over 5 accepted captures (embeddings/centroid.py)
     v
 TEMPLATE PROTECTION  (template_protection/)
     |  HKDF-SHA256 key derivation -> orthonormal projection -> keyed
     |  quantization -> keyed permutation  =>  protected template (bits)
     v
 MATCHING  (template_protection/matcher.py)
     |  Hamming similarity between candidate and stored protected template
     v
 FUSION  (fusion/)
     |  weighted average of submitted modalities' scores + policy (ALL_REQUIRED)
     v
 DECISION  (backend/services/authentication.py)
     |  ACCESS_GRANTED / ACCESS_DENIED / ENROLLMENT_REQUIRED
     v
 FRONTEND RESULT PAGE  (frontend/src/pages/ResultPage.tsx)
     |  DecisionSummary + (DEBUG_SCORES only) FusionDiagnosticsPanel
```

### Modality-specific architectures (traced from actual code, not assumed identical)

#### FACE

```text
Camera (getUserMedia)
  -> canvas.drawImage() + canvas.toBlob('image/png')      [frontend/src/components/capture/{GuidedFaceCapture,FaceCapture}.tsx]
  -> POST multipart/form-data
  -> backend/utils.py::decode_image()  (cv2.imdecode BGR -> cv2.cvtColor RGB)
  -> preprocessing/face.py::FacePreprocessor.detect_and_align()
       MTCNN.detect(landmarks=True) -> box, probability, 5 landmarks
       extract_face(box) + fixed_image_standardization()  -> 160x160x3 aligned crop
       [ENROLLMENT ONLY] quality gates applied to box/landmarks/sharpness (Part 6)
  -> models/face/inference.py::FaceEmbedder  (InceptionResnetV1, 512-d)
  -> BaseEmbedder._l2_normalize()
  -> [ENROLLMENT ONLY] embeddings/centroid.py::centroid_embedding() over up to 5 accepted captures
  -> template_protection/biohash.py::generate_template()  -> protected template
  -> [AUTHENTICATION] template_protection/matcher.py::compare(metric="hamming")
```

#### FINGERPRINT

```text
Camera/scan image capture (single image, no guided multi-capture)
  -> POST multipart/form-data
  -> backend/utils.py::decode_image()
  -> preprocessing/fingerprint.py::FingerprintPreprocessor.preprocess()
       grayscale -> CLAHE(clipLimit=3.0, tileGridSize=(8,8)) -> ridge normalize
       (target_mean=100.0, target_var=100.0) -> Gaussian blur (3x3)
       -> 8-orientation Gabor filter bank, max response -> min-max normalize
       -> resize 224x224 -> replicate to 3-channel -> ImageNet mean/std normalize
  -> models/fingerprint/inference.py::FingerprintEmbedder  (ResNet50 + projection head, 512-d)
  -> BaseEmbedder._l2_normalize()
  -> template_protection/biohash.py::generate_template()  -> protected template
  -> [AUTHENTICATION] template_protection/matcher.py::compare(metric="hamming")
```
**No quality gating exists for fingerprint** (no `check_capture`/multi-capture logic in `FingerprintPipeline` — it only inherits the base `preprocess -> embed` path). FACT, per the fingerprint-pipeline research agent's direct read of `embeddings/pipelines.py`.

#### VOICE

```text
Microphone recording (WAV)                                [frontend/src/hooks/useWavRecorder.ts]
  -> POST multipart/form-data (one recording for auth; TWO recordings for enrollment)
  -> backend/utils.py::decode_audio()
  -> preprocessing/voice.py::VoicePreprocessor.preprocess()
       load WAV (int PCM -> float32 [-1,1]) -> mono -> resample to 16 kHz
       -> VAD/silence trim (energy-based, per-sample mask) -> loudness normalize
       -> fixed-length segment (4.0 s, center-crop or pad) -> 80-bin log-mel
       filterbank (n_fft=400, hop=160) -> per-bin mean normalization
  -> models/voice/inference.py::VoiceEmbedder  (ECAPA-TDNN, 192-d)
  -> BaseEmbedder._l2_normalize()
  -> [ENROLLMENT ONLY] two-recording consistency check (cosine similarity bands,
       backend/services/recording_quality.py) BEFORE any template is generated
  -> template_protection/biohash.py::generate_template()  -> protected template
  -> [AUTHENTICATION] template_protection/matcher.py::compare(metric="hamming")
```

The three modalities are **not** the same pipeline: face has guided multi-capture + geometric/quality gating; fingerprint is single-capture with zero enrollment-time gating; voice is single-capture for authentication but two-recording consistency-checked for enrollment. FACT.

---

## PART 3 — Datasets

**Source verified:** `README.md:296-306`, `docs/DATASETS.md` (full file), `docs/VOICE_MODEL.md`, `docs/PROJECT_FUNDAMENTALS.md`, and directory listings of `models/*/saved/` and `evaluation/results/`.

### 3.1 Face — LFW (Labeled Faces in the Wild)

| Field | Value |
|---|---|
| Source | `http://vis-www.cs.umass.edu/lfw/`, auto-downloaded via `sklearn.datasets.fetch_lfw_people` |
| Access | No login required |
| Subjects used | 62 identities (`min_faces_per_person=20` filter) — **MEASURED RESULT**, `docs/PROJECT_FUNDAMENTALS.md:96-102` |
| Samples used | 3,023 images |
| License | Non-commercial research use (University of Massachusetts); not a commercial license, per `docs/DATASETS.md:13-17` |
| Split | Per-identity (train/val/test); exact split code not independently re-verified for face — see §4.1 note |
| Purpose | Training dataset (fine-tuning) AND evaluation dataset — **the same 62 identities are used for both**, i.e. this is a closed-set result, not open-set generalization. Explicitly caveated in `docs/PROJECT_FUNDAMENTALS.md:103-105`. |
| Retention | Never committed to the repo; exists only during the Colab/Kaggle runtime |
| Preprocessing | MTCNN detect+align before training (same preprocessing module used at inference) |
| Augmentation | Not verified in the current repository for face specifically |

### 3.2 Iris — CASIA-Iris-Thousand

| Field | Value |
|---|---|
| Source | `biometrics.idealtest.org` (official, license agreement required); unofficial Kaggle/HuggingFace mirrors exist but "redistribution rights... are not verified by this project" (`docs/DATASETS.md:26-30`) |
| Access | Licensed; the notebook does not auto-download and requires the user to set `DATASET_ROOT` themselves |
| Subjects/samples used | **Not applicable — no checkpoint was ever trained.** `models/iris/saved/` contains only `.gitkeep`; no `.pt`/`.h5` file exists (verified by direct directory listing). |
| Purpose | **Planned as a training dataset; training was never completed to a usable checkpoint.** `docs/PROJECT_FUNDAMENTALS.md:152-156`: "One run trained fully but a clean evaluated checkpoint was blocked by Kaggle's free GPU quota." |
| Current status | Iris runs in `BaseEmbedder`'s deterministic mock-mode fallback whenever invoked — not biometrically meaningful, and not exposed in the frontend UI. |

### 3.3 Fingerprint — SOCOFing (Sokoto Coventry Fingerprint Dataset)

| Field | Value |
|---|---|
| Source | `https://www.kaggle.com/datasets/ruizgara/socofing`, downloaded via `kagglehub` |
| Access | Free Kaggle account required |
| Subjects | 600 African subjects — **MEASURED RESULT**, `docs/DATASETS.md:46-48` |
| Samples used | 6,000 fingerprint images (the "Real" folder only; synthetically altered variants — obliteration, central rotation, z-cut — exist in the dataset but were **not used**) |
| License | Free for non-commercial academic research (Shehu, Ruiz-Garcia et al., 2018, arXiv:1807.10609) |
| Split | **Subject-disjoint** 70/15/15, seed 42 — implemented in code: `models/fingerprint/dataset.py:67-94::subject_disjoint_split()`, config values in `models/fingerprint/config.py:70-71` and the committed `fingerprint_config.json` |
| Purpose | Training AND evaluation dataset — held-out test set is subject-disjoint from train (900 test samples of the 6,000 total) |
| Preprocessing | CLAHE + ridge normalization + Gabor filtering (see Part 7) |
| Augmentation | Not verified in the current repository (no explicit augmentation config found for fingerprint, unlike voice which has one) |

### 3.4 Voice — VoxCeleb1 (Indian-celebrity subset)

| Field | Value |
|---|---|
| Source | Kaggle dataset `gaurav41/voxceleb1-audio-wav-files-for-india-celebrity` |
| Access | Free Kaggle account required; attached natively to the Kaggle kernel |
| Explicitly NOT | The full ~1,251-speaker VoxCeleb1 corpus — this is a deliberate subset, "impractical to download unattended inside a single Kaggle kernel session" (`docs/VOICE_MODEL.md:72-78`) |
| Speakers used | 24 — **MEASURED RESULT**, `docs/VOICE_MODEL.md:84-89` |
| Utterances used | 4,857 total; split 3,389 train / 717 val / 751 test (70/15/15 per speaker) |
| License | DbCL-1.0 (Database Contents License); redistribution grant from VoxCeleb1's original maintainers (Oxford VGG) "not independently verified" per `docs/DATASETS.md:67-72` |
| Split | Per-identity 70/15/15 — implemented in code: `models/voice/dataset.py:39-49::_split_indices()`, config in `models/voice/config.py:54`, committed values in `training_config.json` |
| Purpose | Training AND evaluation dataset (751-utterance held-out test set) |
| Preprocessing | Resample, VAD, mel-spectrogram (see Part 8) |
| Augmentation | A configurable augmentation pipeline exists (`VoiceConfig.augmentation_*` — Gaussian noise, speed perturbation, time/frequency masking, random gain) but is **disabled by default** (`augmentation_enabled=False`) and confirmed disabled in the actual committed run (`training_config.json`: `"augmentation_enabled": false`). |

### 3.5 Summary table

| Modality | Dataset | Training dataset? | Evaluation dataset? | Checkpoint exists? |
|---|---|---|---|---|
| Face | LFW (62 identities, 3,023 images) | Yes | Yes (closed-set, same identities) | Yes |
| Iris | CASIA-Iris-Thousand | Planned, not completed | No | **No** |
| Fingerprint | SOCOFing (600 subjects, 6,000 images) | Yes | Yes (subject-disjoint) | Yes |
| Voice | VoxCeleb1 subset (24 speakers, 4,857 utterances) | Yes | Yes (per-speaker split, same speakers as train) | Yes |

**No raw image or audio file from any of these datasets is committed to the repository** — `.gitignore` blocks image file types and datasets/ directories outright as a second line of defense beyond the notebooks never writing raw data to disk. FACT, `docs/DATASETS.md`, `.gitignore`.

---

## PART 4 — Model Training

### 4.1 Summary table

| Modality | Model | Pretrained? | Trained by project? | Fine-tuned? | Output dim | Current use |
|---|---|---|---|---|---|---|
| Face | InceptionResnetV1 (`facenet-pytorch`) | Yes, VGGFace2 | Yes (fine-tuning) | Yes — `block8`, `last_linear`, `last_bn` unfrozen; rest frozen | 512 | Real checkpoint loaded, active in production |
| Iris | ResNet18 + projection head | Yes, ImageNet | Attempted; no usable checkpoint produced | N/A — never completed | 256 | **Mock mode only; not offered in the UI** |
| Fingerprint | ResNet50 + projection head ("DeepPrint substitute") | Yes, ImageNet (`IMAGENET1K_V2`) | Yes (fine-tuning) | Yes — ArcFace + label smoothing + hard-negative mining from epoch 10 | 512 | Real checkpoint loaded, active in production |
| Voice | ECAPA-TDNN (SpeechBrain's lower-level module, not the pretrained `EncoderClassifier` pipeline) | **No** — architecture used without SpeechBrain's pretrained speaker-verification weights | Yes (trained via this repo's own pipeline, effectively from architecture-only initialization) | Yes | 192 | Real checkpoint loaded, active in production |

**Important clarification (face vs. voice, easy to conflate):** Face uses a genuinely pretrained checkpoint (VGGFace2) as its fine-tuning starting point. Voice uses the ECAPA-TDNN **architecture** from SpeechBrain but does **not** load SpeechBrain's pretrained `spkrec-ecapa-voxceleb` weights — `docs/VOICE_MODEL.md:17-25` explicitly documents using the lower-level `speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN` module (which takes precomputed features, has no pretrained speaker-verification weights attached) rather than the full pretrained pipeline. This is an **INFERENCE** drawn from the documented architecture choice and the absence of any pretrained-weight-loading code for voice, not a single explicit "trained from scratch" statement in one place — flagged accordingly.

### 4.2 Loss function, optimizer, and hyperparameters — per modality

**All four modalities share the same loss function class:** `models/common/arcface.py::ArcMarginProduct` — ArcFace additive angular margin softmax (Deng et al., CVPR 2019). FACT, `models/common/arcface.py:1-42`.

| Modality | ArcFace margin/scale | Optimizer | LR | Batch | Epochs | Split |
|---|---|---|---|---|---|---|
| Face | not in a config file | Adam | 1e-4 | 32 | 10 | per-identity (§ note 1) |
| Iris | not verified | not verified | not verified | not verified | not verified | none (§ note 2) |
| Fingerprint | m=0.5, s=64.0 | AdamW | 3e-4 | 64 | 30 | subject-disjoint 70/15/15, seed 42 |
| Voice | m=0.50, s=30.0 | AdamW | 1e-3 | 64 | 30 | per-identity 70/15/15 |

**Notes (full detail, not abbreviated in the table above):**
1. **Face:** values sourced only from `docs/PROJECT_FUNDAMENTALS.md:100-102` narrative, **not independently re-verified against notebook code cells** in this documentation pass; no dedicated `config.py` exists for face the way fingerprint/voice have.
2. **Iris:** no checkpoint was ever produced, so no final hyperparameters are attached to any committed artifact — training was attempted but never completed (see Part 3.2).
3. **Fingerprint** additionally uses: `label_smoothing=0.1`, a hard-negative hinge term (margin 0.3, weight 0.1) applied from epoch 10 onward, `weight_decay=1e-4`, `betas=(0.9, 0.999)`, `grad_clip_norm=1.0`, a 3-epoch warmup + cosine LR schedule down to `min_lr=1e-6`, batch = 16 identities × 4 samples/identity via a balanced sampler, and early stopping with patience 8.
4. **Voice** additionally uses: `weight_decay=1e-4`, a `CosineAnnealingLR` schedule, and early stopping with patience 5.

Fingerprint and voice hyperparameters are confirmed both as dataclass defaults (`models/fingerprint/config.py`, `models/voice/config.py`) **and** as the actual committed run configuration (`models/fingerprint/saved/fingerprint_config.json`, `models/voice/saved/training_config.json`), which match the defaults exactly. Face's hyperparameters come only from `docs/PROJECT_FUNDAMENTALS.md`'s narrative — **not independently re-verified against notebook code cells** in this documentation pass (the research agent reading notebook content only opened markdown headers, not code cells, per its task scope).

### 4.3 Training infrastructure notes (real, documented)

- Fingerprint training uses balanced P/K batch sampling (`models/fingerprint/sampler.py`), mixed precision, and hard-negative mining starting at epoch 10 — `models/fingerprint/train.py`.
- Voice training ran on **CPU**, not GPU: Kaggle's free-tier Tesla P100 (compute capability 6.0) was unsupported by the preinstalled PyTorch build; `models/voice/utils.py::detect_device()` smoke-tests CUDA with a real op (not just `torch.cuda.is_available()`) and falls back to CPU. Training completed all 30 epochs "in a little under 8 hours on CPU." **MEASURED RESULT**, `docs/VOICE_MODEL.md:114-121`.
- A real bug in the training pipeline was found and fixed regarding `shutil.make_archive` hanging indefinitely on Windows right after `torch.load` in the same directory — fixed with explicit `zipfile` writes (`models/voice/export.py`, regression-tested in `tests/test_voice_export.py`).

---

## PART 5 — Face Pipeline

### 5.1 Full pipeline trace (checklist form)

| Stage | Implementation | File:function |
|---|---|---|
| Frontend camera capture | `canvas.drawImage(video, 0, 0)` | `frontend/src/components/capture/{GuidedFaceCapture,FaceCapture}.tsx` |
| Image encoding | **PNG (lossless)** for both enrollment and authentication (unified this session — see Part 17, item 13) | `canvas.toBlob(..., 'image/png')` in both components |
| Backend image decoding | `cv2.imdecode` | `backend/utils.py::decode_image()` |
| Color conversion | `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` — OpenCV decodes BGR by default; explicitly converted to RGB immediately | `backend/utils.py:55-66` |
| Face detection | MTCNN, `detector.detect(image, landmarks=True)` | `preprocessing/face.py::FacePreprocessor.detect_and_align()` |
| Bounding box | Returned by `detect()`; used directly for the crop | Same |
| Face size (enrollment gate only) | `face_size_ratio = (y2-y1) / image_height` | `preprocessing/face.py` |
| Centering (enrollment gate only) | Normalized Euclidean offset of box center from image center | `preprocessing/face.py` |
| Angle checks (enrollment gate only) | Roll (eye-line angle) + yaw proxy (nose offset from eye midpoint / inter-eye distance), from the 5 MTCNN landmarks | `preprocessing/face.py` |
| Sharpness | Laplacian variance of the aligned crop | `preprocessing/face.py::BLUR_MIN_SHARPNESS` gate |
| MTCNN config | `image_size=160, margin=0, post_process=True`, device="cpu" | `preprocessing/face.py::FacePreprocessor._get_detector()` |
| Image size | 160×160×3, uint8 | Same |
| Margin | 0 (no padding around the detected box) | Same |
| Landmark behavior | Computed (5 points), used **only** for enrollment quality gating (roll/yaw), **not** for geometric alignment of the crop — see 5.2 below | `preprocessing/face.py` |
| Face embedding model | InceptionResnetV1 (`facenet-pytorch`) | `models/face/inference.py::FaceEmbedder` |
| Embedding dimensionality | 512 | `models/face/inference.py:17` |
| L2 normalization | Applied to every real embedding (and mock embeddings too) | `models/common/base_embedder.py::_l2_normalize()` |
| Five enrollment samples | Up to 5 captures; minimum 3 valid required (`MIN_VALID_POSES = 3`) | `backend/services/face_enrollment.py` |
| Centroid construction | `normalize(mean(L2-normalized embeddings))` | `embeddings/centroid.py::centroid_embedding()` |
| BioHash | HKDF → orthonormal projection → keyed quantize → keyed permute | `template_protection/biohash.py::generate_template()` |
| Protected template | Packed bit array, stored in `protected_templates.protected_template` | `backend/database/models.py::ProtectedTemplate` |
| Authentication comparison | Hamming similarity between fresh candidate template and the stored ACTIVE template | `template_protection/matcher.py::compare(metric="hamming")` |
| Threshold | **0.80**, operator-specified (see Part 18) | `evaluation/results/face_threshold.json` |

### 5.2 IMPORTANT CURRENT FINDING — no landmark-based geometric alignment

> **Update 2026-09-25:** this finding still holds for the deployed default (`FACE_ALIGNMENT=bbox`). A genuine 5-landmark
> similarity alignment now exists as the optional FACE_ALIGNED mode (`preprocessing/face.py::FacePreprocessorAligned`).
> It was evaluated with a retrained checkpoint (protected EER 8.67 % → 7.18 %) but is not deployed. See
> `evaluation/reports/FACE_ALIGNMENT_DECISION.md`.

**FACT, verified directly against the installed `facenet_pytorch` library source this session:**

`MTCNN.detect(img, landmarks=True)` computes both bounding boxes and 5-point landmarks. However, the actual face crop used by this pipeline is produced by `MTCNN.extract()` → `extract_face(img, box, image_size, margin)`, which takes **only the bounding box** — the landmarks are never passed to `extract_face`. There is **no landmark-based rotation/affine correction** anywhere in this pipeline, despite `preprocessing/face.py`'s own module docstring historically describing "alignment... driven by the detected landmarks" (a description now corrected in the current source to explicitly flag this gap — see the docstring update made this session).

This means: the crop is a plain axis-aligned bounding-box crop, resized to 160×160. A head that is rotated or tilted relative to the camera is fed to the embedding model **without** geometric normalization for that rotation. Landmarks are used in this codebase only as a **quality signal** (the roll/yaw checks in Part 6), never to warp the image.

### 5.3 Enrollment protocol change (documented this session)

| OLD UI (removed) | CURRENT UI |
|---|---|
| Front | Full Face / Neutral |
| Left (deliberate head turn) | Full Face / Natural Expression |
| Right (deliberate head turn) | Full Face / Natural Variation |
| Up (chin tilt) | Full Face / Natural Variation |
| Down (chin tilt) | Full Face / Final Capture |

**Why this was changed (FACT — direct engineering rationale from this session's investigation, recorded in `frontend/src/components/capture/GuidedFaceCapture.tsx`'s current comments):** the old protocol deliberately collected large head rotations (left/right/up/down), but the preprocessing pipeline performs no landmark-based rotation correction (5.2 above). A rotated enrollment capture therefore contributes an embedding whose alignment differs from a normal frontal live authentication capture, without any compensating correction — this measurably lowers the resulting centroid's similarity to a fresh frontal capture. The new protocol asks for five **mostly frontal** captures with only natural, small variation, so all five contributions to the centroid are geometrically consistent with what a live authentication capture will look like.

**Internal identifiers are unchanged.** The five pose keys (`front`, `left`, `right`, `up`, `down`) still exist as form-field names and internal dict keys — only the **user-facing labels and instructions** changed. No backend API contract change was required. FACT, `frontend/src/components/capture/GuidedFaceCapture.tsx`, `backend/services/face_enrollment.py::FACE_POSES`.

### 5.4 Presentation UX (documented this session)

The five accepted captures are now presented to the user as **one unified "Face Registration" process**, not five separately-named directional steps:
- Header: "Face Registration" (`frontend/src/pages/RegisterPage.tsx`'s `EnrollmentCard title` prop)
- Subtitle: "Look naturally at the camera. We'll take a few quick samples to create your face profile."
- Progress: a 5-dot indicator (filled = accepted) plus "X of 5 samples" text, driven by `acceptedCount = FACE_POSE_ORDER.filter((pose) => poses[pose]).length` — this can only increase when the backend has already returned `valid: true` for that capture, so a rejected/retaken capture never advances it. FACT, `frontend/src/components/capture/GuidedFaceCapture.tsx`.
- Per-accepted-capture feedback: "Sample captured ✓" (transient).
- Completion message: "Face registration complete" / "Your face profile has been created."

---

## PART 6 — Face Quality Gating

**Enrollment-only checks** (applied in `embeddings/pipelines.py::_evaluate_face_quality()`, called from both the live per-capture check endpoint and the final `embed_poses()` pass). Every threshold below was derived by measuring real MTCNN output on a real photograph (matplotlib's bundled `grace_hopper.jpg`, the same image this repo's own test suite already uses), not guessed. FACT, `preprocessing/face.py`.

| Check | Exact threshold (source) | Rejection verdict |
|---|---|---|
| Detection confidence | `probability < MIN_DETECTION_CONFIDENCE (0.90)` | `LOW_CONFIDENCE` |
| Sharpness | `sharpness < BLUR_MIN_SHARPNESS (25.0)` (Laplacian variance of the aligned crop) | `BLURRY` |
| Face size | `face_size_ratio < MIN_FACE_SIZE_RATIO (0.15)` (box height / frame height) | `TOO_SMALL` |
| Centering | `center_offset > MAX_CENTER_OFFSET (0.35)` (normalized Euclidean offset from frame center) | `OFF_CENTER` |
| Angle (roll) | `abs(roll_degrees) > MAX_ROLL_DEGREES (20.0)` (from eye-to-eye landmark angle) | `TOO_ANGLED` |
| Angle (yaw proxy) | `abs(yaw_ratio) > MAX_YAW_RATIO (0.20)` (nose offset from eye midpoint, normalized by inter-eye distance) | `TOO_ANGLED` |
| Multiple faces | `len(boxes) > 1` from `MTCNN.detect()` | `MULTIPLE_FACES` |
| No face | `boxes is None or len(boxes) == 0` | `NO_FACE` |

Checked **in this fixed order** — the first failing gate is the reported verdict (`embeddings/pipelines.py::_evaluate_face_quality`). Sharpness (`BLUR_MIN_SHARPNESS`) is the **pre-existing** gate from before this session's changes; confidence/size/centering/angle/multiple-faces are new, added this session.

### Retake behavior

A rejected capture is **never** added to the accepted set and never contributes to the centroid. The guided UI (`GuidedFaceCapture.tsx`) shows the rejection reason (`POSE_HINTS` in `backend/services/face_enrollment.py`) and lets the user retake the same slot immediately. `MIN_VALID_POSES = 3` — the backend tolerates up to 2 rejected/skipped captures out of 5 without blocking enrollment (this value was **not changed** this session; it predates the quality-gating work).

### Which checks apply during authentication?

**None of the quality gates above run during authentication.** `FacePipeline`'s authentication path (`preprocess()`) is a separate method from the enrollment path (`detect_and_align()`) and performs only detection + alignment, no quality scoring. A live authentication capture that fails to detect a face at all still raises `ValueError` (converted to HTTP 422 by `backend/utils.py::call_modality_service`), but none of the size/centering/angle/confidence/blur checks apply outside enrollment. FACT, verified directly this session by reading `preprocessing/face.py` in full.

---

## PART 7 — Fingerprint Pipeline

*(See also Part 2's fingerprint diagram.)*

### Input capture
Single image upload (no guided multi-capture, unlike face). `backend/api/enroll.py`.

### Preprocessing (`preprocessing/fingerprint.py`) — entirely classical computer vision, no learned preprocessing step
1. Grayscale conversion (`cv2.cvtColor`, RGB→gray)
2. CLAHE contrast enhancement (`cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))`)
3. Ridge normalization to `target_mean=100.0, target_var=100.0`
4. Gaussian blur (3×3 kernel)
5. 8-orientation Gabor filter bank (`cv2.getGaborKernel((15,15), sigma=4.0, lambd=10.0, gamma=0.5)`), max response across orientations combined
6. Min-max normalization back to `uint8`
7. Resize to 224×224 (`cv2.INTER_AREA`)
8. Channel-replicated to 3-channel RGB, then ImageNet mean/std normalized (`(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`) for the pretrained ResNet50 backbone

**No minutiae extraction and no binarization (black/white thresholding) is performed.** FACT, verified directly this session.

### Feature extraction / model
`FingerprintEmbeddingNet` — ResNet50 backbone (ImageNet-pretrained at training time; loaded fresh with real fine-tuned weights at inference time) → `Linear(2048→1024) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(1024→512)` projection head → L2-normalize. Output: 512-d embedding. `models/fingerprint/model.py`, `models/fingerprint/inference.py`.

### Template creation / protection
Identical mechanism to face and voice: `template_protection/biohash.py::generate_template()` — no fingerprint-specific variant exists. FACT.

### Matching
`template_protection/matcher.py::compare(metric="hamming")` — same shared function every modality uses.

### Threshold
**No calibrated threshold file exists for fingerprint** (`evaluation/results/fingerprint_threshold.json` does not exist, confirmed by directory listing). It falls back to `Settings.match_threshold = 0.9` (`backend/config.py:129`) via `backend/threshold_loader.py`.

### Enrollment / authentication
Both go through the **same generic `ModalityService.enroll()`/`.authenticate()`** used by every modality (`backend/services/base_service.py`) — no fingerprint-specific branch exists anywhere in `backend/api/enroll.py`, `authenticate.py`, or `verify.py`. FACT, confirmed by direct code trace this session.

---

## PART 8 — Voice Pipeline

### Microphone capture / audio format
WAV only — `preprocessing/voice.py` docstring states this is "per the spec." Frontend recording: `frontend/src/hooks/useWavRecorder.ts`.

### Sampling rate
Target: **16,000 Hz** (`TARGET_SAMPLE_RATE = 16_000`). Input is resampled via `scipy.signal.resample_poly` (GCD-reduced up/down factors) if the source rate differs.

### Preprocessing pipeline (`preprocessing/voice.py`)
1. Load WAV, convert int PCM → float32 in `[-1, 1]`
2. Mono conversion (channel average)
3. Resample to 16 kHz
4. VAD / silence trim — energy-based, per-sample keep-mask (frame_length=400, hop_length=160, threshold = 0.02 × peak frame energy)
5. Loudness normalization (target RMS 0.1, clipped to `[-1, 1]`)
6. Fixed-length segment: **4.0 seconds** (`CLIP_SECONDS = 4.0`) — random crop when training, center crop for inference; zero-padded if shorter
7. 80-bin log-mel filterbank (`N_MELS=80, N_FFT=400, HOP_LENGTH=160`), per-bin mean normalization across frames

Two real, documented bugs were found and fixed in this exact module during earlier project phases (see Part 17, items 1-2 for the debugging narrative). Both fixes are still in the current code: the VAD trim uses a per-sample mask (not frame concatenation, which used to duplicate overlapping samples), and short-clip padding happens in the mel/frame domain after normalization (not the waveform domain before it, which used to contaminate the per-clip mean).

### Feature extraction / model
ECAPA-TDNN — SpeechBrain's `speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN` module (the lower-level module, not the full pretrained `EncoderClassifier` pipeline — see Part 4's clarification on why this is not simply "using a pretrained checkpoint"). Output: **192-dimensional** L2-normalized embedding. `models/voice/inference.py`.

### Template creation / protection / matching
Same shared `template_protection/biohash.py` and `template_protection/matcher.py` as every other modality.

### Threshold
**0.80**, operator-specified (`evaluation/results/voice_threshold.json`) — not derived from a calibration run (`eer`/`far`/`frr`/`auc` all `null` in that file). See Part 18 for the documented real scores that motivated this value.

### Enrollment — two recordings, a consistency check unique to voice
Voice enrollment (`POST /enroll` with an `image` and a `confirm_image`) compares the two recordings' ECAPA-TDNN embedding cosine similarity **before anything is stored**:

| Band | Cosine similarity | Outcome |
|---|---|---|
| EXCELLENT | ≥ 0.85 | Enrolled |
| GOOD | 0.75 – 0.84 | Enrolled |
| FAIR | 0.60 – 0.74 | HTTP 409 `LOW_QUALITY_WARNING` — nothing stored unless the caller sets `accept_low_quality=true` |
| POOR | < 0.60 | HTTP 422 `ENROLLMENT_INCONSISTENT` — nothing stored |

`backend/services/recording_quality.py`. This check is **enrollment-only** — it never touches the authentication threshold, fusion, or template generation. The stored template comes from the **first** of the two recordings only.

### Authentication
Single recording, same generic `ModalityService.authenticate()` path as every other modality (via a thin `_VoicePipelineAdapter` in `backend/services/voice_service.py` that bridges `VoicePipeline.embed(waveform, sample_rate)`'s two-argument signature to the generic single-argument contract).

---

## PART 9 — Template Protection

### Why raw biometric embeddings are not stored

A raw embedding is a permanent, un-revocable representation of a person's biometric characteristic — if leaked, it cannot be "reset" the way a password can. `docs/TEMPLATE_PROTECTION.md:9-18`.

### What is stored instead

The output of a **cancelable, BioHashing-style transform**: `template_protection/biohash.py::generate_template()`.

```text
embedding (unit vector, dim D)
     |
     v
KeyMaterial = derive_key(master_secret, application_id, user_id, modality, key_version)
     |   HKDF-SHA256, three independent 32-byte seeds
     v
projection_matrix = build_orthonormal_projection(projection_seed, output_bits, D)
     |
     v
projected = projection_matrix @ embedding          -> real values in [-1, 1]
     |
     v
quantized = quantize(projected, threshold_seed)    -> bits, keyed per-position threshold
     |
     v
template = apply_permutation(quantized, permutation_seed)   -> the protected template
```

### The math, stage by stage

**1. Key derivation.** `derive_key(master_secret, application_id, user_id, modality, key_version)` returns three independent HKDF-SHA256 seeds, salted by a SHA-256 hash of the four identifying fields. Deterministic: same inputs always produce the same seeds. `template_protection/hkdf_keys.py`.

**2. Orthonormal projection.** A `(output_bits, embedding_dim)` matrix, rows are unit vectors, built via QR decomposition of a Gaussian matrix seeded from `projection_seed`. True orthonormality across every row only holds when `output_bits <= embedding_dim`; the code composes multiple orthonormal blocks when this isn't the case (a documented, not hidden, tradeoff). `template_protection/transform.py`.

**3. Keyed quantization.** Each projected value is binarized against a per-position random threshold drawn from `Normal(0, 0.5 * std(projected))`, seeded by `threshold_seed` — centered at zero (a sign-split of the embedding-driven signal) rather than a fixed magic number, so quantization doesn't collapse every embedding to the same template regardless of input.

**4. Permutation.** The bit vector is reordered by a key-derived random permutation — adds no entropy alone, but combined with the keyed projection/quantization means two templates derived under different keys share no positional structure.

### Output-bit setting

`template_protection/biohash.py::DEFAULT_OUTPUT_BITS = 128` is the **generic library default** used when a caller doesn't specify `output_bits`. **The project's actual runtime setting is 256 bits**: `backend/config.py::Settings.template_bits: int = 256`, passed explicitly by `backend/services/base_service.py::build_entry()` for every real enrollment. This is a real distinction worth being precise about — the library default and the deployed configuration are different numbers. FACT, verified this session directly (stored protected templates are 32 bytes = 256 bits).

### Hamming similarity

```text
S_hamming = matching_bits / total_bits
```

where `matching_bits` is the count of bit positions where the candidate and stored templates agree, and `total_bits` is `output_bits` (256 in this deployment). `template_protection/matcher.py::compare(metric="hamming")`.

### Why MASTER_SECRET matters / why losing it makes old templates unusable

Every HKDF seed above is derived from `MASTER_SECRET`. Changing `MASTER_SECRET` changes every downstream seed, which changes the projection matrix, quantization thresholds, and permutation entirely. A template generated under one secret compared against a candidate generated under a **different** secret produces a Hamming similarity statistically indistinguishable from comparing two **unrelated** bit strings (empirically ≈0.50, verified this session — see Part 17 item 4-6). This is not a bug in the comparison; it is the expected, correct behavior of a keyed transform when the key changes. See Part 10.

### Security assumptions, stated honestly (not overclaimed)

- **Information-lossy, not cryptographically one-way.** A many-to-one map (512 or 256 real dimensions → 256 bits); infeasible to reconstruct the embedding from the template *alone*, but this is a data-transformation argument, not a computational-hardness proof.
- **Key compromise weakens the guarantee.** If an attacker recovers both a protected template and the key material, partial reconstruction becomes a studied BioHashing-literature attack.
- **Constant-time comparison is partial** — only the exact-match fast path (`constant_time_equals`) is genuinely constant-time; Hamming-distance computation itself is not hardened against timing side channels beyond that.
- **Mock-mode embeddings (iris) produce mock-mode templates** — not biometrically meaningful.

`docs/TEMPLATE_PROTECTION.md:183-214`.

---

## PART 10 — MASTER_SECRET Key Continuity

### Why the original issue occurred

During this session's development, the local backend was restarted with a **freshly generated** `MASTER_SECRET` because no `.env` file existed at session start. The database already contained protected templates (from `2026-09-17`, a prior session) generated under a **different, now-unrecoverable** secret. Nothing in the system previously verified that a *newly supplied* `MASTER_SECRET` was the *same* one that had protected existing templates — any syntactically valid string was silently accepted.

### Why old templates became incompatible

As explained in Part 9: every seed downstream of `MASTER_SECRET` changes when the secret changes. Authenticating with the new secret against templates generated under the old one produces near-chance Hamming similarity, indistinguishable from an unrelated/impostor comparison — **MEASURED RESULT**, reproduced this session with the real `generate_template`/`compare` functions on synthetic vectors: same embedding + same key → 1.0 similarity; same embedding + different key → 0.496 similarity (n=1 embedding pair, multiple trials averaged, synthetic — not a formal benchmark).

### The fix: a key-continuity mechanism

**New files this session:** `backend/secret_fingerprint.py`, `backend/key_continuity.py`, `backend/database/models.py::MasterSecretFingerprint`, `scripts/rotate_master_secret.py`.

**Fingerprinting MASTER_SECRET (never the secret itself):** `backend/secret_fingerprint.py::fingerprint_master_secret()` computes a one-way HKDF-SHA256 derivation over a **fixed public context** string (`b"master_secret_fingerprint/v1"`), producing a 32-byte value. This is deliberately a **separate** derivation from `template_protection/hkdf_keys.py::derive_key()` — it shares the same cryptographic primitive class (HKDF-SHA256) but a different, unrelated context, and is never used to derive any template key. Storing this fingerprint never exposes the secret (HKDF is one-way); it only answers "is this the same secret as before?"

### Startup states (implemented)

| State | Condition | Behavior |
|---|---|---|
| **A — initial/uninitialized** | No templates exist, no fingerprint recorded | Auto-initializes: records the current secret's fingerprint. Safe because nothing is at risk yet. |
| **B — mismatch-by-absence** | Templates exist, no fingerprint recorded | **Hard fail.** Refuses to start. Never assumes the current secret is correct. |
| **C — matching/current** | Fingerprint recorded, current secret's fingerprint matches | Proceeds normally. |
| **D — mismatch** | Fingerprint recorded, current secret's fingerprint does NOT match | **Hard fail**, regardless of whether templates currently exist. |
| **E — no templates, fingerprint already recorded** | Fingerprint exists, no templates yet | Resolves via the same match/mismatch logic as C/D (folds into "proceed" if it matches). |

`backend/key_continuity.py::KeyContinuityState` (enum: `ok`, `no_fingerprint_recorded`, `mismatch`, `no_templates_yet`), `evaluate_key_continuity()` (read-only, used by `/system/health`), `enforce_key_continuity_at_startup()` (side-effecting, called from `backend/main.py`'s `lifespan()` hook, raises `KeyContinuityError` on hard-fail states).

### Rotation procedure

`scripts/rotate_master_secret.py` — a standalone CLI, **never** invoked automatically by the running server. Requires the operator to type an exact confirmation phrase acknowledging that existing templates protected under a different secret will become permanently unverifiable (never deleted, never modified — they simply stop matching, "the same outcome as a lost password"). Records the new fingerprint via `crud.rotate_master_secret_fingerprint()` — the **only** code path allowed to overwrite an existing fingerprint row; `initialize_master_secret_fingerprint()` (used by State A's auto-init) explicitly refuses to overwrite an existing row.

### What happens when templates exist under another secret

State B applies: the server refuses to start with a clear error message directing the operator to the rotation script, rather than starting "successfully" and silently authenticating every enrolled user against effectively-random data (the exact failure mode this mechanism exists to prevent — see Part 17).

### Why the actual secret is never stored

`MasterSecretFingerprint.fingerprint` is a one-way HKDF output — there is no code path that reverses it back to `MASTER_SECRET`. The secret itself lives only in the process environment / `.env` file (gitignored), consistent with `docs/PRIVACY_AND_SECURITY.md`'s stated principle. FACT.

---

## PART 11 — Enrollment

### Complete flow

```text
User arrives at checkpoint (frontend/src/pages/CheckpointPage.tsx)
        |
        v
"Enroll Biometrics" (continuing the current session user) OR
"+ New Registration" (switches to a brand-new user_id first)
        |
        v
RegisterPage.tsx — independent per-modality enrollment cards
        |
        +--> Face: 5 guided captures -> quality gates -> centroid -> ONE protected template per template set
        +--> Fingerprint: 1 image -> protected template per template set
        +--> Voice: 2 recordings (consistency-checked) -> protected template per template set
        |
        v
Each modality's template is written into every live TEMPLATE SET
(default pool size 4: set 1 ACTIVE, sets 2-4 STANDBY)
        |
        v
"Continue to {building}" -> back to CheckpointPage, now showing this user's real enrollment status
```

### Face specifically

Collects **five accepted samples**, builds **one centroid-based protected template** (not five separate templates). `embeddings/centroid.py::centroid_embedding()`, `backend/services/base_service.py::enroll_poses()`. The five raw embeddings are held in memory only for the duration of the one `/enroll` request and discarded (`embeddings.clear()`) immediately after the centroid is computed — never persisted.

### "New Registration" — implemented this session

`frontend/src/hooks/useAuthSession.ts`: `registerNewUser()` generates a new `user_id` (`OPERATOR-XXXXXX` format, same scheme the app already used for its single-session identity), switches the active session to it, and adds it to a local roster (`knownUsers`). The previous user's `user_id` and templates are never touched. `frontend/src/pages/CheckpointPage.tsx`'s "+ New Registration" button calls this before navigating to `/register`.

### Returning users

The frontend already persisted one `user_id` per browser in `localStorage` (`biometric-demo.user-id`) before this session's work — a returning visit to the same browser reuses the same ID automatically, so an already-enrolled user does not need to re-enroll. This session's addition (`knownUsers`) extends this to support **more than one** person on the same browser/machine, each remembered independently.

### How multiple known users are exposed in the frontend

`CheckpointPage.tsx` shows a `<select>` dropdown of `knownUsers` when more than one is known; a single known user is shown as plain text. **This is a purely local (this-browser-only) roster stored in `localStorage` (`biometric-demo.known-users`) — not a backend concept.**

**Do not claim a centralized user-listing API exists — it does not.** Verified directly this session: `backend/api/user.py` exposes only `GET /user/{id}` (requires knowing the ID already), `GET /user/{id}/enrollment-status`, and `DELETE /user/{id}` — there is no endpoint that lists all enrolled users. The `knownUsers` roster works entirely by the frontend remembering IDs it has itself created or switched to; it queries each one's real enrollment status via the existing per-ID endpoint. **FACT.**

### What is stored where

| Data | Where | Persisted? |
|---|---|---|
| Protected templates (bits) | `protected_templates` table | Yes (SQLite/Postgres) |
| Raw images/audio | Nowhere | No — discarded after the single preprocessing call |
| Raw embeddings | Nowhere | No — discarded after being protected |
| `user_id` (current session) | Browser `localStorage` | Yes, per-browser |
| `knownUsers` roster | Browser `localStorage` | Yes, per-browser, purely local |
| `MASTER_SECRET` | Process environment / `.env` | Not in the database, ever |
| `MASTER_SECRET`'s fingerprint (not the secret) | `master_secret_fingerprint` table | Yes, one row |

---

## PART 12 — Authentication

### Trace

```text
User selection (CheckpointPage dropdown, or the single known user)
        |
        v
AuthenticatePage.tsx — user picks which enrolled modalities to present this session
        |
        v
Capture (per selected modality)
        |
        v
POST /authenticate/fusion  (backend/api/fusion.py)
        |
        +--> Any submitted modality not enrolled -> HTTP 409 ENROLLMENT_REQUIRED
        |    (nothing evaluated; not counted as a failed authentication)
        |
        v
authenticate_samples()  (backend/services/authentication.py)
        |    for each submitted modality: ModalityService.authenticate()
        |    -> preprocess -> embed -> re-protect under the ACTIVE set's key
        |    -> Hamming compare against the stored ACTIVE template
        v
Per-modality scores + per-modality pass/fail
        |
        v
fusion/policy.py::evaluate_fusion_policy()  (Part 13)
        |
        v
ACCESS_GRANTED / ACCESS_DENIED
        |
        v
ResultPage.tsx  ->  DecisionSummary  +  (DEBUG_SCORES only) FusionDiagnosticsPanel
```

### Genuine vs. impostor matching, conceptually

A **genuine** attempt compares a live capture against the ACTUAL enrolled person's stored template, under the same `MASTER_SECRET`-derived key — expected to produce a high Hamming similarity if the underlying biometric matches well. An **impostor** attempt compares a different person's live capture against someone else's stored template, under that stored template's key — expected to produce a similarity near the "unrelated" floor (empirically ≈0.5–0.6 in this project's measurements). The system has **no separate "impostor mode"** — every `/authenticate` request is compared against whichever `user_id` the caller supplies; the genuine/impostor distinction is purely about whether the presenter is actually the enrolled person.

---

## PART 13 — Fusion

### Exact current implementation

**Scores.** Each submitted modality's score is a Hamming similarity in `[0, 1]` (Part 9). No normalization is applied before fusion — the raw Hamming similarities are combined directly. `fusion/score_fusion.py`.

**Weights.** Equal among submitted modalities: weight = `1/N` where `N` is the number of modalities actually submitted in that request. `fusion/score_fusion.py::resolve_normalized_weights()` — every present modality defaults to weight `1.0` before renormalization; no caller in this codebase currently passes custom weights.

**Fused score:**

```text
S_fused = sum(w_i * S_i) / sum(w_i)
```

where `S_i` is modality `i`'s Hamming similarity and `w_i` is its weight (equal for all submitted modalities in the current deployment). `fusion/score_fusion.py::fuse_scores()`.

**Fusion threshold.** `fusion_threshold = mean(per-modality thresholds actually used)` — computed for **informational display only**. `backend/services/authentication.py`.

**Policy.** Default: `ALL_REQUIRED` — `fusion/config.py::DEFAULT_FUSION_POLICY`. Two other policies exist in code (`AT_LEAST_TWO`, requires exactly 3 modalities submitted; `WEIGHTED`, the original weighted-average behavior with a per-modality floor veto) but `ALL_REQUIRED` is what the frontend and all documented flows use.

**The final decision is NOT simply `fused_score >= threshold`.** Under `ALL_REQUIRED`: `authenticated = (every submitted modality's own score >= its own per-modality threshold)`. `fusion/policy.py::evaluate_fusion_policy()`.

### Numerical example (illustrative — not a recorded result unless stated otherwise)

Suppose Face = 0.77 (below its 0.80 threshold) and Voice = 0.88 (above its 0.80 threshold):

```text
S_fused = (0.77 + 0.88) / 2 = 0.825
```

The fused number alone (0.825) is *above* a 0.80-style threshold. But under `ALL_REQUIRED`:
- Face: 0.77 < 0.80 → **fails**
- Voice: 0.88 ≥ 0.80 → passes

Because not every submitted modality passed, the decision is **ACCESS DENIED**, even though the fused average would suggest a pass. This is by design — `fusion/config.py`'s own docstring documents this as the fix for a specific audit finding: the older plain-weighted-average behavior let one strong modality compensate for another's individual failure (e.g. Face=1.0 pass + Fingerprint=0.82 fail averaged to 0.91, clearing a 0.9 threshold anyway) — `ALL_REQUIRED` closes that gap.

A **real recorded** instance of this exact mechanism, from this project's own actual logged attempts (see Part 18): Face = 0.7734, Voice = 0.8828 → fused = 0.8281, which is above the 0.80-style threshold on average — but Face individually failed its own 0.80 threshold, so under `ALL_REQUIRED` the recorded decision was **DENIED**.

---

## PART 14 — Fusion Diagnostics

### Why it was added

To let a developer see, locally, exactly which per-modality score/threshold/status combination produced a given fusion decision — without exposing anything in production. Added this session (`fusion/diagnostics.py`, `backend/database/schema.py::AuthenticationDecision.fusion_diagnostics`, `frontend/src/components/biometric/FusionDiagnosticsPanel.tsx`).

### DEBUG_SCORES

An existing project-wide environment flag (`Settings.debug_scores`, `backend/config.py`), predating this session. When `false` (production default), no per-modality score/threshold/diagnostic field ever leaves the backend — `response_model_exclude_none=True` drops every debug-only field. When `true`, `backend/services/authentication.py::to_public_response()` populates them, including `fusion_diagnostics`.

### Backend diagnostic response — fields

`fusion/diagnostics.py::build_fusion_diagnostics()` reshapes values **already computed** by the real fusion engine — it computes nothing new:

- One entry per modality (`face`, `fingerprint`, `voice`): `score` (nullable), `threshold` (nullable), `verified` (nullable bool), `status` (`verified` / `failed_below_threshold` / `not_presented`)
- `weights`: the exact per-modality weight fraction `fuse_scores()` used (only for submitted modalities — an absent modality gets no weight entry at all, never an implied zero)
- `fused_score`, `threshold` (the fusion threshold), `policy`, `access_granted`

**Why no `score_unavailable` status exists:** `authenticate_samples()` fails the *whole* request closed (a 409/422/500, never a 200) if any submitted modality can't produce a score — so a successful response can never contain a submitted-but-scoreless modality. `fusion/diagnostics.py`'s own module docstring documents this explicitly.

### Frontend panel

`FusionDiagnosticsPanel.tsx` — an expandable section on the Result page, rendered only when `result.fusion_diagnostics` is present (i.e., automatically local-dev-only via the backend's `DEBUG_SCORES` gate, no separate frontend flag). Shows per-modality score/status cards, a "MODALITY SCORES → FUSION → FINAL DECISION" flow indicator, and — when the fused score alone would have suggested a different outcome than the real policy-driven decision — an explicit note explaining the mismatch, so the UI never implies the fused score alone determines access.

### Example response (illustrative — labeled as such)

```json
"fusion_diagnostics": {
  "face":        {"score": 0.97, "threshold": 0.9, "verified": true,  "status": "verified"},
  "fingerprint": {"score": null, "threshold": null, "verified": null, "status": "not_presented"},
  "voice":       {"score": 0.74, "threshold": 0.8, "verified": false, "status": "failed_below_threshold"},
  "weights": {"face": 0.5, "voice": 0.5},
  "fused_score": 0.855,
  "threshold": 0.85,
  "policy": "ALL_REQUIRED",
  "access_granted": false
}
```
This example was generated from the real `evaluate_fusion_policy`/`build_fusion_diagnostics` functions during this session's implementation work (synthetic input scores, real code) to demonstrate the mechanism — it is not a recorded live-user authentication attempt.

---

## PART 15 — Database

Engine: SQLAlchemy, SQLite by default (`sqlite:///./biometric.db`), swappable to Postgres via `DATABASE_URL`. `backend/database/session.py`.

### Tables

**`users`** — `id` (PK, caller-supplied string, e.g. `OPERATOR-XXXXXX`), `username` (nullable), `created_at`. One-to-many to `protected_templates`, cascade delete-orphan.

**`protected_templates`** — the core biometric-derived table. Key columns: `template_id` (PK), `user_id` (FK), `modality`, `application_id`, `template_version` (BioHash format version), `key_version` (HKDF rotation counter), `output_bits`, `protected_template` (`LargeBinary` — the *only* biometric-derived bytes ever stored), `is_active`, `template_status` (ACTIVE/STANDBY/REVOKED), `template_set_version`, `template_set_status`, plus activation/revocation timestamps. Two partial unique indexes enforce: (1) at most one ACTIVE row per (user, modality, application), (2) at most one non-revoked row per (user, application, template_set_version, modality).

**`master_secret_fingerprint`** — added this session. Singleton row (`id` fixed at 1). `fingerprint` (`LargeBinary`, 32 bytes, one-way HKDF output — never the secret). `created_at`.

**`audit_logs`** — one row per `/authenticate`, `/verify/*`, or `/authenticate/fusion` call. Metadata only: `user_id`, `building_id`, `modality_list`, `similarity_scores` (per-modality Hamming scores — recoverable numbers, never a template or embedding), `thresholds_used`, `fusion_score`/`fusion_similarity`, `fusion_policy`, `authentication_state` (ACCESS_GRANTED/ACCESS_DENIED/ENROLLMENT_REQUIRED), `submitted_modalities`/`enrolled_modalities`/`authenticated_modalities`, `authenticated`, `latency_ms`, `template_versions`/`key_versions` (dicts, lets a reviewer see key rotation over time). No relationship/FK to `users`.

**`enrollment_events`** — one row per enrollment attempt: `outcome` (`ENROLLED` / `INCONSISTENT`), `user_id`, `application_id`, `modality`, `created_at`. Exists because a rejected enrollment stores nothing else — this is what turns a not-yet-enrolled voice into `RETRY_REQUIRED` rather than `NOT_REGISTERED`.

### What is persisted vs. intentionally discarded

**Persisted:** protected template bytes, audit metadata (scores/thresholds/decisions, never raw biometric data), the one-way `MASTER_SECRET` fingerprint.

**Intentionally never persisted:** raw images/audio, raw embeddings, `MASTER_SECRET` itself, any reversible representation of a biometric characteristic.

No actual user IDs, secrets, or biometric contents from this project's local database are reproduced anywhere in this document.

---

## PART 16 — Security and Privacy

| Item | Status |
|---|---|
| Raw embeddings not persisted | **Implemented security measure** — `template_protection/biohash.py` output is the only thing ever written to `protected_templates`; raw embeddings are deleted from memory immediately after protection |
| Protected templates instead of raw data | **Implemented** |
| `MASTER_SECRET` handling (env-only, never in DB, never logged/returned) | **Implemented** |
| `.env` gitignored, `.env.example` ships no real value | **Implemented** |
| Key continuity (Part 10) | **Implemented** this session |
| No raw biometric data in debug logs | **Implemented** — `backend/services/face_debug.py`'s diagnostic logging is restricted by construction to scalar floats only (cosine similarities, means/stds), never vectors/images/templates; verified by dedicated tests this session (`tests/test_face_debug.py`) |
| `DEBUG_SCORES` behavior | **Implemented** — gates all per-modality/diagnostic fields off by default |
| `localStorage` user roster | **Implemented, local-only** — a convenience, not a security boundary; anyone with access to the browser profile can see/switch the roster |
| No user/API-caller authentication (no login, no API keys) | **Current limitation**, explicitly documented: `README.md:342-345` — "There is no user/API-caller authentication... every endpoint in `backend/` is reachable by anyone who can reach the process" |
| No liveness / presentation-attack detection | **Future security enhancement** (on hold) |
| No deepfake detection | **Future security enhancement** (on hold) |
| Replay-attack risk (no challenge-response binding a request to a live capture) | **Current limitation**, explicitly documented in `docs/PRIVACY_AND_SECURITY.md:59-65` |

**The system does not claim to be invulnerable.** `docs/PRIVACY_AND_SECURITY.md` and `docs/TEMPLATE_PROTECTION.md` both explicitly disclaim cryptographic one-wayness, regulatory compliance, and formal security proofs — this document preserves those same disclaimers rather than overstating them.

---

## PART 17 — Debugging / Major Problems Found

Presented chronologically as actually investigated (this session's work, plus pre-session milestones recovered from git history and project docs). Note: this session's own fixes (items 4 onward) were committed as a **single** commit (`5c67be1`) rather than incrementally, so there is no fine-grained git history for them individually — the narrative below reflects the actual investigation order within this conversation, not separate commits.

### 1. Initial multimodal architecture
**Problem:** establish one shared interface across four very different modalities. **Fix:** `BaseEmbedder.extract_embedding()` contract, `ModalityPipeline` base class. **Evidence:** `models/common/base_embedder.py`, `docs/ARCHITECTURE.md`. Pre-session, git commit `13594ed` "Phase 1: multimodal preprocessing, embedding models, evaluation, and Colab training notebooks."

### 2. Voice preprocessing bug — frame-overlap duplication in VAD trim (pre-session)
**Problem:** genuine same-speaker recaptures scored as low as -0.18 cosine similarity. **Investigation:** traced to `_trim_silence` concatenating overlapping frames (hop_length 160 < frame_length 400), duplicating samples. **Root cause:** a 4s/64000-sample clip became ~159200 samples. **Fix:** per-sample keep-mask instead of frame concatenation. **Verification:** regression test added. Git commit `7766073` "Fix voice preprocessing bug causing genuine users to be denied."

### 3. Voice preprocessing bug — mel-domain padding contamination (pre-session)
**Problem:** a second bug in the same module: waveform-domain silence padding before mel extraction dragged down the per-clip mel-normalization mean for real speech frames too. **Fix:** pad the mel *frames*, computed on real content only, not the waveform. **Verification:** `docs/AUTHENTICATION_RELIABILITY_REPORT.md`'s "Root Cause (prior pass)" section documents before/after cosine similarity: 0.256 → 0.991 for a 3.6s-vs-4.0s recapture.

### 4. Genuine face authentication low-score investigation (this session)
**Problem:** genuine face scores of 0.6992/0.7422/0.7734 against a 0.80 threshold, while voice scores were consistently higher (0.85-0.88). **Investigation:** traced the entire enrollment and authentication face pipeline; compared preprocessing byte-for-byte; verified `facenet_pytorch`'s actual source to confirm no landmark-based alignment exists. **Root cause candidates evaluated:** mirroring (ruled out — canvas.drawImage reads the raw, unmirrored video buffer regardless of CSS), mock embedder (ruled out — real 112MB checkpoint confirmed loaded), enrollment/authentication pipeline divergence (ruled out — near-identical, one minor JPEG-vs-PNG encoding difference found), centroid averaging (tested via simulation — found to *help*, not hurt, similarity).

### 5. Discovery of MASTER_SECRET mismatch (this session)
**Problem:** during this session, a freshly generated `MASTER_SECRET` was written to a new `.env` (none existed at session start) while the local database still held templates from a prior session (2026-09-17). **Investigation:** audit-log inspection showed a real face template's `created_at` timestamp preceded the current `.env`'s creation timestamp; no persisted OS-level or file-level record of the original secret existed anywhere. **Root cause confirmed:** the running backend was authenticating against templates protected under a secret it no longer had.

### 6. Verification — same embedding + same key ≈ 1.0 similarity (this session)
**MEASURED RESULT** (synthetic vectors, real `generate_template`/`compare` code, throwaway test key, this session): same embedding compared against itself under the same key → Hamming similarity 1.0000.

### 7. Verification — same embedding + different key ≈ chance-level Hamming similarity (this session)
**MEASURED RESULT** (same setup): same embedding, generated under two different keys → Hamming similarity 0.4961 — statistically indistinguishable from two unrelated bit strings, and numerically consistent with the real observed 0.6992–0.7734 range once accounting for genuine embedding-space variation (see Part 18's cosine-mapping table).

### 8. Key continuity mechanism designed and implemented (this session)
**Fix:** the full mechanism described in Part 10 — `backend/secret_fingerprint.py`, `backend/key_continuity.py`, `MasterSecretFingerprint` table, `scripts/rotate_master_secret.py`. **Verification:** the real local database was driven through State B (hard-fail, confirmed via an actual failed `uvicorn` startup with "Application startup failed. Exiting."), then transitioned to State C via the rotation script, confirmed via `/system/health` returning `"key_continuity": "ok"` and the pre-existing templates remaining byte-for-byte untouched.

### 9. Fusion diagnostics implementation (this session)
Added the `fusion_diagnostics` response field and frontend panel (Part 14) specifically to make the ALL_REQUIRED-vs-fused-score distinction (item 10 below) visible to a developer without needing to read backend logs.

### 10. Discovery/confirmation that ALL_REQUIRED — not the fused score — controls final access (this session, but the underlying fix predates this session)
Confirmed via a real fusion-diagnostics-instrumented authentication attempt and via a dedicated regression test (`tests/test_fusion_diagnostics.py::test_all_required_denies_despite_a_fused_score_above_threshold`) that a fused score above threshold does not grant access if any individual submitted modality failed its own threshold. The underlying `ALL_REQUIRED` policy itself was introduced pre-session (git commit history shows it fixing an audit-flagged compensatory-averaging vulnerability), but this session's fusion-diagnostics work is what made the mechanism directly observable.

### 11. Face enrollment pose problem — geometric alignment gap (this session)
**Problem:** the guided enrollment protocol deliberately collected front/left/right/up/down head-turn poses. **Investigation:** direct read of the installed `facenet_pytorch` library source confirmed `extract_face()` uses only the bounding box — landmarks are computed but never used for rotation correction. **Root cause:** a deliberately-rotated enrollment capture, combined with no compensating geometric alignment, plausibly explains a meaningful share of the observed low genuine similarity (see Part 18's caveats — this was not conclusively isolated from capture variation with the available data).

### 12. Enrollment quality gating added (this session)
**Fix:** the checks in Part 6, each threshold measured against real MTCNN output rather than guessed. **Verification:** 18 new tests (`tests/test_face_quality_gating.py`), all passing, including real-MTCNN integration tests proving a too-angled real capture is rejected while a mildly-varied one still passes.

### 13. JPEG vs. PNG encoding discrepancy (this session)
**Problem:** enrollment captured JPEG (q=0.92, lossy); authentication captured PNG (lossless) — an unnecessary, previously-unflagged inconsistency between the two capture paths. **Fix:** unified both to PNG (the less invasive direction — one file changed instead of two).

### 14. New full-face enrollment protocol (this session)
Described in Part 5.3 — replaced deliberate head-turn instructions with five mostly-frontal captures, and (in a follow-up refinement) simplified the UI presentation from five separately-labeled steps to one unified "Face Registration" process (Part 5.4).

### 15. Persistent user selection / New Registration UI (this session)
Described in Part 11 — added a local `knownUsers` roster and an explicit "+ New Registration" action, reusing the existing per-`user_id` backend endpoints rather than inventing a new backend identity system (confirmed no backend change was required or made).

---

## PART 18 — Actual Experimental Results

### Sample size — stated explicitly

**All face/voice/fusion numbers discussed in this session (0.6992, 0.7422, 0.7734 for face; 0.8477, 0.8672, 0.8828 for voice) came from a small number of manual authentication attempts by one person during development/debugging, not a formal benchmark.** The exact number of attempts and their source (manual observation reported during this conversation vs. a committed log file) could not be independently re-verified against a permanent repository artifact in this documentation pass — **Not verified in the current repository** as a committed dataset; treat these as development-diagnostic observations, reported here for traceability, not as a statistically meaningful sample (n is on the order of a handful of attempts by a single person).

### The one specific worked example given during this session

Face = 0.7734, Voice = 0.8828:

```text
S_fused = (0.7734 + 0.8828) / 2 = 0.8281
```

Policy = `ALL_REQUIRED`. Face's own score (0.7734) is below the face threshold (0.80) → Face fails individually → **Final decision: ACCESS DENIED**, regardless of the fused score (0.8281) being numerically above a 0.80-style threshold. This is reported as a real observation from this session's manual testing, not a synthetic illustration — but again, n=1 for this specific combination.

### Formally committed experimental results (distinct from the manual face/voice observations above)

These **are** real, committed artifacts, not manual observations:

| Modality | File | num_samples | genuine pairs | impostor pairs | EER | Accuracy | AUC |
|---|---|---|---|---|---|---|---|
| Face | `evaluation/results/face_metrics.csv` | Not recorded in this file (sparser schema) | Not recorded | Not recorded | 0.010 | 0.990 | 0.999 |
| Fingerprint | `evaluation/results/fingerprint_metrics.csv` | 900 | 4,050 | 400,500 | 0.3077 | 0.6923 | 0.7644 |
| Voice | `evaluation/results/voice_metrics.csv` | 751 | 13,101 | 268,524 | 0.0229 | 0.9771 | 0.9967 |

**Important caveats, stated in the project's own docs and preserved here:**
- Face's result is **closed-set** — the same 62 LFW identities were used for both fine-tuning and evaluation. Not an open-set generalization claim.
- Fingerprint is explicitly self-described in the project's own documentation as "the weakest modality" (`docs/PROJECT_FUNDAMENTALS.md:131-132, 355`) — EER ≈31% is a materially weak result.
- These three EER/AUC numbers are **raw-embedding-space** evaluation results (`evaluation/{fingerprint,voice}_metrics.py`, `evaluation/roc.py`), computed independently of the **deployed protected-template Hamming thresholds** (0.80/0.80/0.90 fallback) — they are related but not the same measurement domain, and no code in this repository numerically reconciles the two. Reported separately rather than conflated.
- Iris has **no** metrics file at all — no checkpoint was ever trained.
- `face_threshold.json` and `voice_threshold.json` are both marked `"source": "operator-specified"` with `eer`/`far`/`frr`/`auc` all `null` — **these are not calibration results**, they are operator judgment calls documented alongside the real genuine-score observations that motivated them (face: 0.738–0.820 range, mean ≈0.77; voice: 0.797/0.848/0.879).
- The four `template_set_*.csv` files in `evaluation/results/` are template-protection **lifecycle logic** tests run against synthetic/stub embeddings — they validate revocation/diversity/exhaustion behavior, not biometric recognition accuracy. Do not conflate them with the metrics table above.

**Do not treat the handful of manual face/voice observations discussed in this session as a statistically valid benchmark.** They are development diagnostics that motivated real engineering decisions (the key-continuity fix, the enrollment protocol change, the quality-gating thresholds), not a scientific evaluation. The formally committed CSV/JSON results above are real measurements but come with their own explicitly documented caveats (closed-set face, weak fingerprint, operator-set rather than calibrated production thresholds).

---

## PART 19 — Testing

### Current total

**512 tests collected**, confirmed this session via `python -m pytest tests/ --collect-only -q` (512 tests collected in 12.02s, zero collection errors) and via a full run (512 passed in the most recent complete suite execution this session).

### What the suite verifies (by category, not exhaustive)

| Area | Representative test modules |
|---|---|
| Template protection math (HKDF, BioHash, transform stages) | `tests/test_hkdf.py`, `tests/test_biohash.py`, `tests/test_transform.py`, `tests/test_matcher.py` |
| Fusion (`fuse_scores`, policy, diagnostics) | `tests/test_fusion.py`, `tests/test_fusion_policies.py`, `tests/test_fusion_endpoint.py`, `tests/test_fusion_diagnostics.py` |
| Key continuity | `tests/test_key_continuity.py` (19 unit tests, all 5 states), `tests/test_key_continuity_startup.py` (4 real-`TestClient` end-to-end tests, including asserting the real app fails to start in hard-fail states) |
| Face enrollment/quality gating | `tests/test_face_enrollment.py`, `tests/test_face_quality_gating.py` (18 tests, real-MTCNN integration + pure unit), `tests/test_face_debug.py`, `tests/test_face_debug_integration.py` |
| Fingerprint | `tests/test_fingerprint_checkpoint.py`, `test_fingerprint_dataset.py`, `test_fingerprint_metrics.py`, `test_fingerprint_model.py`, `test_fingerprint_preprocessing.py`, `test_fingerprint_sampler.py` |
| Voice | `tests/test_voice_backend_integration.py`, `test_voice_checkpoint.py`, `test_voice_dataset.py`, `test_voice_embedding.py`, `test_voice_enrollment_quality.py`, `test_voice_export.py`, `test_voice_metrics.py`, `test_voice_preprocessing.py`, `test_voice_utils.py` |
| Backend API / multi-template / multi-user flows | `tests/test_backend_api.py`, `test_flexible_auth.py`, `test_template_sets.py`, `test_genuine_auth_regression.py`, `test_authentication_debug_sprint.py` |
| Database / migration | `tests/test_database.py`, `test_migration.py` |
| System health / config | `tests/test_system_health.py`, `test_config.py`, `test_main.py` |
| Security validation | `tests/test_security_validation.py` |
| Privacy metrics (revocation/diversity experiments) | `tests/test_privacy_metrics.py`, `test_protected_template_metrics.py`, `test_template_set_experiments.py` |

### Frontend checks (not part of the 512 pytest count — no JS test runner exists in this repo)

- `tsc -b` — TypeScript compilation, confirmed clean (exit 0) this session after every frontend change.
- `oxlint` — confirmed clean (0 errors; only pre-existing style warnings, none introduced this session) across the full `frontend/src` tree.

**No automated frontend test runner (Jest/Vitest/Playwright) exists in this repository** — confirmed by inspecting `frontend/package.json`'s scripts (`dev`, `build`, `lint`, `preview` only). UI behavior changes were verified via type-checking, code-level tracing, and (where applicable) manual browser testing during development, not automated frontend tests. **CURRENT LIMITATION.**

---

## PART 20 — Current Frontend User Flow

```text
LandingPage (campus / building picker)
        |
        v
CheckpointPage (per-building lobby)
        |  shows: "Registered User" (dropdown if >1 known, else plain text)
        |         current user's real enrollment status (per modality)
        |         "Begin Authentication" (if anything is enrolled) or "Enroll Biometrics"
        |         "Manage biometric enrollment" (if already enrolled)
        |         "+ New Registration" (always shown)
        |
        +----------------------------+
        |                            |
        v                            v
  AuthenticatePage            RegisterPage
  (select modalities,               |
   capture each,                    +--> Face Registration (unified 5-sample flow, Part 5.4)
   submit)                          +--> Fingerprint enrollment (single capture)
        |                           +--> Voice enrollment (2 recordings, consistency-checked)
        v                                |
  ResultPage                             v
   - DecisionSummary                "Continue to {building}" -> back to CheckpointPage
   - FusionDiagnosticsPanel
     (only if DEBUG_SCORES=true)
```

### Face registration UX specifically (confirmed against current source this session)

The five internal samples are presented as **one** "Face Registration" process — a single header, a single subtitle, a camera preview, a 5-dot progress indicator ("X of 5 samples"), and a transient "Sample captured ✓" confirmation per accepted capture — **not** five differently-named directional-pose steps. Verified this session by direct inspection of `frontend/src/components/capture/GuidedFaceCapture.tsx` and a full-frontend grep confirming zero remaining visible "Left"/"Right"/"Up"/"Down"/"tilt chin" strings.

---

## PART 21 — Repository Structure

Generated from `git ls-files` this session (accurate as of commit `5c67be1`); only major directories/files listed, not every file.

```text
backend/                FastAPI application
  api/                  One router module per resource: enroll, authenticate, verify,
                         fusion, revoke, templates, user, buildings, metrics, audit, system
  database/              SQLAlchemy models, CRUD, migration (create/upgrade schema), session
  services/               ModalityService (shared enroll/authenticate logic), per-modality
                         service singletons, face enrollment vocabulary, face debug diagnostics,
                         recording quality (voice), authentication orchestration
  auth/                  Minimal auth-adjacent utilities (not user login - see Part 16)
  config.py              Settings (env-driven): MASTER_SECRET, thresholds, template_bits, etc.
  key_continuity.py      MASTER_SECRET key-continuity state machine (this session)
  secret_fingerprint.py  One-way MASTER_SECRET fingerprint (this session)
  main.py                FastAPI app, lifespan startup (incl. key-continuity enforcement)
  security_validation.py Structural security invariant checks
  states.py, utils.py, threshold_loader.py, buildings.py

frontend/                React + Vite + TypeScript dashboard
  src/api/               HTTP client + wire types (mirrors backend/database/schema.py)
  src/components/
    biometric/            Decision UI, enrollment status, Fusion Diagnostics panel (this session)
    capture/               Face/Fingerprint/Voice capture components
    layout/, ui/            Shared layout and design-system components
  src/pages/               LandingPage, CheckpointPage, RegisterPage, AuthenticatePage,
                          ResultPage, TemplateManagementPage, AnalyticsPage, TestingPage, etc.
  src/hooks/                useAuthSession (session/user identity, this session's roster
                          addition), useEnrollmentProfile, useWavRecorder, etc.
  src/context/              SessionContext, BuildingsContext

preprocessing/            Classical CV preprocessing, one file per modality
  face.py                MTCNN detect+align + quality-gating measurements (this session)
  fingerprint.py          CLAHE + ridge normalization + Gabor filtering
  voice.py                Resample + VAD + mel-spectrogram (two historical bugs fixed here)
  iris.py                 Hough + Daugman normalization

embeddings/               One shared interface over all four modalities
  pipelines.py            ModalityPipeline + per-modality pipeline classes; face quality
                         verdict function (this session)
  centroid.py              Multi-pose centroid averaging (face enrollment)
  constants.py             DEFAULT_CHECKPOINTS per modality

models/                   Model architecture + inference + training code, per modality
  common/                 BaseEmbedder interface, shared ArcFace loss head
  face/, fingerprint/, voice/, iris/
    inference.py           Embedding model wrapper (this is what backend/ actually loads)
    model.py, config.py,   Training-only code (fingerprint/voice have dedicated modules;
    train.py, dataset.py   face/iris are trained only via the Colab notebooks)
    saved/                 Checkpoints (Git-LFS-tracked .pt/.h5) - empty for iris

template_protection/       The cancelable BioHashing transform
  hkdf_keys.py             HKDF-SHA256 key derivation
  biohash.py               generate_template() - the orchestration function
  transform.py             Projection / quantization / permutation stages
  matcher.py                Hamming/cosine comparison
  revoke.py                 Library-level key-rotation function

fusion/                    Multimodal score combination
  score_fusion.py           fuse_scores(), resolve_normalized_weights()
  policy.py                  evaluate_fusion_policy() - ALL_REQUIRED / AT_LEAST_TWO / WEIGHTED
  config.py                   Policy enum, defaults, floor constants
  diagnostics.py               build_fusion_diagnostics() (this session)

evaluation/                 Metrics computation + experiment runners
  results/                   Committed CSV/JSON results (Part 18)
  metrics.py, roc.py, fingerprint_metrics.py, voice_metrics.py, privacy_metrics.py,
  threshold_calibration.py, experiments.py, template_set_experiments.py

tests/                      512 tests total (Part 19)

scripts/                    Standalone operator tools (never run automatically)
  rotate_master_secret.py   Explicit MASTER_SECRET rotation acknowledgment (this session)
  migrate_to_multi_template.py, calibrate_protected_thresholds.py,
  debug_authentication_pipeline.py, run_kaggle_kernels.py

docs/                        13 markdown files + 2 PDFs (Part 17's git-history agent
                            confirmed the complete list); this file is new, added
                            by this documentation task

notebooks/, kaggle_kernels/  Colab / Kaggle training notebooks, one pair per modality
config/                       buildings.json (facility definitions - context only, no policy)
```

---

## PART 22 — How to Run on a New Laptop

*(A complete step-by-step PDF guide, `Biometric-Auth-Setup-Guide.pdf`, was produced earlier this session for exactly this purpose. The steps below summarize it and add the explicit warnings this task requires.)*

1. **Prerequisites:** Git + Git LFS, Python 3.10+, Node.js 18+/npm, a webcam and microphone.
2. **Clone:** `git lfs install` (before cloning, so checkpoints download automatically) then `git clone <repo-url>`.
3. **Python virtual environment:** `python -m venv .venv` then activate it.
4. **Dependency installation:** `pip install -r requirements.txt`.
5. **Frontend dependency installation:** `cd frontend && npm install`.
6. **`.env` creation:** `cp .env.example .env` (repo root).
7. **NEW MASTER_SECRET generation:** `python -c "import secrets; print(secrets.token_hex(32))"` — paste the output into `.env` as `MASTER_SECRET=...`.
8. **DEBUG_SCORES (optional, for trying Fusion Diagnostics):** add `DEBUG_SCORES=true` to `.env`.
9. **Database initialization:** automatic — a fresh, empty database with no `MasterSecretFingerprint` row hits State A (Part 10) and self-initializes on first backend startup; no manual step needed.
10. **Backend startup:** `uvicorn backend.main:app --reload` (repo root).
11. **Frontend startup:** `cd frontend && npm run dev` (separate terminal).
12. **Browser URL:** `http://localhost:5173`.
13. **New registration:** on the checkpoint/lobby screen, use "+ New Registration" (or "Enroll Biometrics" if nothing is enrolled yet).
14. **Authentication:** "Begin Authentication" once at least one modality is enrolled.
15. **Fusion Diagnostics:** with `DEBUG_SCORES=true`, expand the panel on the Result page after authenticating.
16. **Multi-user testing:** use "+ New Registration" again to add a second person on the same machine; the "Registered User" dropdown appears once more than one is known.

### Explicit, required warning

**The other laptop must NOT copy:**
- `.env`
- `MASTER_SECRET`
- the local database file (`biometric.db`)
- any protected biometric templates

**Why:** copying an existing database without the exact `MASTER_SECRET` that protected it triggers this project's key-continuity hard-fail (State B, Part 10) — the server will refuse to start rather than silently authenticate against incompatible templates. The other laptop must generate its **own** fresh `MASTER_SECRET` and perform its **own** fresh enrollment from an empty database. This is by design, not a bug to work around.

---

## PART 23 — What Has Not Been Implemented

| Item | Status |
|---|---|
| Liveness detection | **Not implemented.** Explicitly on hold per project direction; `docs/PRIVACY_AND_SECURITY.md` documents the resulting replay-attack risk as an acknowledged, out-of-current-scope gap. |
| Deepfake detection | **Not implemented.** Same status. **Deepfake/liveness detection is currently ON HOLD and is NOT part of the current implemented authentication decision** — stated here exactly as required. |
| Presentation-attack detection | **Not implemented.** |
| Landmark-based geometric face alignment | **Not implemented** — confirmed by direct inspection of the installed `facenet_pytorch` library source this session (Part 5.2). Landmarks are computed and used only as enrollment quality signals, never to warp the image. |
| Larger-scale biometric evaluation (more subjects, more sessions) | **Not implemented / future work.** Current formal results: 62 face identities (closed-set), 600 fingerprint subjects, 24 voice speakers. |
| Formal threshold calibration for face/voice | **Partially implemented** — thresholds are operator-specified (0.80 each) based on a handful of real genuine-score observations, not a calibration run (`eer`/`far`/`frr`/`auc` all `null` in both threshold JSON files). `scripts/calibrate_protected_thresholds.py` exists and is documented as the intended future path, but is not currently applied. |
| Formal threshold calibration for fingerprint/iris | **Not implemented** — both use the uncalibrated `Settings.match_threshold = 0.9` fallback. |
| ROC/DET curves at the protected-template level | **Not implemented** — the committed ROC curves (`evaluation/results/{fingerprint,voice}_roc.csv`) are computed on raw embeddings, not on the deployed Hamming-similarity/protected-template space. |
| FAR/FRR/EER at the protected-template level | **Not implemented** for the same reason. |
| Cross-device evaluation | **Not verified in the current repository.** |
| Larger user population testing | **Not implemented / future work.** |
| Stronger anti-spoofing evaluation | **Not implemented / future work** (depends on liveness/deepfake detection existing first). |

---

## PART 24 — Research/Evaluation Readiness

For a scientifically valid evaluation of the **deployed** system (protected-template Hamming similarity, not just raw-embedding metrics), the following would eventually need to be collected — none of this exists yet at the deployed-system level:

- **Genuine attempts:** many real captures per enrolled person, across multiple sessions/days/lighting conditions, at the protected-template comparison stage.
- **Impostor attempts:** many real captures from different people compared against each other's stored templates, same-key threat model (i.e., an impostor's live capture compared under the *claimed* identity's key, matching how `authenticate()` actually works).
- **False Acceptance Rate (FAR)** and **False Rejection Rate (FRR)** at the protected-template level, across a range of thresholds.
- **Equal Error Rate (EER)** — the threshold where FAR = FRR — computed on protected-template Hamming similarity, not raw embedding cosine (the current `evaluation/results/*_metrics.csv` files are raw-embedding measurements).
- **ROC curve and AUC**, per modality, at the protected-template level.
- **Per-modality performance** broken out (already exists at the raw-embedding level; would need repeating at the protected-template level).
- **Fusion performance** — genuine/impostor distributions of the *fused* score under `ALL_REQUIRED`, not just per-modality.
- **Latency** — end-to-end request time per modality and for fusion (some latency is already recorded per-request in `audit_logs.latency_ms`, but no aggregate statistical analysis of it exists in the repository).
- **Resource usage** — memory/CPU under load; partially informative given this session's own observations of memory pressure during heavy test runs, but not a formal benchmark.

**The current handful of manual test observations (Part 18) are development diagnostics used to make real engineering decisions during this session — they are not a statistically valid final benchmark**, and this document does not present them as one.

---

## PART 25 — Current Status

| Component | Status | Evidence |
|---|---|---|
| Face | Implemented, tested | Real checkpoint (112 MB), 512-d embeddings, quality-gated enrollment, 18 dedicated quality-gating tests |
| Fingerprint | Implemented, tested; requires further evaluation | Real checkpoint (105 MB), but measured EER ≈31% (weakest modality, self-documented) |
| Voice | Implemented, tested | Real checkpoint (25 MB), measured EER ≈2.3%, two documented preprocessing bugs found and fixed |
| Iris | Not implemented (mock mode only) | No checkpoint in `models/iris/saved/`; not offered in the frontend |
| Fusion | Implemented, tested | `ALL_REQUIRED` default policy, equal-weight averaging, dedicated regression test proving fused-score-alone is not sufficient |
| Template protection | Implemented, tested | HKDF-SHA256 + orthonormal projection + keyed quantization + permutation, 256-bit output in the current deployment |
| Key continuity | Implemented, tested | 5-state machine, real startup hard-fail demonstrated this session against the actual local database |
| Enrollment | Implemented, tested | Guided 5-sample face flow, single-capture fingerprint, two-recording voice consistency check |
| Authentication | Implemented, tested | Shared `ModalityService` path across all four modalities |
| User management | Partially implemented | Local per-browser roster only; no backend user-listing API exists |
| Fusion diagnostics | Implemented, tested | `DEBUG_SCORES`-gated backend field + frontend panel, both added this session |
| Testing | Implemented | 512 automated backend tests passing; no automated frontend test runner exists |
| Deployment | Partially implemented | `backend/render.yaml`, `frontend/vercel.json` exist; per `docs/DEPLOYMENT_AUDIT_REPORT.md`, no live Render service had actually been created as of that report — **not independently re-verified in this documentation pass** |
| Liveness | Not implemented | On hold, explicitly out of scope for the current authentication decision |
| Deepfake detection | Not implemented | On hold, explicitly out of scope for the current authentication decision |

---

## Document Provenance

This document was produced by directly inspecting the repository at commit `5c67be1` (branch `main`): reading source files, tests, configuration, and existing project documentation, and by re-deriving several facts empirically during this session (e.g. the cosine-similarity-vs-key-mismatch measurement in Part 17, the `facenet_pytorch` alignment-behavior confirmation in Part 5.2). Four parallel research passes independently verified the datasets/training details, the fingerprint pipeline, the voice pipeline, and the database schema/git history/test count, each with file:line citations; those citations are preserved above.

**Items this document could not fully verify and says so explicitly, for the record:**
- Face's exact training hyperparameters (Adam, lr 1e-4, batch 32, 10 epochs) are sourced from `docs/PROJECT_FUNDAMENTALS.md`'s narrative only — not independently re-confirmed against the training notebook's actual code cells in this pass.
- Iris's training hyperparameters are not available at all — no checkpoint or run artifact exists.
- The exact split methodology for face and iris (unlike fingerprint/voice, which have dedicated, directly-cited split functions) is described only in notebook markdown headers, not in a committed, directly-citable split function.
- The numeric relationship between `voice_threshold.json`'s operator-set 0.8 (protected-template domain) and `voice_metrics.csv`'s computed `eer_threshold` ≈0.322 (raw-embedding cosine domain) is not reconciled anywhere in the codebase — both are reported, neither is used to explain the other.
- The exact sample size behind the specific manual face/voice score observations discussed during this session (0.6992/0.7422/0.7734 etc.) could not be traced to a single permanent committed artifact — reported as development-diagnostic observations from this conversation, not a benchmark dataset.
- Live deployment status (whether a Render/Vercel instance is actually running) was not re-verified in this documentation pass; the most recent committed evidence (`docs/DEPLOYMENT_AUDIT_REPORT.md`) stated no live Render service existed as of that report's date.
