# Roadmap

Three phases. In every phase, all modalities advance to the same stage
together — there is no phase where one modality is further along than the
others. The supported modalities are face and voice; iris and fingerprint were
part of the original Phase 1 plan and have been removed (see the last section).

## Phase 1 — Foundation (current)

**Status: implemented.**

- Modular repo scaffold (`preprocessing/`, `models/`, `embeddings/`, `evaluation/`).
- Face preprocessing: detection + crop (MTCNN).
- A single embedding interface (`BaseEmbedder.extract_embedding`) every
  modality implements identically.
- Colab notebooks (`notebooks/01` face, `notebooks/04` voice) *and* equivalent Kaggle Kernels
  (`kaggle_kernels/`, driven unattended via `scripts/run_kaggle_kernels.py`)
  that download each modality's dataset, fine-tune its embedding model with
  an ArcFace head, **save the resulting checkpoint** (`.pt` + a companion
  `.h5` export), evaluate it (Experiment 1:
  accuracy/FAR/FRR/EER/ROC-AUC), and run a dedicated image-based testing
  section (genuine/impostor pairs + gallery matching, visualized).
- `tests/` — offline, synthetic-image smoke tests proving the
  preprocessing → embedding pipeline is correctly wired for every modality,
  independent of whether real checkpoints exist yet (`pytest tests/`).

### Voice (added alongside Phase 1's foundation)

**Status: implemented, checkpoint trained.** Real Kaggle GPU-kernel run (CPU
fallback — see `docs/VOICE_MODEL.md`): 24 speakers, 4,857 utterances, 2.29%
EER / 97.71% accuracy / 0.9967 AUC on the held-out test set.

Speaker verification via ECAPA-TDNN, added to the same standard as Face:
`preprocessing/voice.py`, `models/voice/`, a Colab notebook
(`notebooks/04_voice_training.ipynb`) and Kaggle Kernel
(`kaggle_kernels/voice_training/`), offline synthetic-audio tests, and full
documentation (`docs/VOICE_MODEL.md`). See that doc for the one interface
nuance this modality required (a mel-spectrogram stands in for the "image"
`BaseEmbedder.extract_embedding()` expects) and `docs/DATASETS.md` for the
VoxCeleb1-subset dataset note.

## Phase 2 — Privacy layer

**Status: implemented.**

- `template_protection/hkdf_keys.py`: HKDF-SHA256 key derivation — per
  (user, modality, application, key_version), three independent seeds never
  stored in plaintext beside the protected template they produce.
- `template_protection/transform.py` + `biohash.py`: the BioHashing-style
  transform (keyed orthonormal random projection → keyed quantization →
  keyed permutation) applied identically to every modality's embeddings —
  see `docs/TEMPLATE_PROTECTION.md` for the full mathematical walkthrough.
- `template_protection/matcher.py` (Hamming/cosine comparison,
  accept/reject) and `revoke.py` (key rotation → a new, unlinkable
  template).
- FastAPI backend (`backend/`) + SQLite: `POST /enroll`, `POST /authenticate`,
  `POST /verify/{face,voice}`, `POST /revoke-template`,
  `GET /user/{id}`, `DELETE /user/{id}` — storing **only** protected
  templates, never raw images or raw embeddings (see `docs/BACKEND_API.md`).
- `evaluation/privacy_metrics.py`: Experiment 1 (protected vs. unprotected
  similarity preservation), Experiment 2 (revocability), Experiment 3
  (diversity), Experiment 4 (FAR/FRR/EER on protected templates).
- `docs/PRIVACY_AND_SECURITY.md`'s Experiment 5 analysis (template leakage,
  replay, key compromise, reconstruction risk, database compromise,
  cross-application linkability) filled in — proportionate, non-overclaiming.

## Phase 3 — Fusion, dashboard, full evaluation

- `fusion/score_fusion.py`: configurable weighted-score fusion across
  whichever modalities are available for a given authentication attempt.
- React (Vite) dashboard: enrollment (face + voice captures + "only protected templates
  stored" confirmation), authentication (per-modality pass/fail + fusion
  score + threshold + verdict), and a security/template view that shows
  "Protected" status without ever rendering raw biometric data.
- Experiment 3: full comparison across all modality combinations (face-only,
  voice-only, face + voice).
- Final README/docs pass, end-to-end demo script, limitations and ethics
  section finalized.

## Phase 3A / 3A.1 - Template-set architecture (done)

Requested during project review; this is the final architecture.
See `docs/MULTI_TEMPLATE_ARCHITECTURE.md`.

- A user's credential is a pool of **template sets** (`TEMPLATE_POOL_SIZE`, default 4). Each set holds
  one cancelable template per enrolled modality (Face V*n* + Voice V*n*); Set 1 is
  ACTIVE, the rest STANDBY, and revocation moves a whole set (all modalities together).
- Lifecycle (`ACTIVE` / `STANDBY` / `REVOKED`) in the existing `protected_templates` table
  (`template_set_version`, `template_set_status`, set created/activated/revoked times).
- Authentication matches the ACTIVE set only, never mixes sets, and returns a single fused
  similarity; per-modality scores are internal (audit / debug only). The denied result screen shows no
  similarity.
- Template-set management (`GET /templates`, revoke, activate, generate) is protected by **biometric
  authorization** against the ACTIVE set (403 on failure) instead of a login layer; 409 when the set
  pool is exhausted / full.
- Runtime set validation, template-set evaluation experiments (diversity / revocation / promotion /
  exhaustion), migration of previously enrolled users, Template Management and Template Protection pages.
- Still open: a real login/authorization layer and calibrated per-modality thresholds.

## Phase 3B / V3 - Flexible, user-driven multimodal authentication (done)

- **Buildings are context only** (`id`, `name`, `clearance_level`, `description` in `config/buildings.json`); no building defines
  required modalities and the loader rejects a config that does. The former building-readiness endpoint is gone.
- **The user chooses**: face, voice or both can be enrolled (status `NOT_REGISTERED` / `REGISTERED` /
  `UPDATED` / `RETRY_REQUIRED`) and any subset of the enrolled modalities presented. `/authenticate/fusion` (and `/authenticate`,
  `/verify/*`) authenticate and fuse exactly the submitted modalities; a submitted modality that is not enrolled is
  `ENROLLMENT_REQUIRED` (HTTP 409, never a denial). The public response adds `authentication_state`, `fusion_distance`,
  `active_template_set`.
- **Face one-time five-pose enrollment** (Front / Left / Right / Slight Up / Slight Down): per pose MTCNN + alignment + a 512-d embedding, only blurry or
  faceless poses rejected, the valid embeddings averaged into a **centroid** and discarded, T1-T4 generated from the centroid alone. Authentication stays a
  single capture.
- **Voice** recorded twice; the ECAPA embedding cosine grades them Excellent / Good (enrolled), Fair (`LOW_QUALITY_WARNING`, the user
  continues or re-records) or Poor (`ENROLLMENT_INCONSISTENT`, nothing stored).
- Template pool unchanged (T1 ACTIVE, T2-T4 STANDBY per modality; revocation promotes the next set), now with a Face / Voice matrix.
- Still open: a real login layer and calibrated per-modality thresholds.

## Modality change - iris and fingerprint removed (2026-09-27)

- Iris and fingerprint were removed completely: models, preprocessing, services, `/verify/iris` and
  `/verify/fingerprint`, their training notebooks/kernels, tests, UI and evaluation rows. The supported modalities are
  face and voice.
- The `AT_LEAST_TWO` fusion policy was removed with them (it required exactly three submitted modalities); the policies
  are `ALL_REQUIRED` (default) and `WEIGHTED`.
- `backend/database/migration.py::upgrade_schema` (run at every startup) deletes stored iris/fingerprint templates and
  enrollment events and drops the `audit_logs.fingerprint_similarity` column. Face and voice data are untouched; old
  audit rows are kept as history.

## Dynamic hand gesture modality (implemented, not yet evaluated)

- Third, optional modality: a handwritten-style Z gesture (the infinity gesture was removed 2026-09-28), tracked in the
  browser with MediaPipe Hands (landmarks only, never video); position/size-normalized palm trajectory + timing +
  orientation features, DTW with a +/-12 deg rotation search against three enrolled samples (median). Threshold from a
  SYNTHETIC calibration only - real recordings still needed.
  `POST /enroll/hand`, `POST /enroll/hand/check`, `POST /verify/hand`, `hand_gesture` in fusion / template-set calls.
- Templates (`hand_gesture_v1`) live in the existing template sets, protected by a keyed orthonormal transform
  (revocable, but invertible with the key). Startup schema upgrade adds nullable columns only.
- `AT_LEAST_TWO` fusion policy reintroduced: at least two submitted modalities, at least two must pass.
- Still open: collecting consented data and calibrating `HAND_GESTURE_DTW_THRESHOLD` (currently a development
  default); non-invertible sequence protection; liveness. See `docs/HAND_GESTURE_MODALITY.md`.

## Protocol update (2026-09-27)

- Registration is numbered by biometric: Step 1 Face, Step 2 Voice, Step 3 Dynamic Hand Gesture (optional); naming the
  user is account setup, not a step.
- Voice AUTHENTICATION is one sentence (`voice_audio_2` is refused); voice ENROLLMENT still records two. Expected cost
  on the dataset measurements: FRR 2.6% -> 12.2% at the unchanged threshold (docs/VOICE_MODEL.md) - to be re-evaluated.
- Hand ENROLLMENT is three independent Z samples (duplicate sequences and a sample unlike both others refused);
  authentication is one sample.
