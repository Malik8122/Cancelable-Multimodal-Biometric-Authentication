> **Modality change (2026-09-27).** Iris and fingerprint have been removed from the system: the implemented
> modalities are **face and voice**, and the `AT_LEAST_TWO` fusion policy (which needed three modalities) is gone.
> This document is a dated record. Its iris / fingerprint material (models, datasets, results, thresholds,
> three-modality fusion) is kept for history only and no longer describes the code. Current state: `README.md`,
> `docs/ROADMAP.md`; current numbers: `evaluation/PAPER_RESULTS_INDEX.md`.

> **Update - superseded parts.** This document is a snapshot of commit `5c67be1`. Since then, face and voice are decided in their own metrics: face on a calibrated **estimate** of cosine similarity (>= 0.80), voice on a calibrated **estimate** of Euclidean distance (<= 0.75, lower is better), both derived from the unchanged 256-bit BioHash Hamming comparison. `evaluation/results/face_threshold.json` and `voice_threshold.json` (0.80 on Hamming similarity), described below, were removed, and registration now asks for a display name. See [BIOMETRIC_METRICS.md](BIOMETRIC_METRICS.md) for the current behaviour; everything else here is unchanged.

<div style="page-break-after: always;">
<div style="text-align:center; margin-top:2.2in;">
<div style="font-size:11pt; letter-spacing:2pt; color:#666666;">CAPSTONE PROJECT — TECHNICAL REFERENCE</div>
<div style="font-size:25pt; font-weight:bold; color:#12233f; margin-top:14pt; line-height:1.3;">
Cancelable Biometric Templates &amp; BioHashing
</div>
<div style="font-size:15pt; color:#333333; margin-top:6pt;">Technical Explanation and Viva Preparation</div>
<div style="font-size:10.5pt; color:#555555; margin-top:26pt;">
Cancelable Multimodal Biometric Authentication for Critical Infrastructure
</div>
<div style="font-size:9.5pt; color:#777777; margin-top:4pt;">
Every claim in this document is traced to the actual repository at commit <code>5c67be1</code> (branch <code>main</code>).<br/>
Nothing here is a generic textbook description — it is a reconstruction of exactly what this project's code does.
</div>
</div>
</div>

## Table of Contents

1. Project-Specific Overview
2. Exact Code Implementation
3. What Exactly Is Being Hashed?
4. BioHashing in This Project
5. MASTER_SECRET
6. Secret Fingerprint / Key Continuity
7. What Makes the Template "Cancelable"?
8. What Happens If the Template Is Stolen?
9. What Happens If MASTER_SECRET Changes?
10. Hamming Similarity
11. Thresholds
12. Multimodal Fusion
13. Why Not Store Raw Biometric Features?
14. Important Limitations — "What I Should NOT Claim in a Viva"
15. Viva Questions and Model Answers (30 questions)
16. One-Page Cheat Sheet
17. Visual Diagrams
18. Exact Numbers and Parameters
19. Source Code References
20. Final "Teacher Traps"

<div style="page-break-after: always;"></div>

## PART 1 — Project-Specific Overview

### 1.1 The problem being solved

A password that leaks can be changed. A face, fingerprint, or voice **cannot** — a person has exactly one face. If a biometric authentication system stores the raw biometric sample (or an unprotected numeric representation of it) as the actual credential, a single database breach permanently compromises that person's biometric identity across every system that ever trusted it. This project's stated design response is: **never store the raw biometric or an unprotected embedding as the credential** — store only a value produced by passing the embedding through a **keyed, one-way, revocable transform**. If that stored value is ever compromised, the system can issue a *new* one from a *new* key without requiring a new biometric sample.

### 1.2 Vocabulary — precise, as used in this project's own code

| Term | Meaning in this project | Where it lives |
|---|---|---|
| **Biometric sample** | The raw capture — a camera frame, a fingerprint scan, a WAV recording. **Never written to disk anywhere in this codebase.** | Exists only for the duration of one HTTP request, in memory |
| **Biometric feature / embedding** | A fixed-length, L2-normalized real-valued vector produced by a trained model from the sample | `models/{face,fingerprint,voice,iris}/inference.py`, via `BaseEmbedder.extract_embedding()` |
| **Biometric template (unprotected)** | In the general biometrics literature, this usually means the stored representation used for matching. **This project deliberately never stores this form** — the embedding above is the closest analog, and it is never persisted. | N/A — not persisted |
| **Protected / cancelable template** | The output of this project's BioHash transform: a fixed-length bit array, packed to bytes | `protected_templates.protected_template` column, `backend/database/models.py::ProtectedTemplate` |
| **Secret / key** | `MASTER_SECRET` — a server-side environment value every derived key traces back to | `backend/config.py::Settings.master_secret` |
| **Authentication score** | The Hamming similarity between a freshly generated protected template and the stored one, a float in `[0, 1]` | `template_protection/matcher.py::compare()` |

### 1.3 The exact pipeline in this project

**Enrollment:**

```text
Biometric input (camera frame / audio clip)
        |
        v
Preprocessing (modality-specific — see Part 3)
        |
        v
Feature extraction / embedding (a trained model, per modality)
        |
        v
L2 normalization (BaseEmbedder._l2_normalize — every real embedding)
        |
        v
[Face only] Multi-sample centroid: average 5 accepted embeddings, re-normalize
        |
        v
Template protection / BioHash (template_protection/biohash.py::generate_template)
        |
        v
Protected binary template (packed bits)
        |
        v
Storage (protected_templates table — this is the ONLY biometric-derived data stored)
```

**Authentication:**

```text
Live biometric input
        |
        v
Same preprocessing (identical code path to enrollment)
        |
        v
Same feature extraction (the same trained model, same checkpoint)
        |
        v
Same L2 normalization
        |
        v
Same template-protection transformation, regenerated fresh
   (uses the SAME key as the stored template — derived from the
   SAME MASTER_SECRET and the SAME user/modality/application/key_version)
        |
        v
New protected binary template (candidate)
        |
        v
Hamming similarity vs. the STORED protected template
        |
        v
Per-modality threshold check -> per-modality pass/fail
        |
        v
Fusion policy (ALL_REQUIRED) -> final ACCESS_GRANTED / ACCESS_DENIED
```

Enrollment and authentication are **not the same code path end-to-end** — enrollment additionally builds a template and writes it to the database; authentication additionally reads a stored template and compares. But the preprocessing → embedding → normalization → protection stages are **the same functions**, called from both places (`backend/services/base_service.py::ModalityService.enroll()` and `.authenticate()`).

<div class="important-box"><strong>IMPORTANT —</strong> No raw image, no raw audio, and no raw (unprotected) embedding is ever written to the database anywhere in this codebase. The embedding exists only as a local Python variable for the duration of one request and is explicitly deleted (`del embedding`) after being protected.</div>

<div style="page-break-after: always;"></div>

## PART 2 — Exact Code Implementation

This table lists the functions a student should be able to name and explain if asked "walk me through the code."

| File | Function / Class | Input | Output | Enrollment, Auth, or Both |
|---|---|---|---|---|
| `models/common/base_embedder.py` | `BaseEmbedder.extract_embedding()` | preprocessed image array | L2-normalized embedding vector | Both |
| `embeddings/centroid.py` | `centroid_embedding(embeddings)` | list of up to 5 L2-normalized face embeddings | one L2-normalized centroid vector | Enrollment only (face) |
| `template_protection/hkdf_keys.py` | `derive_key(master_secret, *, application_id, user_id, modality, key_version)` | `MASTER_SECRET` + 4 identifying fields | `KeyMaterial` (3 seeds) | Both |
| `template_protection/transform.py` | `build_orthonormal_projection(seed, output_bits, embedding_dim)` | `projection_seed` | `(output_bits, embedding_dim)` projection matrix | Both |
| `template_protection/transform.py` | `project(embedding, projection_matrix)` | embedding + projection matrix | real-valued vector in `[-1, 1]` | Both |
| `template_protection/transform.py` | `quantize(projected, seed)` | projected vector + `threshold_seed` | bit vector (`uint8` 0/1 array) | Both |
| `template_protection/transform.py` | `apply_permutation(bits, seed)` | bit vector + `permutation_seed` | reordered bit vector | Both |
| `template_protection/biohash.py` | `generate_template(embedding, key, output_bits)` | embedding + `KeyMaterial` | the final protected template (packed later) | Both |
| `template_protection/utils.py` | `pack_bits()` / `unpack_bits()` | bit array / bytes | bytes / bit array | Both (storage format conversion) |
| `template_protection/matcher.py` | `compare(template_a, template_b, metric="hamming")` | two equal-length protected templates | float similarity in `[0, 1]` | Authentication only |
| `template_protection/matcher.py` | `accept(score, threshold, metric)` | score + threshold | boolean | Authentication only |
| `backend/services/base_service.py` | `ModalityService.enroll()` / `.enroll_poses()` | raw sample(s) | stored `ProtectedTemplate` rows | Enrollment only |
| `backend/services/base_service.py` | `ModalityService.authenticate_embedding()` | live embedding + `user_id` | `AuthenticationResult(score, threshold, authenticated)` | Authentication only |
| `backend/secret_fingerprint.py` | `fingerprint_master_secret(master_secret)` | `MASTER_SECRET` | one-way 32-byte fingerprint | Server startup only |
| `backend/key_continuity.py` | `evaluate_key_continuity()` / `enforce_key_continuity_at_startup()` | current secret's fingerprint vs. stored one | state / hard-fail | Server startup only |

<div style="page-break-after: always;"></div>

## PART 3 — What Exactly Is Being Hashed?

This is the single most important thing to get right in a viva: **the transform never touches a raw image, audio clip, or pixel array. It only ever operates on a fixed-length, L2-normalized numeric vector — the embedding (or, for face, the centroid of five embeddings).**

### FACE — traced exactly

1. MTCNN (`facenet-pytorch`) detects a face and returns a bounding box + 5 landmarks (`preprocessing/face.py::FacePreprocessor.detect_and_align()`).
2. The bounding box is cropped and resized to **160×160×3**, uint8 (landmarks are used only for enrollment quality checks — face size/centering/roll/yaw — never to geometrically warp the crop; there is **no** landmark-based rotation alignment in this pipeline).
3. `InceptionResnetV1` (`models/face/inference.py::FaceEmbedder`) produces a **512-dimensional** embedding.
4. `BaseEmbedder._l2_normalize()` scales it to unit length.
5. **Enrollment only:** up to 5 such embeddings (one per accepted capture) are averaged and re-normalized — `embeddings/centroid.py::centroid_embedding()`: `c = normalize(mean(e_1, ..., e_k))`, `k` between 3 and 5.
6. **The vector that reaches BioHash is: the 512-d L2-normalized centroid (enrollment) or the 512-d L2-normalized single-capture embedding (authentication).**

### FINGERPRINT — traced exactly

1. Classical CV preprocessing only (`preprocessing/fingerprint.py`): grayscale → CLAHE contrast enhancement → ridge normalization → Gaussian blur → 8-orientation Gabor filter bank → min-max normalize → resize to 224×224 → ImageNet mean/std normalize.
2. `ResNet50` + a projection head (`models/fingerprint/inference.py::FingerprintEmbedder`) produces a **512-dimensional** embedding.
3. `BaseEmbedder._l2_normalize()`.
4. **No multi-sample aggregation** — fingerprint enrollment is single-capture, unlike face.
5. **The vector that reaches BioHash is: the 512-d L2-normalized single-capture embedding.**

### VOICE — traced exactly

1. Classical DSP preprocessing (`preprocessing/voice.py`): resample to 16 kHz → mono → VAD/silence trim → loudness normalize → 4.0-second fixed-length segment → 80-bin log-mel filterbank, mean-normalized.
2. ECAPA-TDNN (`models/voice/inference.py::VoiceEmbedder`) produces a **192-dimensional** embedding.
3. `BaseEmbedder._l2_normalize()`.
4. **Enrollment uses two recordings**, but only for a pre-storage *quality/consistency check* (cosine similarity between the two embeddings, `backend/services/recording_quality.py`) — **the stored template is built from the first recording's embedding alone**, not an average of the two.
5. **The vector that reaches BioHash is: the 192-d L2-normalized embedding of the first (or, at authentication, the single) recording.**

### Summary table

| Modality | Raw input | Feature extractor | Feature dimension | Normalization | Aggregation | Input to protection |
|---|---|---|---|---|---|---|
| Face | 160×160×3 aligned crop | InceptionResnetV1 | 512 | L2 (unit norm) | Centroid of 3-5 embeddings (enrollment only) | 512-d L2-normalized vector (centroid or single) |
| Fingerprint | 224×224×3 enhanced image | ResNet50 + projection head | 512 | L2 (unit norm) | None | 512-d L2-normalized vector |
| Voice | 80-bin log-mel, 4.0 s | ECAPA-TDNN | 192 | L2 (unit norm) | None (2nd recording used only for a pre-storage quality check, not averaged in) | 192-d L2-normalized vector (first recording) |
| Iris | Hough+Daugman normalized strip | ResNet18 + projection head | 256 | L2 (unit norm) | None | **Not applicable in practice — no trained checkpoint exists; falls back to a deterministic mock embedding, never a real biometric value** |

<div class="viva-tip"><strong>VIVA TIP —</strong> If asked "do all three modalities feed the same thing into BioHash?", the correct answer is: they feed different-dimensional vectors (512 / 512 / 192) through the identical transform code, because the transform's projection stage is built dynamically from whatever embedding_dim it receives — the BioHash function itself is modality-agnostic.</div>

<div style="page-break-after: always;"></div>

## PART 4 — Explain BioHashing in This Project

### What the project actually implements

This project's BioHash implementation (`template_protection/biohash.py::generate_template`, orchestrating `template_protection/transform.py`) is a **secret-key-derived random orthonormal projection, followed by a keyed thresholded (sign-split) quantization, followed by a keyed permutation.** It is *not* plain SHA-256 hashing, and it is *not* a standard textbook "BioHashing" that XORs a random binary string with a thresholded biometric code — this project's own version uses a random **orthonormal** projection matrix (not an arbitrary random matrix) and a **jittered, data-scaled** quantization threshold rather than a fixed cutoff.

### The exact mathematical formulation, matching the code

**Stage 1 — Projection.** Let `x` be the L2-normalized embedding (dimension `D`: 512 for face/fingerprint, 192 for voice). Let `R` be a `(B, D)` matrix (`B = output_bits`, 256 in this deployment) whose rows are unit vectors, built by QR-decomposing a Gaussian random matrix seeded from `projection_seed` (`template_protection/transform.py::build_orthonormal_projection`):

```text
y = R x            (y is a vector of length B, each y_i in [-1, 1] by Cauchy-Schwarz,
                     since both R's rows and x are unit vectors)
```

**Stage 2 — Quantization.** For each position `i`, a per-position random threshold `t_i` is drawn from `Normal(0, 0.5 * std(y))`, seeded by `threshold_seed` (`template_protection/transform.py::quantize`):

```text
b_i = 1  if y_i > t_i
b_i = 0  otherwise
```

This is **not** a fixed-magnitude threshold (an earlier version of the code used `Uniform(-1, 1)`, which the code's own comments document as a real, since-fixed bug — a fixed-scale threshold overwhelmed the actual signal for high-dimensional embeddings, since `y`'s components are typically as small as ~1/√D). The threshold's spread is scaled to `y`'s own standard deviation, so quantization is still primarily a **sign-split of the embedding-driven signal**, with the key only shifting exactly which positions land on which side of that split.

**Stage 3 — Permutation.** The resulting bit vector is reordered by a key-derived random permutation, seeded by `permutation_seed` (`template_protection/transform.py::apply_permutation`). A permutation alone adds no entropy — but combined with the keyed projection and quantization, it means two templates produced under **different keys** share no positional structure at all.

### What the binary template contains, precisely

- **Number of bits:** `output_bits` — the library's generic default is `DEFAULT_OUTPUT_BITS = 128` (`template_protection/biohash.py:62`), but **this project's actual deployed setting is 256 bits** (`backend/config.py::Settings.template_bits = 256`), explicitly passed by `backend/services/base_service.py::build_entry()` for every real enrollment.
- **Relationship between embedding dimension and template length:** true orthonormality of every row of `R` only holds when `output_bits <= embedding_dim`. For face/fingerprint (512-d) at 256 output bits, `R` is fully orthonormal. For voice (192-d) at 256 output bits, `output_bits > embedding_dim`, so `build_orthonormal_projection` composes **two orthonormal blocks** instead of one fully orthonormal matrix — rows are orthonormal *within* a block but not across blocks. This is an explicit, documented tradeoff in the code, not a hidden one.
- **How the projection matrix is generated:** deterministically from `projection_seed` via `numpy.random.default_rng` (PCG64) + QR decomposition of a Gaussian matrix — **the matrix itself is never stored**; it is recomputed from the seed every time it's needed.
- **Deterministic for the same secret?** Yes — same `MASTER_SECRET` + same `(application_id, user_id, modality, key_version)` → same seeds → same projection matrix, every time, on any machine.
- **Changes when the secret changes?** Yes — a different `MASTER_SECRET` produces entirely different seeds, hence an entirely unrelated projection matrix (see Part 9).
- **Stored?** No — only the final protected template (post-projection, post-quantization, post-permutation) is stored. The matrix, the thresholds, and the permutation are all regenerated from the key on demand and never persisted.
- **Regenerated during authentication?** Yes, in full, from scratch, using the stored template's own recorded `key_version` (`backend/services/base_service.py::authenticate_embedding()`).

<div class="important-box"><strong>IMPORTANT —</strong> This project's own documentation (<code>docs/TEMPLATE_PROTECTION.md</code>) is explicit that this is an information-lossy data-transformation argument, not a computational-hardness proof. Do not describe this as "cryptographically one-way" in a viva without that qualification.</div>

<div style="page-break-after: always;"></div>

## PART 5 — MASTER_SECRET

### Where it comes from and how it is generated/loaded

`MASTER_SECRET` is a plain string, generated by the operator (`python -c "import secrets; print(secrets.token_hex(32))"` — a 64-hex-character, 256-bit random value) and placed in a local `.env` file. It is loaded by `backend/config.py::Settings.master_secret: str` (Pydantic settings, **no default value** — the server refuses to start at all if it is unset).

### Is it stored in the database?

**No.** Confirmed directly from the schema: no table or column in `backend/database/models.py` holds `MASTER_SECRET`. It exists only in the process environment / `.env` file (which is gitignored).

### Is it stored in the protected biometric template?

**No.** The protected template is the *output* of a transform that is *keyed by* a value derived from `MASTER_SECRET` — the secret itself never becomes part of the template's bytes.

### Is it used directly, or transformed first?

**Transformed.** It is never used directly as a projection matrix or as raw entropy for quantization. It is the **input key material (IKM)** to HKDF-SHA256 (`template_protection/hkdf_keys.py::derive_key`), which produces three independent 32-byte seeds (`projection_seed`, `permutation_seed`, `threshold_seed`).

### Is HKDF/HMAC/SHA-256 used?

**Yes — HKDF-SHA256**, via the `cryptography` library's `HKDF` class. The HKDF "salt" is `SHA256(f"{application_id}|{user_id}|{modality}|{key_version}")` (a fixed-length digest, avoiding any delimiter-collision risk from raw variable-length strings); the HKDF "info" field is one of three fixed byte strings (`b"template_protection/projection"`, `b"template_protection/permutation"`, `b"template_protection/threshold"`) that separates the three seeds drawn from the same salt.

### How it influences BioHash generation

Every stage of the transform (Part 4) is seeded from one of these three HKDF outputs. Change `MASTER_SECRET` and all three seeds change, hence the projection matrix, the quantization thresholds, and the permutation all change.

### What happens if the secret changes

See Part 9 in full — in short: existing protected templates, generated under the old secret, become statistically unrelated to anything regenerated under the new secret.

### The three-way distinction, made explicit

| Concept | What it is | Where it lives | Can it reconstruct a biometric? |
|---|---|---|---|
| **MASTER_SECRET** | The one root secret, a plain string | Environment / `.env` only, never the database | N/A — it's the key, not a derived output |
| **Secret fingerprint** | A one-way HKDF-SHA256 output of `MASTER_SECRET` over a **fixed public context** (`b"master_secret_fingerprint/v1"`) | `master_secret_fingerprint` table, one row | No — cannot be reversed to `MASTER_SECRET`, and shares no code path with the biometric key derivation |
| **Protected biometric template** | The BioHash output for one person's one modality, keyed by `MASTER_SECRET`-derived seeds | `protected_templates` table, one row per (user, modality, template set) | No — same one-way argument as Part 4, and additionally requires the biometric itself to reproduce |

<div class="viva-tip"><strong>VIVA TIP —</strong> If a teacher asks "so is the fingerprint just another biometric template?" — the answer is no: the secret fingerprint is derived from <code>MASTER_SECRET</code> alone, using a completely separate HKDF call with a different, fixed, non-per-user context string. It never touches a biometric embedding at all.</div>

<div style="page-break-after: always;"></div>

## PART 6 — Secret Fingerprint / Key Continuity

### Why this exists

During this project's own development, the local database contained protected templates generated under one `MASTER_SECRET`, and the backend was later restarted with a **different**, freshly-generated `MASTER_SECRET` (because no `.env` file had persisted across sessions). Nothing previously checked whether a newly-configured secret was the *same* one that had protected existing templates — any syntactically valid string was silently accepted, and authentication then failed for every enrolled user with no clear explanation. This mechanism (`backend/secret_fingerprint.py`, `backend/key_continuity.py`, `backend/database/models.py::MasterSecretFingerprint`) was built specifically to make that failure mode loud and immediate instead of silent and confusing.

### The actual process

```text
MASTER_SECRET
      |
      v
fingerprint_master_secret()  — HKDF-SHA256(master_secret, salt=None, info=b"master_secret_fingerprint/v1")
      |
      v
32-byte one-way fingerprint (NEVER the secret itself)
      |
      v
Stored: master_secret_fingerprint table, exactly one row (id fixed at 1)
      |
      v
At every server startup: evaluate_key_continuity() re-computes the fingerprint of
the CURRENTLY configured secret and compares it (hmac.compare_digest) with the stored one
      |
      +--> matches / no templates exist yet  -> proceed (or auto-initialize the first time)
      +--> does NOT match, or templates exist with no fingerprint recorded
             -> KeyContinuityError, server refuses to start
```

### Why HKDF, and why a fixed public context string

HKDF (HMAC-based Key Derivation Function, RFC 5869) is a standard construction for deriving one or more cryptographically strong outputs from an input key. It is used here for the *same reason* it is used for the template keys in Part 5 — it is already the trusted primitive this codebase uses, so reusing it (with a different, fixed, public "info" context) keeps the fingerprint mechanism's cryptographic assumptions consistent with the rest of the system rather than introducing a second, independently-reasoned-about primitive.

The context string (`b"master_secret_fingerprint/v1"`) is **fixed and public** — it is not a secret and does not need to be. It exists only to make this specific HKDF call's output namespace-distinct from the per-template HKDF calls in `hkdf_keys.py::derive_key` (which use a per-user/modality/application salt and different info strings). Because the context is fixed, the *only* thing that determines the fingerprint's value is `MASTER_SECRET` itself.

### Why this does NOT expose MASTER_SECRET

HKDF is a one-way expansion function — there is no published, practical algorithm to invert `HKDF-SHA256(secret, ...)` back to `secret` given only the output. The fingerprint answers exactly one question — "is this the same secret as before?" — and nothing else. `backend/key_continuity.py` only ever compares two fingerprints with `hmac.compare_digest`; no code path anywhere converts a fingerprint back into a secret, because none exists.

### What is stored in the database

Only the 32-byte fingerprint (`MasterSecretFingerprint.fingerprint`, `LargeBinary`) and a timestamp. Never `MASTER_SECRET`. Never a template. Never a biometric value.

### The actual incident this project discovered (explained precisely)

1. A face template for a real enrolled user was created on 2026-09-17, protected under whatever `MASTER_SECRET` the backend process had at that time.
2. Later, in a fresh session, no `.env` file existed on disk. A new `MASTER_SECRET` was generated and written to a new `.env`.
3. The backend was restarted with this new secret and pointed at the *same* existing database (same templates, unchanged).
4. The same real person authenticated with their genuine face. The *embedding* produced was, as expected, close to what was originally enrolled.
5. But `derive_key()` under the new secret produced a **completely different** `projection_seed`/`threshold_seed`/`permutation_seed` than the one that had generated the stored template.
6. **Mathematically:** even though the embedding `x` was genuinely close to the original, the candidate template was computed as `quantize(project(x, R_new), thresholds_new)` permuted by `perm_new` — an entirely different projection matrix, entirely different thresholds, entirely different permutation than what produced the stored template. Comparing two BioHash outputs generated under unrelated projection/quantization/permutation parameters is, by construction, statistically equivalent to comparing two **unrelated random bit strings** — even for the exact same input embedding.
7. **This was reproduced directly** during this project's own debugging, using the real `generate_template`/`compare` functions on a synthetic embedding: same embedding + same key → Hamming similarity **1.0000**; same embedding + a different key → Hamming similarity **0.4961** (chance level for 256 independent bits is 0.50).
8. **Operationally:** the observed authentication scores for the real user dropped to the 0.47–0.51 range — indistinguishable from an impostor, even though the person genuinely enrolled was the person authenticating.

<div class="important-box"><strong>IMPORTANT —</strong> This was not a bug in the matching math. Hamming similarity did exactly what it is defined to do. The bug was a missing safeguard: nothing verified that the configured secret was the one that had protected the existing data. Key continuity is that missing safeguard.</div>

<div style="page-break-after: always;"></div>

## PART 7 — What Makes the Template "Cancelable"?

### Analogy

```text
Password:            password  --SHA-256-->            stored_hash
                      (no secret key involved; the hash function alone is the transform)

Cancelable biometric: biometric --feature extraction--> embedding
                                 --MASTER_SECRET-keyed
                                   BioHash transform-->  protected_template
                      (the SAME embedding, under a DIFFERENT key, produces an
                       UNRELATED protected_template — this is what "cancelable" means)
```

A password hash has no revocation lever other than the user choosing a new password. A biometric characteristic cannot be "chosen again." Cancelability moves the revocation lever into the **key** instead: the same face can produce infinitely many different, unlinkable protected templates, one per key.

### The four properties, each judged honestly against this project

**1. Revocability** — *Theoretical meaning:* a compromised template can be invalidated and replaced without requiring a new biometric sample. *What this project provides:* yes — `POST /revoke-template` deactivates the ACTIVE template set and promotes a STANDBY set (pre-generated from the same embedding under a different `key_version`) to ACTIVE. *Demonstrated by:* `template_protection/revoke.py`, `evaluation/privacy_metrics.py::experiment_revocability`, and the multi-template-set architecture (`docs/MULTI_TEMPLATE_ARCHITECTURE.md`). *Not formally proven:* that revocation is unconditionally safe under an adaptive/multi-observation attacker — only measured empirically (expected pairwise Hamming distance ≈0.5 between key versions).

**2. Renewability** — *Theoretical meaning:* a fresh, usable template can be issued from the same biometric after revocation. *What this project provides:* yes — a STANDBY set already exists (pool size 4 by default) and can be promoted immediately; `POST /templates/{user_id}/generate` can create more from a fresh capture. *Demonstrated by:* `evaluation/results/template_set_promotion.csv` (a real logic experiment, on synthetic/stub embeddings — see Part 18's caveat). *Not formally proven:* nothing beyond this project's own tests.

**3. Non-invertibility** — *Theoretical meaning:* the original biometric cannot be recovered from the protected template alone. *What this project provides:* an **information-lossy, many-to-one mapping** (512 or 192 real dimensions collapse to 256 bits) — the project's own documentation explicitly calls this a data-transformation argument, not a proof. *Demonstrated by:* nothing beyond the structural fact that the transform is lossy. *Not formally proven, and explicitly disclaimed in the project's own docs:* this is **not** a cryptographic one-wayness guarantee like a hash function's preimage resistance; if an attacker also recovers the key material, partial reconstruction is a studied attack in the BioHashing literature, and this project does not claim otherwise.

**4. Unlinkability** — *Theoretical meaning:* two protected templates from the same person, produced for different purposes (e.g. different applications), cannot be linked to each other. *What this project provides:* yes, structurally — a different `application_id` (same user, same modality) produces an entirely different key, hence an unrelated template. *Demonstrated by:* `evaluation/privacy_metrics.py::experiment_diversity`, expected pairwise Hamming distance ≈0.5. *Not formally proven:* only measured on this project's own experimental data, not a general population.

<div class="viva-tip"><strong>VIVA TIP —</strong> A very safe, defensible one-line answer: "Cancelability here means the transform is keyed by a secret separate from the biometric itself, so revoking access means rotating the key, not the biometric — demonstrated experimentally by measuring that different-key outputs are statistically unrelated, not proven as a formal cryptographic guarantee."</div>

<div style="page-break-after: always;"></div>

## PART 8 — What Happens If the Template Is Stolen?

### Threat model: attacker obtains only `protected_template`

**What they have:** a packed bit array (32 bytes at 256 bits), plus whatever database metadata sits alongside it (`key_version`, `template_version`, `output_bits`, timestamps, `user_id`, `modality`, `application_id` — all plain, non-secret identifiers).

**What they do NOT have:** the raw biometric sample, the feature/embedding vector, `MASTER_SECRET`, the projection matrix, the quantization thresholds, or the permutation (none of these are stored anywhere — they are all recomputed on demand from `MASTER_SECRET` and never persisted).

**What can they do with the bits alone?** Very little, by design — they cannot run `generate_template()` forward for a candidate embedding without the key, so they cannot test whether a guessed face matches. They cannot reverse the bits back to an embedding without also knowing the projection matrix (i.e., without also knowing `MASTER_SECRET`), and even then the mapping is lossy. This project does **not** offer a formal proof of exactly how much information the bits alone leak — only the qualitative, documented claim that reconstruction from the template *alone* is infeasible in the same sense that a many-to-one hash reduces the search space, not in the sense of a proven cryptographic hardness bound.

### Stronger case: attacker obtains `protected_template` **and** `MASTER_SECRET`

Everything changes. With `MASTER_SECRET`, the attacker can derive the exact same `KeyMaterial` (projection matrix, thresholds, permutation) that produced any stored template for any user/modality/application/key_version — because key derivation is fully deterministic and public-algorithm (HKDF-SHA256, no secret beyond `MASTER_SECRET` itself). At that point, partial reconstruction of the underlying embedding becomes a studied attack in the BioHashing literature (this project's own docs cite this directly, rather than asserting the scheme remains safe under key compromise). This is exactly why `docs/PRIVACY_AND_SECURITY.md` calls `MASTER_SECRET` "the single highest-value secret in a real deployment."

### What this capstone provides vs. what would require formal analysis

| | This capstone provides | Would require further work |
|---|---|---|
| Template-alone compromise | A lossy, keyed transform; no plaintext biometric ever stored | A formal information-theoretic or cryptographic bound on residual leakage |
| Key + template compromise | Explicitly disclosed as a real risk, not hidden | A quantified reconstruction-attack resistance analysis |
| Revocation after compromise | Implemented and demonstrated (Part 7) | Formal proof that revoked/rotated templates remain unlinkable under an adaptive attacker |

<div class="important-box"><strong>IMPORTANT —</strong> Never claim the protected template is "as safe as a hashed password" — a password hash's security rests on a proven cryptographic one-way function over a low-entropy secret the user can change; this scheme's security rests on a secret-keyed, information-lossy transform over a biometric the user cannot change, and the project's own documentation is careful not to conflate the two.</div>

<div style="page-break-after: always;"></div>

## PART 9 — What Happens If MASTER_SECRET Changes?

**Case A — Same biometric, same MASTER_SECRET.** Preprocessing → embedding → protection are all deterministic given deterministic inputs, so the same capture (or a capture that reproduces the same embedding exactly) produces the same protected template. In practice, two *different* live captures of the same real person never produce byte-identical embeddings (natural capture variation), so this case is really "the resulting Hamming similarity is high, not necessarily 1.0" — genuine recaptures were observed in this project's own real logs in the 0.70s–0.88 range depending on modality and capture conditions, not always near-perfect.

**Case B — Same biometric, different MASTER_SECRET.** `derive_key()` produces entirely different seeds → entirely different projection matrix, thresholds, permutation → the resulting protected template is **statistically unrelated** to one generated under the old secret, **even for the exact same embedding**. Measured this session: Hamming similarity ≈0.496 for identical embeddings under two different keys — no better than the 0.50 chance floor for 256 independent bits.

**Case C — Different biometric, same MASTER_SECRET.** The embeddings differ (two different people), so the projected/quantized/permuted outputs differ — Hamming similarity is expected to sit near the project's measured impostor range (documented as roughly 0.55–0.66 for face, in the project's earlier calibration notes), generally lower than a genuine match but not necessarily as low as the pure-chance floor, since the projection/quantization is still applied consistently.

**Case D — Different biometric, different MASTER_SECRET.** Both sources of difference compound; similarity is expected to sit at or below the chance floor.

| Biometric | Secret | Expected protected template relationship |
|---|---|---|
| Same | Same | Same/deterministically reproducible — high Hamming similarity for genuine recaptures |
| Same | Different | Statistically unrelated — Hamming similarity near chance level (~0.50), regardless of how good the biometric match actually is |
| Different | Same | Different — Hamming similarity in the measured impostor range |
| Different | Different | Different — no meaningful signal either way |

This is exactly why the system depends on **key continuity** (Part 6): Case B is indistinguishable, from the Hamming-similarity number alone, from "this isn't the same person" — the system has no way to tell the two apart after the fact without an independent check that the secret itself hasn't drifted.

<div style="page-break-after: always;"></div>

## PART 10 — Hamming Similarity

### The exact implementation

`template_protection/matcher.py::compare(template_a, template_b, metric="hamming")`:

```python
return float(np.mean(template_a == template_b))
```

(An exact byte-for-byte equality fast path, using `hmac.compare_digest` for constant-time comparison, runs first and short-circuits to `1.0` if the two templates are identical; otherwise the Hamming similarity above is computed.)

### Mathematical definition, matching the code

For two equal-length binary templates `A` and `B`, each of `N` bits:

```text
Hamming distance:              d_H(A, B) = number of positions where A_i != B_i

Normalized Hamming distance:   d_H(A, B) / N

This project's score is a SIMILARITY, computed directly (not as 1 - distance/N
as a separate step, though the two are numerically identical):

    S_hamming(A, B) = matching_bits / N  =  1 - (d_H(A, B) / N)
```

### Worked example

```text
Template A: 1 0 1 1 0 0 1 0
Template B: 1 0 1 0 0 0 1 1
Compare:    =  =  =  x  =  =  =  x     (x = differing bit)

Differing bits: 2 (position 4 and position 8)
N = 8
d_H(A, B) = 2
Normalized Hamming distance = 2/8 = 0.25
S_hamming = 1 - 0.25 = 0.75
```

In this project's real deployment, `N = 256` (the configured `template_bits`), not 8 — this example uses 8 bits purely to make the by-hand counting legible.

### What a score like 0.97 / 0.85 / 0.74 / 0.50 actually means

It is the **fraction of bit positions that agree** between a freshly regenerated candidate template and the stored one — nothing more. It is **not a probability** that the two biometrics belong to the same person; it has no calibrated statistical interpretation on its own (that would require a genuine/impostor distribution analysis — see Part 11's calibration discussion). A score of exactly 0.50 is meaningful in a specific way: for a 256-bit template, two *unrelated* bit strings are expected to agree on ~50% of positions purely by chance — a score near 0.50 is the signature of "these two templates carry no shared signal," whether because the biometrics differ or because the keys differ (Part 9).

<div class="viva-tip"><strong>VIVA TIP —</strong> If asked "is 0.75 a 75% chance this is the right person?" — the correct answer is no: it is a bit-agreement fraction, not a probability. Whether 0.75 should be treated as a match or not is exactly what the threshold (Part 11) is for, and that threshold in this project was chosen from a handful of real observed genuine scores, not derived from a formal probability model.</div>

<div style="page-break-after: always;"></div>

## PART 11 — Thresholds

| Modality | Threshold | Stored in | Calibrated or manual? |
|---|---|---|---|
| Face | 0.80 | `evaluation/results/face_threshold.json` | **Manual** — `"source": "operator-specified"`, `eer`/`far`/`frr`/`auc` all `null` in the file. Chosen from real observed genuine scores (0.738–0.820 range) and a deliberate decision to keep it higher than the lowest observed genuine score, to avoid also accepting AI-generated/synthetic faces at a lower threshold. |
| Fingerprint | 0.90 (fallback) | No calibration file exists (`evaluation/results/fingerprint_threshold.json` does not exist) — falls back to `backend/config.py::Settings.match_threshold = 0.9` | **Uncalibrated fallback**, not chosen for fingerprint specifically at all |
| Voice | 0.80 | `evaluation/results/voice_threshold.json` | **Manual**, same "operator-specified" status as face, chosen from real observed genuine scores (0.797/0.848/0.879) |
| Fusion threshold | `mean(per-modality thresholds actually used)` | Computed at request time, `backend/services/authentication.py` | Derived, not independently set — and **informational only** (see below) |

### score vs. threshold vs. decision

```text
score      = Hamming similarity (Part 10), a measured number in [0, 1]
threshold  = a fixed cutoff this project chose per modality
decision   = score >= threshold   (this IS how a single modality's own pass/fail works)
```

`template_protection/matcher.py::accept(score, threshold, metric)` implements exactly `score >= threshold`.

### The fusion case is different — and this is a common trap

This project's default fusion policy is `FusionPolicy.ALL_REQUIRED`. **The fused score itself does NOT control the final decision.** `fusion/policy.py::evaluate_fusion_policy()`:

```python
authenticated = (len(failed_modalities) == 0)
```

where `failed_modalities` is computed from each **individual** modality's own `score >= its own threshold` check. The fused score (a weighted average across submitted modalities) is computed and reported for diagnostics, but under `ALL_REQUIRED` it plays **no role in the accept/reject decision** — a single failing modality denies access regardless of how high the fused average is. (Under the *other* policy this project implements, `WEIGHTED`, the fused score does matter, together with a per-modality floor veto — but `ALL_REQUIRED` is this project's actual default and the one the frontend uses.)

<div class="important-box"><strong>IMPORTANT —</strong> This is the single most commonly-misunderstood part of this project's design. Memorize the distinction: individual modality thresholds gate individual modality pass/fail; the fusion POLICY (not the fusion score) decides the final outcome.</div>

<div style="page-break-after: always;"></div>

## PART 12 — Multimodal Fusion

### How the per-modality scores are produced (independently)

Each submitted modality is matched **completely independently** — its own preprocessing, its own embedding, its own key-derived protected template regeneration, its own Hamming comparison against its own stored template. `backend/services/authentication.py::authenticate_samples()` loops over exactly the submitted modalities and calls `ModalityService.authenticate()` for each.

### How the scores are combined

`fusion/score_fusion.py::fuse_scores()` / `resolve_normalized_weights()`. This project uses **equal weights** among whichever modalities were actually submitted:

```text
w_i = 1 / N              (N = number of modalities submitted this request; no
                           caller in this codebase currently passes custom weights)

fused_score = sum(w_i * s_i) / sum(w_i)     for i in submitted modalities
```

An absent modality is never treated as a zero score — it simply doesn't enter the sum at all.

### Fusion threshold and final decision policy

`fusion_threshold = mean(the per-modality thresholds actually used this request)` — computed and shown, but (Part 11) **not** what decides the outcome under `ALL_REQUIRED`. The actual decision:

```text
authenticated = True  IFF every submitted modality's own score >= its own threshold
```

### Worked example, using this project's own real logged numbers

Face = 0.7734 (below the 0.80 face threshold), Voice = 0.8828 (above the 0.80 voice threshold):

```text
fused_score = (0.7734 + 0.8828) / 2 = 0.8281
fusion_threshold = (0.80 + 0.80) / 2 = 0.80
```

The fused score (0.8281) is *above* the fusion threshold (0.80) — if the decision were simply `fused_score >= fusion_threshold`, this would grant access. **It does not.** Face's own score (0.7734) is below Face's own threshold (0.80), so Face is in `failed_modalities`, so under `ALL_REQUIRED`, `authenticated = False` — **ACCESS DENIED**, regardless of the fused number.

This exact behavior was introduced (per this project's own fusion-module docstring) to fix a previously audit-flagged issue: the older, simpler "compare the fused average to a threshold" approach let one very strong modality compensate for another modality's individual failure — `ALL_REQUIRED` closes that gap by never letting the average alone decide.

<div style="page-break-after: always;"></div>

## PART 13 — Why Not Store Raw Biometric Features?

| Option | What's stored | Benefit | Limitation |
|---|---|---|---|
| **1. Raw face image** | The actual pixels | Maximum utility for the matcher; trivially re-derivable into anything | Worst possible privacy exposure — a breach directly exposes the person's actual face; no revocation possible at all |
| **2. Raw embedding** | A 512/192-d float vector | Smaller than an image, still very discriminative for matching | **Still a raw, unprotected, permanent biometric representation.** Embeddings are not equivalent to passwords — they are dense, mathematically structured summaries of the biometric itself, and are known in the literature to be at least partially invertible toward a recognizable likeness of the original biometric with the right techniques. No revocation possible: the same face always produces the same (or a very close) embedding. |
| **3. Protected/cancelable binary template (this project's actual choice)** | A 256-bit array, keyed by `MASTER_SECRET` | Revocable (rotate the key), unlinkable across applications, information-lossy | Matching is coarser (a bit-agreement fraction, not the original embedding's fine-grained geometry); security still ultimately rests on `MASTER_SECRET` staying secret (Part 8) |

The critical point for a viva: **Option 2 is not a safe middle ground.** An embedding is still sensitive, permanent biometric data — this project's whole design argument is that "just store the embedding instead of the image" does not solve the actual problem (no revocability, no unlinkability), which is why Option 3's keyed transform exists at all.

<div style="page-break-after: always;"></div>

## PART 14 — Important Limitations — "What I Should NOT Claim in a Viva"

- Do not claim that hashing automatically makes a biometric template irreversible — this project's transform is lossy by construction, not proven one-way.
- Do not claim that BioHashing is equivalent to a password hash — a password hash has no secret key; this scheme's security depends entirely on `MASTER_SECRET` staying secret.
- Do not claim perfect non-invertibility without formal analysis — the project's own docs explicitly disclaim this ("not a computational-hardness proof").
- Do not claim perfect cancelability unless multiple independent transformed templates have actually been demonstrated — this project does demonstrate this experimentally (`evaluation/privacy_metrics.py`), on its own limited test data, not as a general proof.
- Do not claim cryptographic security merely because SHA-256/HKDF is present — HKDF here is a key-derivation tool, not itself a proof that the overall scheme is secure.
- Do not call Hamming similarity a probability — it is a bit-agreement fraction.
- Do not say changing `MASTER_SECRET` is harmless — existing protected templates become incompatible (Part 9), which is exactly why key continuity exists.
- Do not claim liveness or deepfake resistance — **not implemented**, explicitly on hold, and the project's own privacy documentation states there is no challenge-response or liveness binding an `/authenticate` request to a live capture.
- Do not claim the thresholds are statistically calibrated — face and voice are operator-specified from a handful of real observations; fingerprint uses an uncalibrated fallback.
- Do not claim FAR/FRR/EER have been measured at the *protected-template* level — the EER/AUC numbers that do exist in this repository (Part 18) were measured on raw embeddings, in a different domain from the deployed Hamming-similarity threshold.
- Do not claim the system has been evaluated at scale — face's committed evaluation used 62 identities closed-set; fingerprint used 600 subjects with a measured ~31% EER (self-documented as the weakest modality); voice used 24 speakers.

<div style="page-break-after: always;"></div>

## PART 15 — Viva Questions and Model Answers

### A. Basic questions

**1. What is a biometric template?**
In general biometrics, it's the stored representation used for matching. In this project specifically, nothing resembling an unprotected template is ever stored — the closest analog (the embedding) is computed fresh for every request and discarded; only the *protected* (BioHash) template is persisted.

**2. Why can't you simply hash a face embedding with SHA-256?**
Two reasons. First, SHA-256 is designed for exact-match verification — flipping one bit of input changes the output completely (the avalanche effect), but two genuine captures of the same face never produce bit-identical embeddings, so a SHA-256 hash would never match twice for the same real person. Second, SHA-256 has no secret key, so it provides no revocability — the same embedding always hashes to the same value forever.

**3. What makes your template cancelable?**
The transform is keyed by `MASTER_SECRET`-derived seeds. The same embedding, protected under a different key (e.g. a different `key_version` after revocation), produces a statistically unrelated protected template — so compromising one template doesn't compromise the person's biometric identity everywhere it's used.

**4. Where exactly is the secret used?**
`MASTER_SECRET` is the input key material to `template_protection/hkdf_keys.py::derive_key()`, which produces three HKDF-SHA256 seeds that drive the projection matrix, the quantization thresholds, and the permutation in `template_protection/transform.py`.

**5. What happens if the secret changes?**
Every downstream seed changes, so the projection/quantization/permutation all change. A candidate template regenerated under the new secret is statistically unrelated to a stored template protected under the old one — even for the exact same person's biometric (Part 9).

**6. Why can't you recover the original face from the protected template?**
The transform maps a 512-dimensional real vector down to 256 bits — a many-to-one, information-lossy reduction. Reconstructing the exact original embedding from the bits alone is infeasible as a consequence of that information loss, not because of a proven cryptographic hardness guarantee.

**7. Is your BioHash cryptographically irreversible?**
No — and the project's own documentation is explicit about this. It's an information-lossy transform, argued from data reduction, not a hash function with a proven preimage-resistance bound.

**8. Why are you using Hamming similarity?**
The protected template is a bit vector; Hamming similarity is the natural, cheap metric for comparing two equal-length bit vectors — the fraction of positions that agree.

**9. What is the difference between Hamming distance and Hamming similarity?**
Hamming distance counts differing bit positions; Hamming similarity (this project's actual metric) is the fraction of *matching* positions, i.e. `1 - (distance / N)`.

**10. What is the role of HKDF?**
It deterministically expands one input secret (`MASTER_SECRET`) into multiple independent, fixed-length pseudorandom seeds, each used for a different stage of the transform, without ever needing to store those seeds.

### B. Technical questions

**11. Why do you need key continuity?**
Because a wrong or changed `MASTER_SECRET` produces valid-looking but meaningless authentication scores (near chance level) with no obvious error — key continuity detects this at startup and refuses to run rather than silently failing every user.

**12. What happens if the database is stolen?**
The attacker gets protected template bytes and metadata (user IDs, modality, key version, timestamps) — never a raw image, raw embedding, or `MASTER_SECRET`. Without the key, they cannot generate a valid candidate template to test guesses against, and cannot straightforwardly reverse the bits back to a biometric.

**13. What happens if MASTER_SECRET is stolen?**
Much worse — the attacker can now derive the exact key material for any user/modality/application and, combined with the stolen templates, attempt partial reconstruction (a studied BioHashing-literature attack), and could also forge valid-looking templates for arbitrary embeddings.

**14. Why not encrypt the biometric template instead?**
Encryption is reversible by design (given the key) — decrypting gives back the exact original data. This project's transform is deliberately *not* reversible even with correct-looking access, because the goal is that the stored value should never need to be treated as equivalent to the raw biometric, even inside the system.

**15. How is cancelability different from encryption?**
Encryption protects confidentiality of a fixed underlying value that can be recovered exactly with the key. Cancelability's goal is different: producing a *new, unlinkable* protected value from the same underlying biometric on demand, specifically so a compromised value can be abandoned rather than decrypted and reused.

**16. Can two different users have similar protected templates?**
In principle, two different people's genuinely different embeddings, keyed under two different per-user derived keys, are expected to land near the measured impostor range (documented roughly 0.55–0.66 for face) — not identical, but not guaranteed to be far apart either; this project has not formally bounded worst-case collision likelihood.

### C. Mathematical questions

**17. What is FAR?**
False Acceptance Rate — the rate at which an impostor's attempt is wrongly accepted.

**18. What is FRR?**
False Rejection Rate — the rate at which a genuine user's attempt is wrongly rejected.

**19. What is EER?**
Equal Error Rate — the operating point where FAR equals FRR; this project has measured EER at the *raw embedding* level for face (1.0%, closed-set), fingerprint (30.8%), and voice (2.3%) — but **not** at the deployed protected-template Hamming-similarity level.

**20. How was your threshold selected?**
For face and voice: manually, by an operator, after observing real genuine-attempt scores and choosing a cutoff above the lowest observed genuine score but not so low it would also admit low-quality/synthetic input. For fingerprint: not selected at all — it uses an uncalibrated 0.9 fallback.

**21. Does your current threshold have statistical calibration?**
No, for any of the three active modalities — the two `*_threshold.json` files explicitly record `"source": "operator-specified"` with `eer`/`far`/`frr`/`auc` all `null`.

**22. Why is a similarity of 0.5 significant?**
For an `N`-bit template, two statistically unrelated bit strings are expected to agree on about 50% of positions purely by chance — a score near 0.5 is the signature of "no shared signal," whether from a different key or a different biometric.

### D. Security questions

**23. Why did changing the secret cause authentication failure?**
Because every stage of the transform is keyed by secret-derived seeds — a different secret produces a different projection matrix, thresholds, and permutation, so the regenerated candidate template shares no meaningful structure with a template stored under the old secret (Part 9, Case B).

**24. Why can't the old templates simply be reused?**
Because "reusing" them would require deriving the *same* key that originally produced them — and that key traces back to a secret that is no longer available. There is no operation that converts an old protected template into a new one without either the original secret or a fresh biometric capture.

**25. What is stored in your database?**
Protected template bytes, template/key version numbers, lifecycle status (active/standby/revoked), audit metadata (scores, thresholds, decisions — never raw biometric data), and a one-way `MASTER_SECRET` fingerprint. Never a raw image, raw audio, raw embedding, or the secret itself.

**26. What exactly is regenerated during authentication?**
The full pipeline from the live capture forward: preprocessing, embedding, L2 normalization, key derivation (using the stored template's own recorded `key_version`), projection, quantization, and permutation — producing a brand-new candidate protected template, which is then compared against the stored one.

**27. Does the server ever store the raw face image?**
No — confirmed directly from the code: the raw image exists only as a local variable for the duration of one preprocessing call and is never written to disk or database.

### E. Implementation/code questions

**28. What is the difference between the biometric embedding and the protected template?**
The embedding is a real-valued, L2-normalized vector (512 or 192 dimensions) produced directly by a trained model — sensitive, unprotected, never stored. The protected template is a 256-bit array produced by passing that embedding through the secret-keyed BioHash transform — this is the only form ever persisted.

**29. What security guarantees are actually demonstrated by your tests?**
The test suite demonstrates: HKDF determinism (same inputs → same seeds), BioHash determinism (same embedding + key → same template), that different keys or different embeddings produce statistically unrelated templates (~0.5 Hamming similarity, `evaluation/privacy_metrics.py`), that revocation invalidates the old key version, and — this session — that a wrong `MASTER_SECRET` is caught at startup rather than silently accepted. It does **not** demonstrate any formal cryptographic security bound, nor a calibrated FAR/FRR at the protected-template level.

**30. What would you improve in the template-protection scheme in future research?**
A formal calibration of thresholds at the protected-template (not raw-embedding) level with real multi-person genuine/impostor data; a quantitative analysis of reconstruction risk under key compromise; and, separately from template protection itself, adding liveness/presentation-attack detection, since the current scheme protects *what happens to a captured sample*, not *whether the sample was genuinely live*.

### F. Critical/challenging questions

**"Isn't this just security theater if the secret is stored in a plain `.env` file?"**
The `.env` file is the deliberate, documented trust boundary — the project doesn't claim `MASTER_SECRET` is itself protected by some further mechanism; it claims that *given* the secret stays confidential (an operational requirement, not a cryptographic one), the stored templates are cancelable and the raw biometric is never exposed. This is stated as an explicit assumption, not hidden.

**"If two different keys give ~0.5 similarity and two different people also give ~0.5-0.66, how do you know a denial is really the wrong person and not a stale key?"**
You don't, from the Hamming score alone — this is exactly the ambiguity that motivated key continuity: instead of relying on the authentication score to distinguish "wrong person" from "wrong key," the system independently verifies key continuity at startup, so by the time any authentication score is produced, the key is already known to be the correct one.

<div style="page-break-after: always;"></div>

## PART 16 — One-Page Cheat Sheet

**KEY TERMS**

- **Biometric:** the raw physical characteristic (face, fingerprint, voice) — never stored.
- **Embedding:** a fixed-length, L2-normalized numeric vector extracted by a trained model (512-d face/fingerprint, 192-d voice) — never stored.
- **Template (in this project):** there is no separate unprotected "template" stage — the embedding (or, for face, the centroid of up to 5 embeddings) goes directly into protection.
- **Cancelable template:** the protected output — revocable because it is keyed by a secret, not just derived from the biometric alone.
- **BioHash:** this project's specific transform — orthonormal projection → keyed thresholded quantization → keyed permutation.
- **MASTER_SECRET:** the one root secret; environment-only, never in the database, never in a template.
- **HKDF:** HMAC-based key derivation function — expands `MASTER_SECRET` into three independent seeds per (user, modality, application, key_version).
- **Hamming distance:** number of differing bit positions between two equal-length bit strings.
- **Hamming similarity:** fraction of *matching* bit positions — this project's actual score, in `[0, 1]`.
- **Threshold:** the fixed cutoff a modality's own score must meet or exceed to pass individually.
- **FAR:** False Acceptance Rate — impostors wrongly accepted.
- **FRR:** False Rejection Rate — genuine users wrongly rejected.
- **EER:** the threshold where FAR = FRR; measured in this project only at the raw-embedding level, not the deployed protected-template level.

**THE 30-SECOND EXPLANATION**

"Our system does not store the raw biometric, or even the raw feature embedding, as the authentication credential. The biometric is converted into a fixed-length embedding, which is transformed using a `MASTER_SECRET`-derived BioHash process — an orthonormal random projection, a keyed thresholded quantization, and a keyed permutation — into a 256-bit protected template. During authentication, the exact same pipeline is applied to the live biometric, and the resulting binary template is compared against the stored one using Hamming similarity — the fraction of matching bits, not a probability. Because every stage of the transform is deterministically derived from `MASTER_SECRET`, the secret must remain consistent: if it changes, previously generated protected templates become statistically unrelated to anything regenerated afterward, which is exactly why our system implements a key-continuity check that verifies the configured secret against a one-way fingerprint of the original secret before the server will even start."

<div style="page-break-after: always;"></div>

## PART 17 — Visual Diagrams

**Diagram 1 — Enrollment pipeline**
```text
Raw biometric
   -> preprocessing (modality-specific)
   -> feature extraction (trained model)
   -> L2-normalized embedding
   -> [face only] centroid of up to 5 embeddings, re-normalized
   -> BioHash / template protection (MASTER_SECRET-keyed)
   -> binary protected template (256 bits, packed to bytes)
   -> database (protected_templates table)
```

**Diagram 2 — Authentication pipeline**
```text
Live biometric
   -> preprocessing (SAME code path as enrollment)
   -> feature extraction (SAME model/checkpoint)
   -> L2-normalized embedding
   -> SAME BioHash transformation, using the stored template's key_version
   -> new binary "candidate" protected template
   -> Hamming comparison vs. the STORED protected template
   -> similarity score (bit-agreement fraction, in [0, 1])
   -> per-modality threshold check -> fusion policy -> ACCESS_GRANTED / ACCESS_DENIED
```

**Diagram 3 — Key dependency**
```text
MASTER_SECRET
   -> HKDF-SHA256 (per user, modality, application_id, key_version)
   -> three seeds: projection_seed, threshold_seed, permutation_seed
   -> BioHash (orthonormal projection -> keyed quantize -> keyed permute)
   -> protected template
```

**Diagram 4 — Key continuity**
```text
MASTER_SECRET
   -> fingerprint_master_secret()  (a SEPARATE, fixed-context HKDF-SHA256 call)
   -> one-way 32-byte fingerprint
   -> master_secret_fingerprint table (one row)
   -> compared at every server startup against a fresh fingerprint of the
      currently configured secret
   -> match: server starts.  mismatch: server refuses to start.
```

**Diagram 5 — Threat model: what is stored vs. what must stay secret**
```text
Raw biometric  --------------------------------  NEVER stored, NEVER secret-dependent
     |                                            (doesn't exist after one request)
     v
Embedding  ------------------------------------  NEVER stored (but IS sensitive if it were)
     |
     v
Protected template  ---------------------------  STORED (database) - safe to store
                                                  ONLY because MASTER_SECRET is not

MASTER_SECRET  ---------------------------------  MUST STAY SECRET (environment only,
                                                  never database, never logged)
```

<div style="page-break-after: always;"></div>

## PART 18 — Exact Numbers and Parameters

| Parameter | Current value | Source file | Purpose |
|---|---|---|---|
| Face embedding dimension | 512 | `models/face/inference.py` | InceptionResnetV1 output size |
| Fingerprint embedding dimension | 512 | `models/fingerprint/inference.py` | ResNet50 + projection head output size |
| Voice embedding dimension | 192 | `models/voice/inference.py` | ECAPA-TDNN output size |
| Iris embedding dimension | 256 | `models/iris/inference.py` | ResNet18 + projection head output size (no trained checkpoint exists) |
| Protected template length (deployed) | 256 bits | `backend/config.py::Settings.template_bits` | Actual runtime BioHash output length |
| Protected template length (library default) | 128 bits | `template_protection/biohash.py::DEFAULT_OUTPUT_BITS` | Generic default when `output_bits` isn't explicitly passed (not what's actually used) |
| HKDF seed length | 32 bytes | `template_protection/hkdf_keys.py::SEED_LENGTH_BYTES` | Length of each of the 3 derived seeds |
| Secret-fingerprint length | 32 bytes | `backend/secret_fingerprint.py::FINGERPRINT_LENGTH_BYTES` | Length of the one-way `MASTER_SECRET` fingerprint |
| Secret-fingerprint HKDF context | `b"master_secret_fingerprint/v1"` | `backend/secret_fingerprint.py::FINGERPRINT_CONTEXT` | Fixed, public, namespace-separating string |
| Quantization jitter scale | `0.5 * std(projected)` | `template_protection/transform.py::quantize()` | Per-position random threshold spread |
| Face threshold | 0.80 | `evaluation/results/face_threshold.json` | Operator-specified accept cutoff |
| Voice threshold | 0.80 | `evaluation/results/voice_threshold.json` | Operator-specified accept cutoff |
| Fingerprint threshold | 0.90 (fallback) | `backend/config.py::Settings.match_threshold` | Uncalibrated default (no calibration file exists) |
| Fusion policy (default) | `ALL_REQUIRED` | `fusion/config.py::DEFAULT_FUSION_POLICY` | Every submitted modality must individually pass |
| Fusion weights | `1/N` (equal) | `fusion/score_fusion.py::resolve_normalized_weights()` | No custom weighting is used anywhere in this codebase |
| Template pool size (per user) | 4 | `backend/config.py::Settings.template_pool_size` | 1 ACTIVE + 3 STANDBY template sets |
| Face enrollment samples | 5 requested, 3 minimum accepted | `backend/services/face_enrollment.py::MIN_VALID_POSES` | Centroid input count |
| Random seed algorithm | `numpy.random.default_rng` (PCG64) | `template_protection/transform.py` | Deterministic, portable, non-entropy-consuming |
| Projection matrix orthonormality | Full, when `output_bits <= embedding_dim` | `template_protection/transform.py::build_orthonormal_projection` | Face/fingerprint (512-d): fully orthonormal at 256 bits. Voice (192-d): composed of 2 blocks at 256 bits |
| Measured same-key/same-embedding Hamming similarity | 1.0000 | Reproduced this session, synthetic vector, real code | Sanity check of determinism |
| Measured different-key/same-embedding Hamming similarity | 0.4961 | Reproduced this session, synthetic vector, real code | Chance-floor confirmation |
| Face raw-embedding EER (measured) | 1.0% | `evaluation/results/face_metrics.csv` | Closed-set, 62 identities — **not** the protected-template EER |
| Fingerprint raw-embedding EER (measured) | 30.8% | `evaluation/results/fingerprint_metrics.csv` | Subject-disjoint, 600 subjects |
| Voice raw-embedding EER (measured) | 2.3% | `evaluation/results/voice_metrics.csv` | Per-speaker split, 24 speakers |
| Protected-template-level FAR/FRR/EER | Not verified from implementation | — | No code in this repository computes this |

<div style="page-break-after: always;"></div>

## PART 19 — Source Code References

`template_protection/hkdf_keys.py`
Function: `derive_key(master_secret, *, application_id, user_id, modality, key_version)`
Purpose: expands `MASTER_SECRET` into three independent HKDF-SHA256 seeds (projection, threshold, permutation) for one (user, modality, application, key_version) context.

`template_protection/transform.py`
Function: `build_orthonormal_projection(seed, output_bits, embedding_dim)`
Purpose: deterministically builds the keyed random orthonormal (or block-orthonormal) projection matrix via QR decomposition of a seeded Gaussian matrix.

`template_protection/transform.py`
Function: `quantize(projected, seed)`
Purpose: binarizes the projected vector against a per-position, key-seeded, data-scaled random threshold.

`template_protection/transform.py`
Function: `apply_permutation(bits, seed)`
Purpose: reorders the quantized bits with a key-derived permutation.

`template_protection/biohash.py`
Function: `generate_template(embedding, key, output_bits)`
Purpose: orchestrates the three transform stages above into the single protected-template-generation call.

`template_protection/matcher.py`
Function: `compare(template_a, template_b, metric="hamming")`
Purpose: computes Hamming similarity (fraction of matching bits) between two protected templates.

`template_protection/matcher.py`
Function: `accept(score, threshold, metric)`
Purpose: implements `score >= threshold` for one modality's individual pass/fail.

`backend/secret_fingerprint.py`
Function: `fingerprint_master_secret(master_secret)`
Purpose: one-way HKDF-SHA256 fingerprint of `MASTER_SECRET` over a fixed public context, used only to detect secret drift.

`backend/key_continuity.py`
Functions: `evaluate_key_continuity()`, `enforce_key_continuity_at_startup()`
Purpose: compares the current secret's fingerprint against the stored one at every server startup; hard-fails the server if templates exist under an unverifiable or mismatched secret.

`backend/database/models.py`
Class: `ProtectedTemplate`
Purpose: the SQLAlchemy model for the stored protected biometric template and its lifecycle metadata (never a raw embedding).

`backend/database/models.py`
Class: `MasterSecretFingerprint`
Purpose: the single-row table holding the one-way `MASTER_SECRET` fingerprint.

`fusion/score_fusion.py`
Function: `fuse_scores(scores, weights=None)`
Purpose: equal-weighted average of whichever modalities were actually submitted.

`fusion/policy.py`
Function: `evaluate_fusion_policy(scores, individually_authenticated, policy, fusion_threshold)`
Purpose: implements `ALL_REQUIRED` (every submitted modality must individually pass) as the actual accept/reject decision.

`evaluation/privacy_metrics.py`
Functions: `experiment_revocability()`, `experiment_diversity()`, `experiment_similarity_preservation()`
Purpose: the experimental measurements behind this project's cancelability claims (Part 7) — real code exercising the real template-protection path, on this project's own synthetic/limited test data.

<div style="page-break-after: always;"></div>

## PART 20 — Final "Teacher Traps"

**1. Hashing vs. encryption.** Hashing is one-way by design; encryption is reversible with the right key. This project's BioHash is closer in spirit to hashing (lossy, not meant to be reversed) but is keyed like an encryption scheme — it is neither a pure hash nor pure encryption; it's a purpose-built cancelable transform.

**2. Hashing vs. BioHashing.** A generic hash (e.g. SHA-256) has no tolerance for small input variation — one differing bit changes everything. BioHashing-style schemes like this project's are specifically designed so that *similar* biometric inputs still produce *similar* (high-Hamming-similarity) outputs, which a cryptographic hash deliberately does not provide.

**3. BioHashing vs. ordinary SHA-256.** SHA-256 has no secret key and is meant for exact-match integrity checking. This project's BioHash uses `MASTER_SECRET`-derived randomness at every stage and is meant for approximate-match biometric comparison, not exact equality.

**4. Cancelability vs. confidentiality.** Confidentiality means "the value stays hidden." Cancelability means "the value can be replaced without needing a new underlying secret (the biometric)." This project provides both to different degrees — confidentiality of the biometric (never stored raw) and cancelability of the stored template (revocable via key rotation).

**5. Non-invertibility vs. one-way hashing.** One-way hashing (e.g. SHA-256) has a mathematically studied preimage-resistance property. This project's non-invertibility claim is a *data-reduction* argument (many-to-one dimensionality collapse), explicitly not the same class of guarantee.

**6. Hamming distance vs. Hamming similarity.** Distance counts *differing* bits; similarity (this project's actual metric) is the fraction of *matching* bits — numerically, `similarity = 1 - (distance / N)`.

**7. Similarity score vs. probability.** The score is a deterministic bit-agreement fraction from one comparison — it carries no calibrated statistical meaning about likelihood unless a separate genuine/impostor distribution analysis has been done (which this project has done only at the raw-embedding level, not the protected-template level).

**8. MASTER_SECRET vs. secret fingerprint.** The secret is the one root value, never stored. The fingerprint is a one-way, non-reversible derivative of it, stored specifically to detect if the secret has changed — it cannot be used to recover the secret or to derive any template key.

**9. Raw embedding vs. protected template.** The embedding is a real-valued, unprotected, permanent representation of the biometric — never stored. The protected template is the keyed, quantized, information-lossy output that actually gets persisted.

**10. Threshold vs. cryptographic security.** A threshold is a statistical/engineering decision boundary for accept/reject; it says nothing about whether the underlying transform is cryptographically hard to invert. The two are independent claims, and this project only makes the former (and even that, only informally calibrated for two of three active modalities).

**11. Key rotation vs. key replacement.** Key rotation (`key_version` increment, `template_protection/revoke.py`) is a *controlled*, intentional process that also regenerates and stores a new STANDBY template set from the same embedding at enrollment time. Key replacement (changing `MASTER_SECRET` itself) is a much bigger, project-wide event that invalidates *every* existing template for *every* user at once — which is exactly why key continuity exists to catch it.

**12. Same biometric + different key.** Statistically unrelated protected templates, even though the underlying person is genuinely the same (Part 9, Case B) — this is the exact incident this project discovered and fixed.

**13. Different biometric + same key.** Different protected templates, in the measured impostor similarity range — this is the intended, desired behavior of the matcher (correctly rejecting a different person).

**14. Authentication score vs. final fusion decision.** A per-modality score feeds a per-modality pass/fail against its own threshold. Under this project's `ALL_REQUIRED` policy, the *final* decision is "did every submitted modality individually pass" — not any function of the fused average score (Part 11, Part 12).

---

*End of document. Generated from a direct inspection of the repository at commit `5c67be1`. No algorithm, parameter, or security property described above was invented — every claim traces to a cited file, and every place the implementation falls short of a textbook property is stated explicitly rather than assumed away.*
