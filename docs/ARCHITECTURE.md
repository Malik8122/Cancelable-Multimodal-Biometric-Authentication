# Architecture

## System flow

```text
                              USER
                               |
                               v
                     Biometric Acquisition
                               |
             +-----------------+-----------------+
             v                                   v
           FACE                                VOICE
             |                                   |
             v                                   v
    preprocessing/face                    preprocessing/voice
    (MTCNN detect + crop)                 (16 kHz mono, VAD, RMS norm,
                                           4 s segment, 80-bin log-mel)
             |                                   |
             v                                   v
     models/face                           models/voice
   (InceptionResnetV1,                   (ECAPA-TDNN, 192-d,
    VGGFace2 pretrained,                  ArcFace trained)
    ArcFace fine-tuned)
             |                                   |
             +-----------------+-----------------+
                      v (Phase 1 boundary)
             BaseEmbedder.extract_embedding()
             -> fixed-length, L2-normalized embedding
                      |
                      v
          template_protection/hkdf_keys.py + biohash.py
   (HKDF key v1..vN -> keyed orthonormal projection -> keyed quantize -> keyed permute)
                      |
                      v
     Template SET pool (each set = one template per enrolled modality):
        Set 1 [Face V1|Voice V1]                 ACTIVE
        Sets 2..N  (same shape)                  STANDBY
                      |
                      v
   backend/ (FastAPI + SQLite/PostgreSQL: protected templates + set lifecycle only)
                      |   authentication compares the ACTIVE set only, never mixing sets
                      v
              fusion/ (ALL_REQUIRED | WEIGHTED)
        per-modality similarities stay internal -> ONE fusion similarity
                      |
                      v
              ACCESS GRANTED / ACCESS DENIED

  Revocation: ACTIVE set -> REVOKED, oldest STANDBY set -> ACTIVE, all modalities together
  (authorized by a biometric match against the ACTIVE set; 409 when the set pool is exhausted).
  Details: docs/MULTI_TEMPLATE_ARCHITECTURE.md
```

## Why one interface for very different modalities

`models/common/base_embedder.py` defines `BaseEmbedder`, which every
modality's embedder subclasses. The contract is deliberately narrow:

```python
embedding = embedder.extract_embedding(image)  # -> np.ndarray, L2-normalized
```

This is what lets `template_protection/` (Phase 2) and `fusion/` (Phase 3)
treat face and voice uniformly, without knowing anything about MTCNN or
log-mel extraction — those details are fully contained in each modality's
`preprocessing/*.py` and `models/*/inference.py` (voice's log-mel spectrogram
is passed as a 3-channel array to honor the same contract). Swapping a
backbone later, or adding a modality, only requires the new class to honor
this same interface.

## Mock mode

Before a modality has a trained checkpoint (fresh clone, before running the
relevant Colab notebook), `BaseEmbedder` falls back to a deterministic
pseudo-random embedding derived from the image's own pixel content
(`_mock_embedding`). This exists purely so the rest of the system — tests,
the backend once it exists, the fusion layer — can be exercised end-to-end
without requiring a GPU or trained weights. Mock embeddings are **not**
biometrically meaningful and must never back a real enrollment/authentication
decision; `ModalityPipeline.is_mock` exposes this flag so calling code can
refuse to use mock embeddings outside of tests/demos.

## Classical CV vs. deep learning, per modality

| Modality | Classical CV | Deep learning |
|---|---|---|
| Face | bounding-box crop (default `FACE_ALIGNMENT=bbox`); optional 5-landmark similarity alignment (`FACE_ALIGNED`), see `evaluation/reports/FACE_ALIGNMENT_DECISION.md` | MTCNN detection/landmarks; InceptionResnetV1 embedding |
| Voice | resampling to 16 kHz mono; energy VAD; RMS normalization; fixed 4 s segment; 80-bin log-mel | ECAPA-TDNN embedding |
| Hand gesture | wrist/palm-centre translation, palm-size scaling, gap interpolation, smoothing, resampling to 64 frames, 12 hand-crafted features; DTW matching | MediaPipe Hands landmarks (pretrained, runs in the browser) |

## Model decisions and why (Phase 1)

- **Face — InceptionResnetV1 pretrained on VGGFace2** (`facenet-pytorch`):
  strong off-the-shelf embeddings, pip-installable, needs only light
  fine-tuning (last inception block) rather than training from scratch.
- **Voice — ECAPA-TDNN (SpeechBrain), 192-d embedding:** a state-of-the-art
  speaker-verification architecture; see `docs/VOICE_MODEL.md`.

Both training heads use the same ArcFace-style loss
(`models/common/arcface.py`) so the modalities are trained consistently
rather than with different, harder-to-compare recipes.

## Flexible multimodal authentication (V3)

The user decides which modalities to enroll and which to present; a building is context only (details:
`docs/MULTI_TEMPLATE_ARCHITECTURE.md`):

```text
  User Enrollment Profile        Building (context only)          Authentication Engine
  which modalities the user      id, name, clearance level,       authenticates + fuses EXACTLY the
  enrolled: NOT_REGISTERED /     description - no biometric       modalities the user submits
  REGISTERED / UPDATED /         policy (config/buildings.json)   (POST /authenticate/fusion)
  RETRY_REQUIRED (voice)                 |                                  ^
            \____________ submitted modalities all enrolled? _____________/
                      no  -> ENROLLMENT_REQUIRED (409, nothing verified)
                      yes -> fuse the submitted modalities -> ACCESS_GRANTED | ACCESS_DENIED
```

Template sets hold templates only for the modalities enrolled so far (T1 ACTIVE, T2-T4 STANDBY for each); enrolling another
modality later adds its templates to the existing live sets without regenerating the others.
