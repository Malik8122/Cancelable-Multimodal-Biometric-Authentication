# Final project validation

> **Modality change (2026-09-27).** Iris and fingerprint have been removed from the system: the implemented
> modalities are **face and voice**, and the `AT_LEAST_TWO` fusion policy (which needed three modalities) is gone.
> This document is a dated record. Its iris / fingerprint material (models, datasets, results, thresholds,
> three-modality fusion) is kept for history only and no longer describes the code. Current state: `README.md`,
> `docs/ROADMAP.md`; current numbers: `evaluation/PAPER_RESULTS_INDEX.md`.

Date: 2026-09-25; repository at `main` @ `873c13e` plus uncommitted work.

Every metric carries an evidence label:

| Code | Meaning |
|---|---|
| [A] | REAL DATA |
| [B] | SYNTHETIC |
| [C] | SOFTWARE TEST |
| [D] | CONFIGURATION |
| [E] | DERIVED |
| [F] | FUTURE / NOT AVAILABLE |

Each number is traced to its file and script in `evaluation/PAPER_RESULTS_INDEX.md`. The findings behind it are in
`docs/MASTER_PROJECT_AUDIT.md`.

## 1. Architecture

- Browser capture → FastAPI → per-modality preprocessing → embedding → L2 normalization → 256-bit BioHash under an
  HKDF key → Hamming similarity against the ACTIVE template set → calibrated estimate → threshold → fusion → grant or
  deny → audit log [D].
- Modalities:
  - face and voice are deployed;
  - fingerprint is deployed and optional;
  - iris is not implemented, because there is no checkpoint and it only runs on a mock embedder [D].
- The user presents any enrolled subset of modalities [D].

## 2. Face alignment

- **Two modes exist:**
  - FACE_BASELINE, the bbox crop, which is the default and bit-identical to before [C];
  - FACE_ALIGNED, a 5-landmark similarity transform [C].
- **Geometric normalization improved:** mean eye-line angle 2.89° → 0.98°, landmark RMS to the template (p95)
  30.9 → 12.1 px, with 0 alignment failures in 6,141 images [A].
- **With a retrained aligned model:**
  - protected EER 8.67 % → 7.18 % (paired Δ −1.49 pp, 95 % CI [−1.90, −0.69]) [A];
  - FRR at 0.80: 81.2 % → 76.8 % [A];
  - no latency cost: 47.2 vs 46.7 ms [A].
- **With the deployed checkpoint, alignment is worse** (+1.46 pp EER); that combination is an ablation only [A].
- **Not deployed:** switching needs the new checkpoint plus re-enrollment (`evaluation/reports/FACE_ALIGNMENT_DECISION.md`).

## 3. Face model

- Deployed model: InceptionResnetV1 (VGGFace2) fine-tuned on LFW (62 identities), 512-D [D].
- Held-out protected EER 8.50 % and AUC 0.9641 [A]. Raw EER 7.12 % [A].
- The notebook's EER of 1.0 % is closed-set, measured on training identities, and cannot be verified [F].

## 4. Voice model

- ECAPA-TDNN trained from scratch, 192-D [D].
- VoxCeleb1 test set, 24 speakers: protected EER 2.32 %, AUC 0.9967 [A].
- At distance ≤ 0.75: FAR 0.26 %, FRR 23.8 % [A].

## 5. Fingerprint model

- ResNet50 + head, 512-D [D].
- SOCOFing, 900 samples, 404,550 pairs: EER 30.77 %, AUC 0.7644 [A], verified by re-extraction.
- The old 44.5 % / 0.579 figures belong to a superseded model (`docs/FINGERPRINT_DOCUMENTATION_AUDIT.md`).
- The threshold is Hamming ≥ 0.90 and is uncalibrated [D].

## 6. BioHash

- 256 bits, stored as 32 bytes [D].
- HKDF-SHA256 key per (application, user, modality, key_version) [D].
- The evaluation's transform was asserted bit-identical to production [C].

## 7. Calibration

**Synthetic fit** [B], labelled SYNTHETIC CALIBRATION, not accuracy:

| Modality | Embedding dim | R² | RMSE (Hamming) |
|---|---|---|---|
| Face | 512 | 0.99986 | 0.0014 |
| Voice | 192 | 0.99990 | 0.0012 |
| Fingerprint | 512 (was fitted at 256; corrected) | 0.99993 | 0.0010 |

**Real-embedding validation** [A]:

| Modality | RMSE, all pairs | RMSE, genuine pairs | Bias |
|---|---|---|---|
| Face | 0.100 | 0.072 | +0.006 |
| Voice | 0.093 | 0.046 | +0.004 |

## 8. Thresholds

- Face estimated cosine ≥ 0.80 and voice estimated distance ≤ 0.75 [D], with `THRESHOLD_SOURCE = TEACHER_REQUESTED_BASELINE`. They are not optimized.
- The baseline findings are verified and kept: face FRR 81.5 % and ALL_REQUIRED face+voice FRR 84.9 % [A].
- No new threshold is selected (`evaluation/reports/BASELINE_THRESHOLD_FINDINGS.md`).

## 9. Fusion

Chimeric users [A]:

| Policy | FAR | FRR |
|---|---|---|
| ALL_REQUIRED (default) | 0 | 84.9 % |
| WEIGHTED | 0 | 59.2 % |
| Face only | 0.003 % | 79.9 % |
| Voice only | 0.13 % | 24.4 % |

Score-level mean(face, voice): EER 1.67 %, AUC 0.9989 [A]. Fusion logic is unchanged.

## 10. Template revocation

- 4 sets: 1 ACTIVE, 3 STANDBY [D].
- Revoked template vs the new active one: Hamming 0.500 for face and 0.504 for voice, with 0 acceptances [A].
- Genuine similarity is unchanged after 3 revocations: face 0.761 → 0.766, voice 0.811 → 0.804 [A].
- Pool exhaustion returns HTTP 409 [B][C].

## 11. Security

- Stored: protected templates, key and set versions, audit scores [D].
- Not stored: raw images, audio, embeddings or landmarks [D][C].
- MASTER_SECRET has no default and is never logged [C].
- Unlinkability, D↔_sys at bin 1/64 [A]: face 0.018 (small, but above the null p95 of 0.013); voice and fingerprint
  cannot be distinguished from the null.
- Not claimed: irreversible, unhackable, or secure against all attacks.

## 12. Protected-template evaluation

Evaluated end to end on the deployed pipeline, on identical pairs. The cost of template protection [A]:

| Modality | EER, raw | EER, protected | Change |
|---|---|---|---|
| Face | 7.12 % | 8.50 % | +1.38 pp |
| Voice | 2.21 % | 2.32 % | +0.12 pp |

Raw scores are never reported as system performance.

## 13. Threshold evaluation

- Face 0.60–0.90 and voice 0.50–1.00, step 0.01 (`threshold_sensitivity.csv`) [A].
- Operating points are selected on an identity-disjoint development split and evaluated once on test.
  For face, FAR ≤ 1 % gives FAR 0.83 % and FRR 22.9 % on test [A].
- Voice splits have only 12 speakers each, so its operating points are unstable [A].

## 14. Latency

Hardware: i5-1235U, 8.3 GB RAM, Windows 11, Python 3.13.1, torch 2.14.0+cpu, no GPU. Warm means [A]:

| Stage | Time |
|---|---|
| Face pipeline | 69 ms |
| Voice pipeline | 74 ms |
| BioHash (face) | 44 ms |
| Hamming comparison | 0.006 ms |
| DB lookup | 2.8 ms |
| API face+voice (full request) | 778 ms; P99 1,025 ms |

Cold start: face 6.0 s, voice 3.4 s [A]. Aligned and baseline preprocessing take the same time [A].

## 15. Software tests

- Backend: **636 passed, 0 failed** [C], re-measured, not assumed. The earlier count was 563.
- Frontend: typecheck 0 errors, lint 0 errors and 25 existing warnings, build OK [C].
- New tests cover modality dimensions, face alignment, the end-to-end workflow and the real-user harness [C].
- Software tests are not biometric authentication trials.

## 16. Training reproducibility

- Recorded runs (config, log, environment, manifest) [A]:
  - face reproduction: COMPLETED;
  - `face_bbox_fullframe_v2` and `face_aligned_v2`: COMPLETED (`training/face/ALIGNMENT_RUNS.md`);
  - `face_aligned_v1`: COMPLETED but confounded.
- Voice reproduction was interrupted after 8 of 30 epochs [F].
- Historical per-epoch curves of the deployed face and fingerprint models: NOT AVAILABLE [F].

## 17. Known limitations

1. No real-user study yet. The protocol and harness are ready (`evaluation/REAL_USER_PROTOCOL.md`) [F].
2. The teacher thresholds give a very high face FRR [A].
3. Fusion is evaluated on chimeric users, assuming the modalities are independent [A].
4. The voice test set is small (24 speakers) [A].
5. The fingerprint model is weak, and its threshold is uncalibrated [A][D].
6. The aligned face model is not deployed. It is sensitive to landmark errors on extreme poses [A].
7. No presentation-attack detection, and no formal security proof [F].
8. Iris is not implemented [F].
9. Latency was measured on a single laptop CPU [A].
