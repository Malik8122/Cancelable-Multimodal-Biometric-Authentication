# Dynamic Hand Gesture Modality - the Z (`hand_gesture_v2`)

The third biometric modality, next to face and voice: a behavioral biometric based on temporal hand movement. The user
draws a handwritten-style **Z** in the air - top-left → top-right, diagonally down-left, then left → right - and *how the
hand moves through the Z* (palm trajectory, direction changes, rhythm, hand orientation) is compared with the user's
three enrolled samples.

**Status: implemented; calibrated on SYNTHETIC data only.** No real hand-gesture recordings exist in this repository
yet. The decision threshold was selected on a synthetic genuine / impostor dataset (§11) - that shows the software
separates the variation the generator models; it is **not** evidence of real-world FAR / FRR. Real consented recordings
are needed before any accuracy claim. Nothing here claims novelty: MediaPipe hand tracking, trajectory normalization and
DTW are established techniques.

> **2026-09-28: the infinity (figure-eight) gesture was removed** and replaced by the Z; enrollment went from five to
> three samples. Templates of the old representation (`hand_gesture_v1`, gesture `infinity`) cannot be compared with
> a Z: such users are reported as *not enrolled* for the hand (`backend/services/enrollment.py`) and must re-enroll it.

## 1. Why fingerprint was replaced

Iris and fingerprint were removed from the system on 2026-09-27 (commit `8e0a3b5`). The fingerprint path needed
uploaded scanner images, a separate trained model and a separate dataset, and the project's own validation recorded it
as weak with an uncalibrated threshold (dated record: `docs/FINAL_PROJECT_VALIDATION.md`, limitation 5). The replacement
modality uses only the camera the system already uses for face capture. This is a change of modality, not a claim that
the new one is more secure than fingerprint - that has not been measured.

## 2. Why a dynamic Z gesture

A static hand pose mostly captures hand geometry. A dynamic gesture adds *how* the person moves - which is a
behavioral signal. The Z was chosen over the infinity sign because it has three clearly separated strokes and two
sharp direction changes (easy to explain, easy to check for completeness, and the corners are where people's timing
differs), and it cannot be confused with an idle hand wobble. Natural variation is expected and normalized away: a
bigger or smaller Z, faster or slower, anywhere in the frame, slightly rotated, mild curvature, pauses at the corners.
Gesture types are an explicit registry (`preprocessing/hand_gesture.py::GESTURE_TYPES`), stored per template.

## 3. MediaPipe Hands (landmark detection)

- **Where:** in the browser, with the official `@mediapipe/tasks-vision` `HandLandmarker` (VIDEO mode, one hand), loaded
  from `src/hand/landmarker.ts`. The model (`hand_landmarker.task`, float16) is used as-is - it is not trained here.
- **Why the browser:** live feedback needs per-frame detection anyway; no video is uploaded (only landmark coordinates
  leave the browser); the backend needs no heavy vision dependency.
- **Runtime/model URLs:** WASM from jsdelivr (pinned to 1.0.1), model from Google's model storage; override with
  `VITE_MEDIAPIPE_WASM_URL` / `VITE_HAND_MODEL_URL` to self-host.
- **Trust:** the backend cannot tell MediaPipe landmarks from fabricated ones (as with any client-supplied sample); see §14.

## 4. The 21 landmarks and the wire format

Each video frame yields, for one hand, 21 landmarks `(x, y, z)`: `x`, `y` normalized to the frame width/height, `z` a
relative depth with the wrist as origin. Indices used: wrist 0, finger MCPs 5 / 9 / 13 / 17, fingertips 4 / 8 / 12 / 16 / 20.

One gesture sample is uploaded as JSON (`application/json`, never video):

```json
{"gesture_type": "z", "mirrored": true, "image_width": 640, "image_height": 480,
 "frames": [{"t": 0.0, "landmarks": [[x, y, z], ... 21], "handedness": "Right"}, {"t": 33.4, "landmarks": null}, ...],
 "client_metrics": {"mediapipe_ms_mean": 11.5, "mediapipe_ms_max": 20.1, "fps": 29.8}}
```

`landmarks: null` means no hand in that frame. `mirrored: true` (the default) says the user watched a mirrored selfie
preview while drawing: the raw camera image is flipped relative to what the user saw, so the backend flips `x` back
before checking the Z's left / right structure. Parsing rejects anything structurally wrong - including the removed
`"infinity"` gesture type - as `MALFORMED_CAPTURE`.

## 5. Capture validation (capture failure ≠ mismatch)

`preprocessing/hand_gesture.py::process_capture`, applied to every enrollment sample, by `POST /enroll/hand/check`,
and at authentication - the SAME function, so enrollment and authentication produce the same representation. Only the
span between the first and last frame with a hand is used.

| Code | Meaning | Maps to |
|---|---|---|
| `NO_HAND_DETECTED` | no frame contains a hand | `CAPTURE_ERROR` |
| `INSUFFICIENT_FRAMES` | fewer than 15 frames with a hand | `CAPTURE_ERROR` |
| `MALFORMED_CAPTURE` | unreadable / structurally invalid upload | `CAPTURE_ERROR` |
| `CAPTURE_QUALITY_FAILURE` | hand lost in > 40 % of the frames, tracked at < 10 fps, movement shorter than 0.5 s or longer than 10 s, hand too small; or (enrollment) one sample unlike both others (§9) | `QUALITY_INSUFFICIENT` |
| `INVALID_TRAJECTORY` | the palm moved < 1 palm size, or the movement is not a complete Z (§6.4) | `QUALITY_INSUFFICIENT` |
| `GESTURE_MISMATCH` | a valid Z that did not match | `VERIFICATION_MISMATCH` |

Tracking rate: 20 and 15 fps are good, 12 acceptable, 10 the minimum. Only a capture that passes every check is
compared; a failed check is "hand capture invalid", never "hand not matched". Capture failures never count toward the
suspicious-attempt mismatch streak and never trigger template rotation (`backend/services/failure_policy.py`).

## 6. Normalization pipeline (enrollment and authentication identical)

1. **User-view isotropic coordinates.** `x * width / height` so both axes share one physical unit; with `mirrored`,
   `x` is flipped so the user's left → right is `+x` (image `y` points down). Lengths and angles are unchanged.
2. **Gap filling + smoothing.** Frames where the hand was briefly lost are linearly interpolated in time. Then a
   **Gaussian kernel in time** (σ = 60 ms, on the real timestamps) averages out MediaPipe jitter. Because it is defined in
   milliseconds, not frames, the blur is the same at 12, 15 or 30 fps: an earlier 3-frame window smoothed a 30 fps capture
   but not a 15 fps one, and the same Z tracked at the two rates then differed by ~0.10 DTW (synthetic); now ~0.055.
3. **Idle trimming.** The gesture is the stretch from the first to the last frame where the palm centre moves faster
   than max(0.5 palm sizes/s, 15 % of the capture's 90th-percentile speed), plus 50 ms margin. The still hand before and
   after is removed; slow-downs and pauses at the Z's corners lie in between and are kept.
4. **Z-structure check (validity only).** The palm trajectory is resampled to 48 points by arc length; an exhaustive
   search over the two corner positions looks for three strokes whose chords point right (± 55°), down-left
   (90°-180°, y down) and right again, each ≥ 10 % of the chords, together explaining ≥ 55 % of the path length. This
   rejects an incomplete Z, a backwards or mirrored Z, an N, a circle, a straight swipe. It decides only *whether* the
   capture is a Z worth comparing - it is never used for identity. The tolerances are deliberately lenient (loosened
   after real use): sloppy, slanted, bowed, wobbly, rotated (~25°), tall or wide Z's pass.
5. **Temporal resampling.** Uniform in time to **64 points** (a Z takes ~1.5-3.5 s; at 10-30 fps that is 15-105
   frames). This removes the overall duration and the frame rate; DTW absorbs the remaining non-uniform speed changes.
6. **Translation and scale.** The palm-centre trajectory is centred on its own centroid (the Z's position in the frame
   does not matter) and divided by its RMS radius (a small and a large Z become comparable). Hand-shape features use
   the wrist as origin and the palm size as unit.

## 7. Representation: 13 features per point (`FEATURE_NAMES`)

The biometric is primarily **how the palm moves**, not where each finger is. The **palm centre** (mean of the wrist and
the four finger MCP joints) is the stable reference point: far less jittery than a fingertip and independent of finger
posture.

| Priority | Features | Dims | Weight |
|---|---|---|---|
| Primary | palm-centre trajectory (centroid-centred, RMS-radius units) | 2 | 1.0 |
| Primary | unit direction of palm motion (fades to 0 when nearly still, so a pause is not random noise) | 2 | 0.6 |
| Secondary | hand orientation: unit vector wrist → middle MCP | 2 | 0.3 |
| Temporal | relative speed (÷ this gesture's mean speed) | 1 | 0.3 |
| Temporal | relative acceleration (change of relative speed, scaled ×5, clipped ± 3) | 1 | 0.15 |
| Low | fingertip-to-wrist distance of the 5 fingers, in palm sizes | 5 | 0.1 |

Absolute duration and absolute size are deliberately **not** features (recorded as metadata only), so a genuine user who
draws faster, slower, bigger or smaller still matches. The weights are development defaults stored with each template.
Their effect was checked on the synthetic calibration set (§11): dropping the temporal / orientation / finger groups
(trajectory + direction only) raised the held-out EER from 0.2 % to 1.7 %.

## 8. Matching: DTW with a small rotation search (lower = better)

`template_protection/dtw.py`: Euclidean frame cost, Sakoe-Chiba band of 20 % of the length, accumulated cost divided by
the number of steps on the optimal warping path (the mean per-step frame distance). Verified: identical sequences give
0; the distance does not grow with the number of frames (path-length normalization; all sequences are 64 points anyway);
it is symmetric; the features are normalized once in preprocessing and the DTW normalizes only by path length (no
double normalization). The implementation is a plain-Python row loop (bit-identical to the textbook double loop,
~0.7 ms per 64 × 64 comparison).

**Rotation:** small camera / hand orientation differences are absorbed by comparing the live gesture also rotated by
0, ± 6° and ± 12° (`HAND_GESTURE_ROTATION_TOLERANCE_DEG`, `rotate_features`: trajectory, direction and orientation are
rotated; speeds and distances are invariant) and keeping the closest alignment per enrolled sample. The tolerance is
deliberately small: a 35° rotation is **not** aligned back (test `test_rotation_tolerance_is_bounded_not_full_invariance`) -
full rotation invariance would discard how the user orients the Z. On the synthetic set the search lowered the held-out
EER from 5.0 % to 0.2 %.

**Decision.** For the live sample A and the enrolled samples S1..S3:

```
d_i = min over r in {0, ±6°, ±12°} DTW(rotate(A, r), S_i)        i = 1, 2, 3
d   = median(d_1, d_2, d_3)                                         HAND_GESTURE_AGGREGATION
MATCH  <=>  d <= HAND_GESTURE_DTW_THRESHOLD (0.216)
```

Why the median: with three samples it is the second-closest one - the live Z must agree with two of the three samples,
which tolerates one poor sample without being decided by a single lucky alignment (`min`). On the synthetic data
min / median / mean were statistically tied (§11); the median was kept for that robustness argument, not because it
made any particular attempt pass. `mean` and `min` remain available for experiments.

For fusion the distance is mapped to the higher-is-better scale the other modalities use:
`s = clip(1 - d / (2T), -1, 1)` - `s = 0.5` at the threshold `T`. A transparent, uncalibrated mapping, not a probability.

## 9. Enrollment (three independent Z samples) and authentication (one)

- **Enrollment** (`POST /enroll/hand`, field `attempts` × 3; UI "Sample 01 / 03 ... 03 / 03"): three INDEPENDENT Z
  samples - separate, natural performances. The UI checks each immediately (`POST /enroll/hand/check`) and asks for a
  retake of just that sample; accepted samples are appended, never replaced. At submission every sample is validated
  again; the same captured sequence twice is refused (422). The three sequences are **not averaged** point by point (they
  are not time-aligned; an average would blur the corners) - all three are kept, separately.
- **Robust template:** the pairwise distances D(S1,S2), D(S1,S3), D(S2,S3) (same rotation search) give the **medoid**
  (the most representative sample, stored as metadata) and detect an **outlier**: a sample whose distance to the closer
  of the other two exceeds both the threshold and 2 × the distance between those two. Such a sample (incomplete,
  abnormal, someone else's Z) is refused as `CAPTURE_QUALITY_FAILURE` with its number and nothing is stored, instead of
  poisoning the template. Synthetic check: no genuine enrollment of 40 was flagged; a foreign sample planted among two
  genuine ones was flagged in 37 of 40 cases.
- **Authentication** (`POST /verify/hand`, field `gesture`, or `hand_gesture` in `POST /authenticate/fusion`): ONE Z
  sample → validation → features → DTW (rotation search) against the ACTIVE set's three samples → median → threshold.
  No sample counter is shown.

## 10. Template storage and keyed (cancelable) protection

- **Table:** `protected_templates`, one row per template set (T1 ACTIVE, T2-T4 STANDBY), revoked / rotated / activated
  together with face and voice. `template_format = hand_gesture_v2`, `gesture_type = z`, `template_metadata` (samples,
  per-sample frames / durations / tracking rate, feature dimension, weights, DTW band, rotation tolerance, pairwise
  distances, medoid - never features).
- **Protection:** enrollment = normalized weighted sequences → keyed orthonormal 13 × 13 matrix from the set's HKDF
  key (`template_protection/sequence_transform.py`) → stored. Authentication = normalized weighted live sequence →
  rotation variants (plaintext) → the SAME key's matrix → DTW. One transform per sequence, same key, dimensionality,
  feature order and normalization on both sides - no double transformation. Because the matrix is orthonormal and the
  DTW frame cost is Euclidean, every distance is identical to comparing the untransformed sequences (tests
  `test_keyed_transform_is_orthonormal_and_preserves_every_dtw_distance`, `test_keyed_matching_equals_plaintext_matching`).
- **Honest limitation:** unlike BioHash, this transform is **invertible by anyone who holds the key**; its protection
  rests on `MASTER_SECRET` staying secret. A non-invertible sequence protection is future work.
- New template sets carry the ACTIVE set's enrolled samples re-keyed under the new key (a retired-format template is
  refused - re-enroll instead).
- Never stored: video, frames, raw landmarks, untransformed features. Distances appear in API responses only with
  `DEBUG_SCORES=true`.

## 11. Threshold calibration

**Method** (`evaluation/hand_gesture_evaluation.py`, the shipped service end to end): every probe of every participant
is compared with every participant's enrollment under the target's key (genuine vs zero-effort impostors who draw the
same Z). Distances are aggregated with min / median / mean; the threshold is the EER threshold on a **DEV** half of the
participants, and FAR / FRR are reported at that threshold on the held-out **TEST** half. Captures that fail the gate
count as failures to acquire, never as scores.

**Synthetic calibration (the only data so far).** `tests/hand_signals.py` models each synthetic person's Z style
(proportions, slant, stroke bows, per-stroke speed profile and timing, corner pauses, hand orientation, finger geometry)
and, per execution, natural variation (1.5-3.1 s, Z size ×0.75-1.3, position, camera distance, ± 6° rotation,
proportion / timing jitter, landmark jitter). Probes per person: normal, slow 3.1 s, fast 1.5 s, 2 s of idle padding,
12 fps tracking, and moved + 20 % smaller + 10° rotated + 10 % dropped frames.

- *Design set* (people seeded 1000+, 40 people, used to choose the representation, rotation search and aggregation):
  held-out TEST half, median rule: EER 0.2 %, ROC-AUC 0.9999; min / median / mean dev-split EER 1.7 % / 2.4 % / 1.7 % (120 genuine dev probes: the difference is about one probe - a tie). Median dev-EER threshold **0.216**. Ablations (held-out EER, median): without the rotation search 5.0 %; trajectory + direction only 1.7 %. Genuine median distance per probe condition (mean): normal 0.122, slow 0.129, fast 0.134, idle padding 0.133, 12 fps 0.136, moved / scaled / rotated / dropouts 0.128.
- *Independent confirmation* (people seeded 3000+, never used for design; `python -m evaluation.hand_gesture_evaluation
  --synthetic 40`, results in `evaluation/results/hand_gesture_synthetic_*.csv`, label SYNTHETIC CALIBRATION):
  240 genuine and 9 360 impostor comparisons, no failures to acquire. Genuine distance mean 0.129 (SD 0.021, p95 0.165, max 0.215); impostor mean 0.370 (SD 0.085, p5 0.243, min 0.150). At the operating threshold 0.216: **FAR 1.35 % (126 / 9 360), FRR 0 % (0 / 240)**; pooled EER 0.42 %, ROC-AUC 0.9999. This set's own dev/test split gives a similar threshold (0.202; held-out FAR 0.53 %, FRR 0 %).

The threshold **0.216** (`HAND_GESTURE_THRESHOLD_SOURCE = DEVELOPMENT_DEFAULT`) is the median rule's dev-EER threshold on the design set, confirmed once on the independent set (an earlier 0.23, chosen before the smoothing change, gave FAR 2.9 % there and was discarded).
These numbers show the software separates the variation the generator models; synthetic separation is optimistic by
construction (real people vary in ways the generator does not model) and is **not** a real-world FAR / FRR.

**Real calibration (to do).** Collect consented genuine and impostor Z samples (Testing page → *Hand Gesture Data
Collection*; impostors draw their *own* natural Z), file them with `python -m evaluation.hand_gesture_evaluation --root
DATA --import-downloads DOWNLOADS`, run `python -m evaluation.hand_gesture_evaluation --root DATA --consent-confirmed`,
then set `HAND_GESTURE_DTW_THRESHOLD` from the dev split and `HAND_GESTURE_THRESHOLD_SOURCE=EXPERIMENTALLY_CALIBRATED`.
Do not pick a threshold from one person or one successful attempt.

## 11a. Tempo robustness - real-user diagnosis and representation experiment (2026-09-28, EXPERIMENTAL)

**Real observation (one user, diagnostic only).** Enrollment S1-S2 0.251, S1-S3 0.250, S2-S3 0.154; authentications
0.132 / 0.181 / 0.285 (median 0.181, MATCH) and 0.419 / 0.268 / 0.280 (median 0.280, MISMATCH at 0.216). S1 was drawn
slower (3.58 s movement vs 2.79 / 2.76 s); shape (aspect 0.67 / 0.70 / 0.62) and hand orientation (-102 / -101 / -98 deg)
were consistent; best rotations 0 / -6 deg. Speed + acceleration carried 30-43 % of the squared genuine DTW cost although
weighted 0.45 together (`python -m scripts.diagnose_hand_template --user ... --application-id ...`). Enrollment and
authentication preprocessing were verified identical. Conclusion: capture PASS, matching executed correctly, the
threshold failed - which does not by itself prove the threshold wrong; the synthetic threshold did not generalize to
this real user's tempo variation.

**Mechanism.** With uniform TIME resampling a Z drawn with a different rhythm puts the same spatial points at different
indices. Controlled check (same spatial Z, re-timed): trajectory difference 0.45 with time resampling vs 0.10 with
ARC-LENGTH resampling, where the rhythm difference moves into a separate timing feature (0.10). A purely uniform
slow-down is absorbed by both (0.02 / 0.03). Tests: `tests/test_hand_representation_experiment.py`.

**Experiment** (`evaluation/hand_representation_experiment.py`; production pipeline unchanged). Ten predefined
variants - A current, B without speed, C trajectory + direction, D trajectory only, E1 tempo-robust weights, E2
shape-dominant weights, F-I the same ideas with arc-length resampling plus a relative-timing feature (elapsed time /
total time along the path, i.e. relative stroke timing independent of the absolute duration). Same gate, rotation
search, DTW and median as production; development / held-out split by participant; per-variant EER threshold chosen on
development, reported once on held-out participants.

**Synthetic tempo-stress result** (30 synthetic users, 15 dev / 15 held-out; every execution at an independent pace
1.4-3.8 s with a non-uniformly changed speed profile, stroke timing and pauses, 12-30 fps, a second session with
rotation / size change; held-out: 150 genuine, 2 100 impostor comparisons; `--synthetic 30`,
`evaluation/results/hand_representation_synthetic_*.csv`):

| Variant | Genuine mean / p95 | Impostor mean / p5 | Held-out FAR / FRR (dev threshold) | EER | ROC-AUC | at 0.216: FAR / FRR |
|---|---|---|---|---|---|---|
| A current (time) | 0.181 / 0.236 | 0.407 / 0.265 | 3.5 % / 3.3 % (0.250) | 3.3 % | 0.9958 | 0.8 % / 14.7 % |
| B no speed | 0.156 / 0.207 | 0.352 / 0.230 | 3.0 % / 2.0 % (0.220) | 2.6 % | 0.9971 | 2.6 % / 2.7 % |
| C traj + direction | 0.132 / 0.177 | 0.297 / 0.180 | 6.2 % / 1.3 % | 4.7 % | 0.9910 | 17.5 % / 0 % |
| D trajectory only | 0.096 / 0.144 | 0.229 / 0.112 | 12.5 % / 9.3 % | 11.2 % | 0.9655 | 47.9 % / 0 % |
| E1 tempo-robust | 0.143 / 0.191 | 0.326 / 0.208 | 4.7 % / 1.3 % | 2.6 % | 0.9960 | 6.5 % / 0 % |
| E2 shape-dominant | 0.128 / 0.174 | 0.291 / 0.177 | 6.4 % / 2.0 % | 4.7 % | 0.9918 | 19.2 % / 0 % |
| F arc, current weights + timing | 0.142 / 0.197 | 0.319 / 0.208 | 4.0 % / 3.3 % | 3.4 % | 0.9945 | 6.9 % / 1.3 % |
| G arc, shape + timing | 0.111 / 0.152 | 0.248 / 0.158 | 2.3 % / 10.7 % | 4.1 % | 0.9919 | 33.4 % / 0 % |
| H arc, traj + direction | 0.108 / 0.148 | 0.231 / 0.138 | 3.9 % / 17.3 % | 7.5 % | 0.9801 | 44.1 % / 0 % |
| I arc, tempo-robust + timing | 0.115 / 0.158 | 0.256 / 0.165 | 3.1 % / 7.3 % | 4.0 % | 0.9929 | 29.0 % / 0 % |

Reading: (1) the tempo-stress data reproduces the real symptom - the current representation at 0.216 rejects 14.7 % of
genuine attempts; (2) every tempo-robust variant lowers genuine distances, but mostly by shrinking the WHOLE distance
scale - impostor distances fall too, so "passing at 0.216" would be threshold fitting in disguise; (3) by separation
(EER / AUC) nothing clearly beats the current representation: B and E1 are marginally better (2.6 % vs 3.3 % EER, about
one genuine probe of 150), arc-length variants were not better, and removing timing / orientation (C, D, E2, G, H)
lost separation; (4) genuine distance barely depends on the duration relative to enrollment in any variant (A: FRR 6 /
3 / 0 % for faster / similar / slower than enrolled). Caveat: synthetic users differ from each other partly BY their
speed profile, which favours temporal features by construction. **Decision: none yet.** The production representation
is unchanged; candidates for the real development set are A (control), B, E1 and F. Select on real development data
by EER / AUC (not by lowest genuine distance), prefer the simplest adequate one, calibrate its threshold there, evaluate
once on held-out real participants.

**Real pilot required** (Testing page data collection): at least 5, preferably 10+ consented participants; each 3
enrollment Z's, 5-10 genuine Z's in session 1 and again in session 2 (another day), and 5 impostor Z's. Then
`python -m evaluation.hand_representation_experiment --root DATA --consent-confirmed` (variant comparison, dev / held-out)
and `python -m evaluation.hand_gesture_evaluation --root DATA --consent-confirmed` (production calibration).

## 12. Development diagnostics

With `DEBUG_SCORES=true`, the verification diagnostics show per Z capture: tracking rate, raw / valid frames, gesture
duration, idle trimmed, normalized duration (vs the enrolled median), palm path length and mean speed, the share of time
per Z stroke, trajectory points, Z validity, raw DTW vs sample 1 / 2 / 3 with the best rotation and the top feature
groups' share of each distance, the aggregated distance, threshold and PASS / FAIL; the enrollment check shows the capture-gate values and Z validity per sample. Never raw landmarks.
In normal mode the user sees only the Z guide, the real replay of their captured palm trajectory (drawn from the
MediaPipe data, never animated), and the outcome.

## 13. Fusion

`hand` is a third modality in the unchanged fusion engine (`fusion_similarity` = equal-weight mean of the fusion-scale
scores of the submitted modalities; policies `ALL_REQUIRED` (default), `WEIGHTED`, `AT_LEAST_TWO`). A capture error or
failed quality gate in any submitted modality denies the attempt without fusion (fail closed). Registration requires
face and voice; the hand gesture is an optional enrollment step.

## 14. Performance (backend)

`python -m scripts.benchmark_hand_gesture` (synthetic landmark input; timing depends on the sequence length, not on who
performed it): parse + validate ≈ 6 ms, feature extraction ≈ 8 ms, one DTW comparison ≈ 1.8 ms, full one-gesture authentication (5 rotations × 3 enrolled samples + keyed transform + DB) ≈ 31 ms mean / 35 ms p95 on this laptop CPU. MediaPipe runs in the browser; its per-frame inference time is sent with every capture
(`client_metrics`) and audited.

## 15. Limitations

- **No real data yet:** real FAR / FRR / EER are unknown; the threshold comes from synthetic users, and one real user
  already showed more tempo variation than the original synthetic model (§11a). Synthetic results are not real-world
  FAR / FRR. Real-camera
  validation with several people (genuine attempts over days, impostors, different gestures) is still required.
- The Z is a simple, public gesture: impostors can watch and imitate it. Only zero-effort impostors (drawing their own
  Z) are modelled; imitation attacks are not evaluated.
- Behavioral biometrics vary with fatigue, injury, practice and posture; left vs right hand, and camera placement,
  change the representation.
- No liveness / presentation-attack detection; the backend trusts client-supplied landmarks.
- 2D trajectory only (MediaPipe image landmarks have no global depth).
- The template protection is revocable but invertible with the key (§10).
- Users enrolled with the removed infinity gesture must re-enroll the hand.
