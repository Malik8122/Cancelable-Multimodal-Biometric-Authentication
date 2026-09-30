# Face alignment decision

## Result in one paragraph

- **What was compared.** Two face pipelines, each trained under identical conditions:
  - the bounding-box crop (the current default);
  - the new 5-landmark similarity alignment.
- **Result on identical held-out pairs.** Alignment lowers the protected-template EER from **8.67 % to 7.18 %**
  (paired Δ −1.49 pp, 95 % CI [−1.90, −0.69]). It raises TAR at FAR 0.1 % from 0.585 to 0.632 and cuts FRR at the
  teacher threshold from 81.2 % to 76.8 %. It makes the model markedly more robust to 30° rotation and to lighting
  changes, adds no latency and produced no alignment failures.
- **Where it does not help.** The gain shrinks under ALL_REQUIRED fusion, where the face threshold still dominates. It
  depends entirely on training the model on aligned faces: feeding aligned crops to the deployed checkpoint makes EER
  **worse** by 1.46 pp. Alignment also amplifies MTCNN landmark errors on extreme poses.
- **Decision.** Aligned + `face_aligned_v2` is the recommended production face pipeline. It is **not** switched on in
  this change; see §8 for why and how to switch.

## Systems and protocol

**Systems** (`evaluation/ieee/face_alignment_eval.py`):

| System | Checkpoint | Preprocessing | Role |
|---|---|---|---|
| A | deployed `models/face/saved/face_embedder.pt` | bbox | production today |
| B | deployed | aligned | **inference-time preprocessing ablation**; the checkpoint never saw aligned faces |
| C | reproduced bbox model (`training/face/runs/face-20260925T031117Z`) | bbox | v1 pair: **confounded**, see below |
| D | `face_aligned_v1` | aligned | v1 pair: confounded |
| **E** | `face_bbox_fullframe_v2` | bbox | **primary baseline** |
| **F** | `face_aligned_v2` | aligned | **primary aligned model** |

- **Why E vs F is controlled.** E and F were trained with identical settings: the same 3,023 LFW full-frame images
  (kept-set SHA-256 `ad6a0cb5…`), the same 2091/426/506 split, seed 42, the same model, loss, optimizer and 10 epochs.
  The only difference is the preprocessing, used for both training and inference.
- **Why v1 is confounded.** The v1 pair was trained on scikit-learn's 125×94 slice. That leaves the aligned training
  crops with about 39 % black border, compared with about 1 % at evaluation (`training/face/aligned_v1/README.md`).
  v1 is reported but not used for the decision.
- **Canonical template** (`preprocessing/face.py::ALIGNMENT_TEMPLATE_160`): the mean landmark position in baseline crops
  of the training split, made left-right symmetric (`evaluation/results/face_alignment_template.json`). This keeps the
  framing close to what the model learned.

**Evaluation protocol** (evidence **REAL DATA**):

- Held-out LFW identities with 2–19 images. They are disjoint from the 62 fine-tuning identities.
- 1,618 enrolled identities and 6,141 images; every image is valid under both preprocessors.
- One MTCNN detection per image produces both crops. The code was checked against the real preprocessors on 40 images.
- The first image of each identity is the reference and its other images are genuine probes (4,523). Impostor probes
  are 50 per identity (80,900), each templated under the target's key, seed 20260925.
- **Every system is scored on identical pairs.**
- Protected score = the unchanged 256-bit BioHash, whose Hamming similarity gives the calibrated estimated cosine.
- Reproduce: `python -m evaluation.scripts.evaluate_face_alignment`.

## 1. Did alignment improve geometric normalization? **Yes.**

`face_alignment_quality.csv` and `.png`, over 6,141 images. Positions are in the 160×160 face crop.

| Measure | Baseline bbox crop: mean / SD / p95 | Aligned: mean / SD / p95 |
|---|---|---|
| Eye-line angle (absolute) | 2.89° / 2.67 / 7.68 | **0.98° / 0.96 / 2.77** |
| Eye-midpoint offset from centre | 22.4 / **6.09** / 33.3 px | 21.2 / **1.56** / 23.5 px |
| Mouth-midpoint y | 120.7 / 4.20 / 127.2 | 125.1 / **2.41** / 128.2 |
| Inter-eye distance | 71.8 / 4.22 | 60.8 / 3.98 |
| Landmark RMS to template | 14.6 / 8.51 / 30.9 px | **8.08 / 2.50 / 12.1 px** |

- The spread of landmark positions, which is what normalization removes, falls three- to four-fold. The p95 of the
  angle and of the RMS falls about 2.5×.
- The residual RMS of about 8 px, and the smaller eye distance (60.8 vs the template's 72.8), are expected. The
  template comes from bbox crops, which are stretched unevenly (a non-square box resized to 160×160), and a
  uniform-scale similarity cannot reproduce that stretch.

## 2. Did alignment improve raw-embedding performance? **Yes, when the model is trained for it.**

Raw embedding cosine, `face_alignment_metrics.csv`. The raw cosine is not what the deployed system computes.

| System | EER (95 % CI) | ROC-AUC | TAR at FAR 0.1 % |
|---|---|---|---|
| E bbox v2 | 6.99 % (6.47–7.41) | 0.9697 | 0.681 |
| **F aligned v2** | **5.77 %** (5.32–6.39) | **0.9735** | **0.752** |
| A deployed | 7.12 % | 0.9682 | 0.669 |
| B deployed + aligned (ablation) | 8.25 % | 0.9647 | 0.601 |

## 3. Did alignment improve protected-template performance? **Yes.**

Deployed pipeline, 256-bit templates:

| System | EER (95 % CI) | ROC-AUC | TAR at FAR 0.1 % | Genuine / impostor mean estimated cosine |
|---|---|---|---|---|
| E bbox v2 | 8.67 % (7.87–9.14) | 0.9630 | 0.585 | 0.630 / 0.052 |
| **F aligned v2** | **7.18 %** (6.83–7.88) | **0.9674** | **0.632** | **0.654** / 0.052 |
| A deployed | 8.50 % | 0.9641 | 0.534 | 0.630 / 0.059 |
| B ablation | 9.96 % | 0.9542 | 0.461 | 0.609 / 0.067 |

**Paired bootstrap** (`face_alignment_paired_bootstrap.csv`: 300 resamples of enrolled identities, identical pairs):

| Contrast | ΔEER | 95 % CI | Resamples showing no improvement |
|---|---|---|---|
| **F − E** | **−1.49 pp** | [−1.90, −0.69] | 0 of 300 |
| F − A | −1.31 pp | [−1.77, −0.62] | 0 of 300 |
| B − A (ablation) | **+1.46 pp** (worse) | [+0.91, +2.24] | — |
| D − C (confounded v1) | +6.31 pp (worse) | [+5.47, +6.98] | — |

Template protection costs about the same with either preprocessing: +1.68 pp EER for E and +1.41 pp for F.

## 4. Did alignment affect FAR / FRR / EER? **Lower FRR at equal or lower FAR.**

`face_threshold_comparison.csv` and `.png`; protected estimated cosine; the threshold is not changed.

| Threshold | E: FAR / FRR | F: FAR / FRR |
|---|---|---|
| 0.60 | 0.215 % / 33.0 % | 0.234 % / **28.6 %** |
| 0.70 | 0.027 % / 55.7 % | 0.028 % / **50.9 %** |
| **0.80 (teacher baseline)** | 0 % / 81.2 % | 0.0012 % / **76.8 %** |
| 0.90 | 0 / 97.6 % | 0 / 96.5 % |

FRR at the teacher threshold stays very high under both pipelines. Alignment reduces it by 4.4 pp; it does not fix it
(`BASELINE_THRESHOLD_FINDINGS.md`).

## 5. Did alignment affect fusion? **Slightly, in favour of alignment.**

`face_alignment_fusion.csv`; chimeric face+voice users, 20 pairings; voice unchanged.

| Policy (FRR; FAR is ≈ 0 for all) | E | F |
|---|---|---|
| Face only | 79.4 % | **76.8 %** |
| Face + voice ALL_REQUIRED | 83.3 % | **82.9 %** |
| Face + voice WEIGHTED | 59.4 % | **55.1 %** |
| Score-level mean(face, voice), EER | 1.26 % | 1.31 % (no gain; voice dominates) |

## 6. Did alignment affect latency? **No measurable difference.**

`face_alignment_latency.csv`. Hardware: i5-1235U, 8.3 GB RAM, Windows 11, Python 3.13.1, torch 2.14.0+cpu, no GPU.
The two pipelines ran interleaved.

| Preprocessing | Warm, 100 runs: mean / median / P95 / P99 | Cold, 10 runs: mean |
|---|---|---|
| bbox | 47.2 / 45.1 / 63.8 / 80.3 ms | 4.67 s |
| aligned | 46.7 / 44.3 / 63.2 / 73.2 ms | 4.55 s |

The warp takes well under a millisecond next to MTCNN detection.

## 7. Did alignment increase failure or rejection cases? **No failures on LFW, but it is sensitive to landmark errors.**

- ALIGNMENT_FAILED: 0 of 6,141 held-out images, and 0 of 3,023 training images. No face detected: 0 in both
  pipelines. The enrollment quality gates are unchanged and still run on the geometry of the original capture.
- **Robustness** (`face_alignment_robustness.csv`): 200 genuine + 200 impostor pairs per condition, so each EER is
  only ±2–3 pp.

  | Condition | E: EER / FRR at 0.80 | F: EER / FRR at 0.80 |
  |---|---|---|
  | clean | 8.0 % / 80.5 % | 7.25 % / 78.5 % |
  | rotation 15° | 6.5 % / 82.5 % | 7.5 % / 77.0 % |
  | **rotation 30°** | **13.0 % / 95.5 %** | **6.75 % / 76.0 %** |
  | dark (γ 2.2) | 13.0 % / 94.5 % | 10.0 % / 84.0 % |
  | bright (γ 0.45) | 8.0 % / 85.5 % | 6.5 % / 77.5 % |
  | scale 0.5 | 7.25 % / 82.5 % | 7.75 % / 77.0 % |
  | scale 0.25 | 14.25 % / 93.5 % | 11.5 % / 95.0 % |
  | blur σ = 2 | 10.0 % / 92.5 % | 11.0 % / 90.5 % |

  Failure to acquire was 0 in every condition. Alignment is clearly better under 30° roll: the genuine mean holds at
  0.657, while bbox drops to 0.517. It is also better under both lighting changes. It is marginally worse, within the
  noise, at 15° rotation, scale 0.5 and blur.
- **Failure mode:** when MTCNN misplaces the landmarks on a strongly turned head, the aligned crop is rotated wrongly.
  An example is `evaluation/figures/face_alignment_examples/largest_roll.png`, where the eye landmarks land on the nose
  and cheek. The bbox crop degrades more gently in that case. The enrollment gates (roll ≤ 20°, yaw ratio ≤ 0.20)
  reject such captures at enrollment, but authentication runs no quality gate in either mode.

## 8. Which pipeline should be the final production pipeline?

**FACE_ALIGNED (`FACE_ALIGNMENT=similarity`) together with the `face_aligned_v2` checkpoint.**

- **Never** combine FACE_ALIGNED with the currently deployed checkpoint. That is system B, which is worse than today.
- **The default is not switched in this change** (it stays `FACE_ALIGNMENT=bbox`), for three reasons:
  1. The `face_aligned_v2` checkpoint is a CPU reproduction run. It is not yet installed at
     `models/face/saved/face_embedder.pt`; that step needs a Git LFS commit that replaces the deployed model.
  2. Switching the face model or the preprocessing makes every existing face enrollment unmatchable. All face users
     must re-enroll. The templates are revocable, so this is possible, but it is an operational change that someone
     must decide on and announce, not something to do silently.
  3. The evidence comes from LFW only. The gain should be confirmed with real users under
     `evaluation/REAL_USER_PROTOCOL.md` before deployment.
- **To switch:**
  1. Copy `training/face/aligned_v2/runs/face_aligned_v2/checkpoints/face_embedder.pt` to a new path, and set
     `FACE_MODEL_PATH` to it and `FACE_ALIGNMENT=similarity` (or `FACE_ALIGNED`).
  2. Clear or re-enroll the face templates.
  3. Re-run `evaluation/ieee/threshold_analysis.py` with the new model before revisiting the threshold.
     `THRESHOLD_SOURCE` stays TEACHER_REQUESTED_BASELINE unless a threshold is selected on a validation split.

## 9. What evidence supports that decision?

| Criterion | Better | Size | Evidence |
|---|---|---|---|
| Geometric normalization | aligned | landmark spread down 3–4× | `face_alignment_quality.csv` |
| Protected EER | aligned | −1.49 pp, CI excludes 0 | `face_alignment_paired_bootstrap.csv` |
| TAR at FAR 0.1 % | aligned | +0.047 | `face_alignment_metrics.csv` |
| FRR at teacher threshold | aligned | −4.4 pp (still 77 %) | `face_threshold_comparison.csv` |
| ALL_REQUIRED fusion FRR | aligned | −0.4 pp (small) | `face_alignment_fusion.csv` |
| Score-level fusion EER | tie | +0.05 pp | same |
| Robustness to roll and lighting | aligned | 30° roll EER 13.0 → 6.75 % | `face_alignment_robustness.csv` (small n) |
| Robustness to blur and mild rotation | bbox (marginal) | ≤ 1 pp, within noise | same |
| Latency | tie | < 1 ms | `face_alignment_latency.csv` |
| Failures | tie | 0 / 6,141 | `face_alignment_quality.csv` |
| Works with the deployed checkpoint | bbox | aligned + old checkpoint is +1.46 pp worse | ablation B |
| Operational cost | bbox | aligned needs a new checkpoint and re-enrollment | — |

The decision rests on several criteria that point the same way (EER, TAR at low FAR, FRR, robustness), with no
latency or failure cost. It does not rest on any single metric. The costs are named: a retrained checkpoint,
re-enrollment, sensitivity to landmark errors, and LFW-only evidence.

## Limitations

- LFW is web imagery, not webcam enrollment captures.
- One training seed per pipeline. The paired CI covers sampling of test identities, not training randomness.
- The robustness conditions are synthetic transforms on 400 pairs each.
- The v2 checkpoints were trained on CPU with the reproduced recipe, not the original Kaggle run.
- The template is fitted to bbox-crop framing, and a template tuned for aligned framing was not explored.
- Authentication has no quality gate in either mode.

## Privacy

- Landmarks are used transiently inside `preprocessing/face.py`. They are never logged, stored or returned by the
  backend, and no raw image is stored.
- The evaluation cache (`data/eval_cache/face_alignment_embeddings.npz`, git-ignored) holds embeddings and landmarks of
  **public LFW** images, for research only.
- The example figures show public LFW images only, never participants. They match the repository's `*.png` rule in
  `.gitignore`, so they are not committed.
