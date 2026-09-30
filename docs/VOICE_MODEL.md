# Voice Model

Voice is the second biometric modality, alongside Face. It follows the exact same design philosophy — modality-specific
preprocessing (`preprocessing/voice.py`) feeding a modality-specific
embedding model (`models/voice/`) behind the one shared
`BaseEmbedder.extract_embedding()` interface — with one interface nuance
explained in detail below.

## Architecture: ECAPA-TDNN

[ECAPA-TDNN](https://arxiv.org/abs/2005.07143) (Emphasized Channel Attention,
Propagation and Aggregation in TDNN, Desplanques et al., 2020) is the current
state-of-the-art architecture for speaker verification: a time-delay neural
network (TDNN) backbone with Squeeze-Excitation channel attention and
multi-layer feature aggregation, producing a fixed-length "speaker
embedding" from variable-length speech. This project uses SpeechBrain's
`speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN` implementation
(`models/voice/model.py::VoiceEmbeddingNet`) rather than the full
`speechbrain.inference.speaker.EncoderClassifier` pipeline — the lower-level
module takes precomputed features directly, so `preprocessing/voice.py`
(not SpeechBrain's own internal feature extractor) owns log-mel extraction,
matching how `preprocessing/face.py` owns its modality's classical
preprocessing rather than delegating it into the model.

Fine-tuning uses the same ArcFace (additive angular margin) loss every other
modality uses — `models/common/arcface.py::ArcMarginProduct`, wrapped by
`models/voice/losses.py::build_arcface_head` rather than reimplemented, so
every modality trains with one reviewed, consistent loss.

## Integration: why `extract_embedding` takes an array, not a path

An early draft of this module's spec called for
`VoiceEmbedder.extract_embedding(audio_path)`. That can't hold literally:
`models/common/base_embedder.py::BaseEmbedder.extract_embedding` is a
**concrete** method — every modality only implements `_load_checkpoint` and
`_extract_embedding_impl`, never `extract_embedding` itself — that validates
`image.ndim == 3 and image.shape[2] == 3` before dispatching. This is
deliberate: it's the one contract that lets `embeddings/pipelines.py`, and
later fusion/backend code, treat every modality uniformly without knowing
anything about MTCNN or log-mel filterbanks.

The resolution generalizes "image":
`preprocessing/voice.py::VoicePreprocessor.preprocess()` returns an 80-mel-bin
× time-frame log-mel spectrogram — a bare 2D array.
`embeddings/pipelines.py::ModalityPipeline`'s existing
`if processed.ndim == 2: stack to 3 channels` fallback then turns it into a
pseudo-"RGB" `(n_mels, n_frames, 3)` array purely to satisfy `BaseEmbedder`'s
shape contract — no `base_embedder.py` change, no special-casing anywhere
else. `VoiceEmbedder._extract_embedding_impl` takes channel 0 back out
(lossless, since all 3 channels are identical) before feeding it to
ECAPA-TDNN.

The spec's actual functional ask — a path-based `extract_embedding`,
`load_model`, `compare_embeddings` — is implemented as **free functions in
`models/voice/inference.py`**, built on top of the array-based contract
rather than replacing it:

```python
from models.voice.inference import compare_embeddings
similarity = compare_embeddings("speaker_a.wav", "speaker_b.wav")
```

`embeddings/pipelines.py::VoicePipeline` is the other integration point —
like `FacePipeline`, but its `embed()`
is overridden to also accept a `sample_rate` argument, since a raw waveform
alone (unlike a raw image) doesn't say what sample rate it was captured at.

## Dataset: VoxCeleb1 (subset)

Trained against [gaurav41/voxceleb1-audio-wav-files-for-india-celebrity](https://www.kaggle.com/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity)
on Kaggle — a real-audio subset of VoxCeleb1 (Indian-celebrity speakers), not
the full ~1,251-speaker corpus (impractical to download unattended inside a
single Kaggle kernel session — see `docs/DATASETS.md`'s Voice section for
the full licensing/scope note). Number of speakers, utterance counts, and
the resulting train/val/test split are **not hardcoded** — they're printed
at run time by `kaggle_kernels/voice_training/voice-embedding-training.ipynb`'s
dataset cell from whatever the attached copy actually contains. From the
real Kaggle GPU-kernel run that produced the committed checkpoint:

| Metric | Value |
|---|---|
| Speakers | 24 |
| Total utterances | 4,857 |
| Train / Val / Test | 3,389 / 717 / 751 (70% / 15% / 15% per speaker) |

## Real training results

From the same run (`kaggle_kernels/voice_training/`, CPU — see "A note on GPU
compatibility" below), evaluated on the 751-utterance held-out test set via
`evaluation/voice_metrics.py::run_voice_experiment`:

| Metric | Value |
|---|---|
| EER | 2.29% |
| Accuracy (at EER threshold) | 97.71% |
| AUC | 0.9967 |
| Precision | 67.57% |
| Recall | 97.71% |
| F1 | 79.89% |
| Genuine-pair cosine similarity (sample) | 0.895 |
| Impostor-pair cosine similarity (sample) | 0.132 |

Precision is lower than the other metrics because this evaluation counts
every genuine/impostor *pair* among the 751 test utterances (13,101 genuine
pairs vs. 268,524 impostor pairs — impostor pairs vastly outnumber genuine
ones for a 24-speaker test set), not because the model performs poorly; EER
and AUC are the metrics that matter for verification quality and both are
strong for a from-scratch fine-tune on a 24-speaker subset.

### A note on GPU compatibility

The Kaggle free-tier GPU (Tesla P100, compute capability 6.0) available for
this run was **not supported** by the preinstalled PyTorch build (which only
supports compute capability 7.0+) — `models/voice/utils.py::detect_device()`
smoke-tests CUDA with a real op and falls back to CPU rather than crashing
mid-training when this happens, which is what training actually ran on.
Training completed in full (all 30 epochs) in a little under 8 hours on CPU.

**Split methodology:** per-identity 70/15/15 (`models/voice/dataset.py::VoxCelebDataset`),
rather than VoxCeleb1's official fixed trial-list verification
protocol — a deliberate, documented scope reduction, not a silent
gap.

## Preprocessing pipeline

```text
raw waveform (any sample rate, mono or stereo)
        |
Mono conversion (average channels)
        |
Resample to 16 kHz (scipy.signal.resample_poly)
        |
Voice activity detection (energy-threshold, trims near-silent frames)
        |
Loudness normalization (scale to a target RMS level)
        |
Fixed-length segment extraction (4s: random crop when training, center crop
                                  at inference; zero-padded if shorter)
        |
80-bin log-mel filterbank (manual triangular filterbank + scipy.signal.stft,
                            mean-normalized)
        |
(n_mels, n_frames) float32 array  -> embeddings/pipelines.py stacks to
                                      pseudo-RGB -> BaseEmbedder contract
```

**Zero hard dependencies for preprocessing/testing.** `preprocessing/voice.py`
uses only `numpy` + `scipy` (`scipy.io.wavfile`, `scipy.signal`) — no
`torchaudio`, `speechbrain`, or `webrtcvad` required just to preprocess audio
or run this project's offline tests. Those three are genuinely needed (and
listed in `requirements.txt`) for the real ECAPA-TDNN backbone and Kaggle/
Colab training, imported lazily inside `models/voice/model.py`/`inference.py`
exactly like `models/face/inference.py` defers `facenet_pytorch` — so
`tests/test_voice_preprocessing.py`, `test_voice_dataset.py`, and
`test_voice_embedding.py` all pass with zero optional dependencies installed,
same as Face's mock-mode tests. VAD defaults to a dependency-free
energy-threshold implementation (`VoiceConfig.vad_backend = "energy"`);
`webrtcvad` is opt-in, not required, since it needs a compiled C extension
that isn't reliably installable on every dev machine.

Augmentation (Gaussian noise, speed perturbation, time/frequency masking,
random gain — configured via `VoiceConfig`'s `augmentation_*` fields) is
training-only and never applied by `VoicePreprocessor.preprocess()` at
inference (`training=False`).

## Capture robustness: quality gate and two-sentence voice (REAL DATA + SOFTWARE VALIDATION)

Evaluation: `python -m evaluation.ieee.voice_capture_pipeline` -> `evaluation/results/voice_capture_pipeline.csv`. It is
a NEW, separately labelled result; the published voice numbers (`voice_metrics.csv`, `protected_vs_raw.csv`,
`robustness.csv`) are unchanged. Data: the same VoxCeleb1 test split (24 speakers, 751 clips, closed-set), protected
domain as deployed, decision rule `estimated distance <= 0.75` (CONFIGURATION, teacher-requested, not tuned).

### Why genuine voice attempts fail (measured)

1. **The configured threshold is strict even on clean audio:** single-utterance FRR at 0.75 is 24% (FAR 0.26%).
2. **Noise:** FRR at 0.75 rises to 68.5% with white noise at 20 dB SNR and 90% at 10 dB (robustness.csv), although the
   EER at 20 dB is only 10% - noise pushes genuine scores below the fixed threshold.
3. **Short speech:** probes cut to their first 4 / 3 / 2 / 1.5 / 1 s give FRR 29.5 / 48 / 66.5 / 78.5 / 90.5%.
4. **One enrollment recording:** the template came from the first of the two recordings only.
5. **Browser audio processing:** `getUserMedia({audio: true})` enables echo cancellation, noise suppression and automatic
   gain control, which alter each recording differently; the model was trained on unprocessed audio. These are now
   switched off explicitly (`frontend/src/config/voice.ts`). The effect on real microphones is **not measured**
   (no re-recorded data) - FUTURE WORK.

### Threshold change to 0.90 and quality gate to 10 dB (2026-09-30)

**Noise experiment** (`python -m evaluation.ieee.voice_noise_frontend`, REAL DATA, VoxCeleb1 test split, 200 genuine
+ 200 impostor single-sentence probes, clean enrollment, noise on the probe only; `evaluation/results/voice_noise_frontend.csv`).
FRR / FAR at 0.75 per signal front-end applied to enrollment and probe alike:

| Probe condition | none | high-pass 80 Hz | + spectral subtraction | MetricGAN+ enhancement |
|---|---|---|---|---|
| clean | 24.0 / 0 % | 26.5 / 0 % | 33.5 / 0 % | 44.5 / 2.0 % |
| white noise 20 dB | 71.5 / 0 % | 71.0 / 0 % | 75.5 / 0 % | 66.0 / 5.0 % |
| white noise 10 dB | 89.0 / 0 % | 89.0 / 0 % | 92.0 / 0.5 % | 79.0 / 7.0 % |
| hum + rumble 10 dB | 33.5 / 0 % | 32.0 / 0 % | 41.0 / 0 % | 45.0 / 1.0 % |
| pink noise 15 dB | 60.5 / 0 % | 62.0 / 0 % | 66.0 / 0 % | 62.5 / 5.5 % |
| babble 15 dB | 28.0 / 0 % | 30.5 / 0 % | 40.0 / 0 % | 46.5 / 4.0 % |

No front-end helps: high-pass is neutral, spectral subtraction and neural enhancement remove speaker information along
with the noise (MetricGAN+ raises EER from 3 % to 16.5 % on clean audio). None was adopted. What the data does show is
that noise pushes GENUINE distances just past 0.75 while impostors stay far away (closest impostor 0.753 clean, >= 0.79
in every noise condition). Same probes, no front-end, FRR / FAR by threshold:

| Condition | 0.75 | 0.85 | **0.90** | 0.95 |
|---|---|---|---|---|
| clean | 24.0 / 0 % | 13.5 / 0.5 % | **9.5 / 0.5 %** | 6.0 / 0.5 % |
| white 20 dB | 71.5 / 0 % | 55.0 / 0.5 % | **45.5 / 1.0 %** | 35.5 / 1.5 % |
| hum 10 dB | 33.5 / 0 % | 17.0 / 0 % | **13.0 / 0 %** | 10.0 / 0.5 % |
| pink 15 dB | 60.5 / 0 % | 47.0 / 0 % | **34.0 / 0 %** | 26.0 / 0 % |
| babble 15 dB | 28.0 / 0 % | 17.0 / 0.5 % | **11.5 / 0.5 %** | 7.0 / 1.5 % |

**Decision:** `voice_euclidean_threshold` 0.75 -> **0.90** (full-set sweep at 0.90: FRR 8.3 %, FAR 0.96 %,
`threshold_sensitivity.csv`), together with two-sentence verification (below). The value was chosen from these sweeps,
not on a held-out validation split, so it is not labelled EXPERIMENTALLY_SELECTED; FAR resolution with 200 impostor
probes is 0.5 %. Voice is combined with face under ALL_REQUIRED. `voice_min_estimated_snr_db` 13 -> **10 dB**: live
recordings at normal speaking volume in a room measured 9-12 dB and were refused; at 10 dB clean clips are still never
rejected, while clips with 5 dB white noise are rejected 10 % of the time instead of 99.3 % (they are then compared
and usually mismatch rather than being asked to re-record). Neither change has been measured on live recordings yet.

### Protocol change (2026-09-30): two-sentence verification restored (2 / 2)

The frontend again records TWO sentences at authentication and sends them as `voice_audio` + `voice_audio_2`; the
backend gates and embeds each and combines them with the enrollment rule (`BaseModalityService.embed_recordings`), then
compares once. `voice_audio` alone is still accepted (2 / 1) for other clients. Reason: live testing with a genuine,
freshly re-enrolled user at 0.75 gave distances 0.70-1.13 (1 of 9 accepted) with one sentence; the 2 / 2 row below is
the measured operating point with the lowest FRR and FAR at the unchanged threshold. Not yet re-measured on live
recordings.

### Protocol change (2026-09-27, superseded 2026-09-30): two-sentence enrollment, ONE-sentence verification

Voice authentication now records exactly one sentence (`POST /authenticate/fusion` refuses `voice_audio_2`); enrollment
still records two. By the measurements below this moves the deployed operating point from the **2 / 2** row to the
**2 / 1** row: on the same dataset clips at the unchanged 0.75 threshold, FRR rises from 2.6% to 12.2% and FAR from
0.013% to 0.03%. The threshold was NOT re-tuned for the new protocol and the effect has not been measured on live
recordings - re-evaluate before relying on it. The section below documents the earlier two-sentence verification.

### Two-sentence enrollment and verification (earlier protocol; enrollment part still deployed)

ECAPA-TDNN maps ONE utterance (a centred 4 s log-mel crop) to one 192-D embedding. Each sentence is therefore embedded
separately with the unchanged pipeline and the two unit embeddings are combined as
`e = normalize((e1 + e2) / 2)` (`embeddings/centroid.py`, the rule face enrollment already uses). The result is still a
192-D unit vector, so BioHash, HKDF, calibration and the threshold are unchanged, and one template is stored per set.
Score-level combination of two separate protected comparisons was rejected: it would need two templates per set and a
new decision rule on top of the calibrated threshold.

| Enrollment / probe (same speakers and utterances) | FRR at 0.75 | FAR at 0.75 | EER |
|---|---|---|---|
| 1 utterance / 1 utterance (published protocol, reproduced) | 24.1% | 0.26% | 2.31% |
| 1 / 2 | 12.8% | 0.30% | 1.69% |
| **2 / 1 (deployed from 2026-09-27)** | **12.2%** | **0.03%** | **1.27%** |
| 2 / 2 (deployed until 2026-09-27) | 2.6% | 0.013% | 0.52% |

Caveats: clean dataset clips of >= 4 s; two utterances provide roughly twice the speech; the speakers are shared with
training (closed-set); the two utterances usually come from the same video session. This is a real improvement at the
same threshold, but it has not been measured on live recordings.

Existing voice enrollments keep their first-recording templates and still verify (same embedding space); re-enrolling
gives the two-sentence centroid.

### Capture-quality gate (the change that labels failures correctly)

`backend/services/voice_quality.py`, applied to every recording at enrollment AND authentication before embedding
(same measurements, `VoicePreprocessor.quality_metrics`). A failing recording is a capture-quality failure - the user
is asked to re-record that sentence; it is never compared, never counted as a mismatch, never a reason to rotate.

| Check | Default | Evidence |
|---|---|---|
| no samples / non-finite | INVALID_AUDIO | - |
| digital silence / no voiced frame | NO_SPEECH | - |
| > 1% of samples at full scale | CLIPPED | conventional engineering value (no clipped clips in the dataset) |
| < 1.5 s of speech after silence trimming | TOO_LITTLE_SPEECH | FRR 78.5% at 1.5 s, 90.5% at 1 s |
| estimated SNR < 13 dB | TOO_NOISY | rejects 0% of clean test clips, 3.7% at 10 dB SNR, 99.3% at 5 dB SNR |

**What the gate does not do:** it does not make accepted recordings verify better - FRR of the accepted recordings is
unchanged at every tested threshold (e.g. about 68% at 20 dB SNR). The frame-level SNR estimate cannot tell which noisy
recordings will still verify; it reliably catches only severe noise. A 15 dB gate would also catch 30% of 10 dB
recordings at the cost of rejecting 1.5% of clean clips. Report it as a capture/usability improvement, not as better
biometric discrimination.

### Latency (SOFTWARE VALIDATION, CPU, median of 30 runs)

| | 1 sentence | 2 sentences |
|---|---|---|
| quality metrics | 28 ms | 47 ms |
| embedding extraction | 130 ms | 212 ms |
| full voice decision (gate + embed + BioHash + comparison) | 193 ms | 293 ms |

The two-sentence enrollment consistency check keeps its bands (>= 0.75 enrolled, 0.60-0.74 warning, < 0.60 rejected).
Measured on two DIFFERENT utterances of the same speaker (REAL DATA): 8.1% of genuine pairs fall below 0.60 and 29.4%
below 0.75; 0.23% of different-speaker pairs reach 0.60.

## Training configuration

All defaults live in `models/voice/config.py::VoiceConfig` — one dataclass,
not scattered constants, since Voice has enough independently-tunable knobs
to warrant it (unlike Face's handful of module-level
constants).

| Parameter | Default |
|---|---|
| Sample rate | 16,000 Hz |
| Clip length | 4 seconds |
| Mel bins | 80 |
| Embedding size | 192 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Batch size | 64 |
| Epochs | 30 |
| Mixed precision | Enabled (CUDA only) |
| Early stopping patience | 5 epochs |
| ArcFace margin / scale | 0.50 / 30.0 |

## Embedding dimension

**192-dimensional**, L2-normalized — matches SpeechBrain's standard
`spkrec-ecapa-voxceleb` embedding size and the spec's requirement.

## Output checkpoints

`models/voice/train.py::train()` (and `models/voice/export.py::export_checkpoint`
for standalone re-export) produce, in `models/voice/saved/`:

- `voice_embedder.pt` — canonical, loaded by `models/voice/inference.py`.
- `voice_embedder.h5` — interoperability export
  (`models/common/checkpoint_io.py::save_state_dict_as_h5`, the same utility
  Face uses).
- `training_config.json` — the exact `VoiceConfig` used, for reproducibility.

Until a checkpoint exists, `VoiceEmbedder` transparently falls back to
`BaseEmbedder`'s deterministic mock-mode embedding, exactly like every
modality — the rest of the system (tests, `embeddings.pipelines.VoicePipeline`,
a future backend service) can be exercised end-to-end without a GPU.

## Kaggle GPU training

`kaggle_kernels/voice_training/` (`kernel-metadata.json` +
`voice-embedding-training.ipynb`) is the unattended, real-GPU counterpart to
`notebooks/04_voice_training.ipynb`'s manual/interactive Colab version — same
relationship face already has between `kaggle_kernels/face_training/` and
`notebooks/01_face_training_and_testing.ipynb`.
Run via:

```bash
python scripts/run_kaggle_kernels.py --only voice
```

which pushes the kernel, polls until it finishes on Kaggle's GPU, and
installs the resulting `.pt`/`.h5` checkpoints into `models/voice/saved/`
automatically — see `kaggle_kernels/README.md`.
