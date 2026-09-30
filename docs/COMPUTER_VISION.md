# Computer Vision — Detailed Technical Reference

This document explains, in depth, every computer-vision technique used anywhere in this project: what it is, why it was chosen, exactly how it is implemented in this codebase (with file/line pointers), and what a reviewer would need to know to defend it in an evaluation. It is a companion to `PROJECT_REPORT.md`, which covers the project's motivation and status; this document is purely the "how does the vision pipeline actually work" reference.

Face is the project's only image modality (voice is audio; its log-mel preprocessing is documented in `VOICE_MODEL.md`). Its pipeline has two stages, and it matters to be able to say precisely which stage is which:

1. **Preprocessing** (`preprocessing/`) — takes a raw image and produces a normalized representation, using a small pretrained detection network (MTCNN) for the classical CV task of "find and crop the face region."
2. **Embedding** (`models/`) — takes the preprocessed representation and produces a fixed-length numeric vector (the "embedding") using a deep convolutional neural network, fine-tuned with metric learning. This stage **is** deep learning.

| Modality | Preprocessing (classical CV) | Embedding (deep learning) |
|---|---|---|
| Face | MTCNN detection + bounding-box crop (default `FACE_ALIGNMENT=bbox`); optional 5-landmark similarity alignment (`FACE_ALIGNED`), see `evaluation/reports/FACE_ALIGNMENT_DECISION.md` | InceptionResnetV1 (512-d) |

---

## 1. Face: Detection, Alignment, and Embedding

**Code:** `preprocessing/face.py`, `models/face/inference.py`

### 1.1 Detection & alignment — MTCNN

MTCNN (Multi-task Cascaded Convolutional Networks; Zhang et al., 2016) is a three-stage cascade of small CNNs (P-Net → R-Net → O-Net) that progressively refines candidate face bounding boxes and produces five facial landmark points (eyes, nose, mouth corners) in a single pass. It is used here purely as the detection/localization front-end (`FacePreprocessor._get_detector` in `preprocessing/face.py`, via the `facenet-pytorch` package's `MTCNN` class), not as the identity-recognition model itself.

```python
detector = MTCNN(image_size=FACE_INPUT_SIZE, margin=0, post_process=True, device=self.device)
aligned = detector(pil_image)
```

**Correction (code audit, 2026-09-25):** an earlier version of this section said `MTCNN.__call__` warps the face to a canonical pose using the landmarks. It does not: `facenet-pytorch`'s `extract_face` crops the detected **bounding box** and resizes it to 160×160 (no rotation, no landmark use); the landmarks are used only as enrollment quality signals (roll, yaw). That bounding-box crop is the deployed default (`FACE_ALIGNMENT=bbox`, `FacePreprocessorBaseline`). A genuine 5-landmark similarity alignment (Umeyama least-squares rotation + uniform scale + translation onto a canonical 160×160 template, `preprocessing/face.py::FacePreprocessorAligned`) now exists as the optional `FACE_ALIGNED` mode; its measured effect is in `evaluation/reports/FACE_ALIGNMENT_DECISION.md`. The output is renormalized back to a standard `uint8` RGB array (`arr = ((arr * 128.0) + 127.5).clip(0, 255).astype(np.uint8)`) so the rest of the pipeline never has to know facenet-pytorch's internal tensor normalization convention.

**Why alignment matters at all:** a recognition network trained on canonically-posed faces performs substantially worse on faces at arbitrary rotation/scale/position — alignment removes that variance *before* the network has to learn to be invariant to it, which is both more accurate and requires less training data than expecting the embedding network to learn pose invariance itself.

### 1.2 Embedding — InceptionResnetV1

`models/face/inference.py` uses `InceptionResnetV1` (Szegedy et al.'s Inception architecture combined with residual connections, as adapted for face recognition and pretrained on VGGFace2 by the `facenet-pytorch` project) to map the aligned 160×160×3 face crop to a 512-dimensional embedding:

```python
tensor = (tensor - 127.5) / 128.0   # facenet-pytorch's expected input normalization
embedding = self._model(tensor)     # -> 512-d vector
```

The embedding is then L2-normalized by `BaseEmbedder.extract_embedding` (`models/common/base_embedder.py`), so that comparing two embeddings via cosine similarity is equivalent to comparing them via Euclidean distance — a standard convention in metric-learning-based recognition that keeps the comparison scale-invariant.

**Fine-tuning strategy:** rather than fine-tuning the entire network (expensive, and prone to catastrophic forgetting of the strong pretrained features), only the last Inception block plus the final linear/batch-norm layers are unfrozen:

```python
for name, param in backbone.named_parameters():
    param.requires_grad = any(name.startswith(p) for p in ['block8', 'last_linear', 'last_bn'])
```

This is a standard transfer-learning pattern: keep the low- and mid-level filters (edges, textures, local facial-part detectors) that VGGFace2 pretraining already learned well, and only adapt the highest-level, most task-specific layers to this project's specific enrolled identities.

---

## 2. The Shared Metric-Learning Approach: ArcFace

**Code:** `models/common/arcface.py`

Every modality's fine-tuning (face here, and voice via `models/voice/losses.py`) uses the same loss function: **ArcFace** (Additive Angular Margin Loss; Deng, Guo, Xue & Zafeiriou, CVPR 2019). Understanding why a *specialized* loss is needed at all, rather than plain softmax cross-entropy, is important:

A network trained with plain softmax cross-entropy to classify *training* identities learns features that are separable enough to tell those specific training identities apart — but biometric verification needs something stronger: at *test* time, the system must compare two embeddings of people (or images) it has never necessarily seen labeled together before and decide "same person or different person" via a similarity threshold. That requires the embedding space itself to have **large angular margins between different identities and tight clustering within the same identity**, not just enough separability to pick the right softmax class during training.

ArcFace achieves this by adding an angular margin penalty `m` directly to the angle between an embedding and its true class's weight vector, inside the softmax:

```python
cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))   # cos(theta) between embedding and each class center
phi = cosine * self.cos_m - sine * self.sin_m                          # = cos(theta + m), the angular-margin-penalized version
phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)      # numerical-stability guard, see below
...
output = (one_hot * phi + (1 - one_hot) * cosine) * self.s             # only the true class gets the margin penalty
```

In words: for the correct identity class, the model is penalized as if the angle between the embedding and that class's center were `m` radians (`m = 0.5`, roughly 28.6°) larger than it actually is, forcing the network to push same-identity embeddings *closer* together and different-identity embeddings *farther apart* than plain softmax would ever require, in order to still minimize the loss. The `s = 30.0` scale factor rescales the resulting cosine values before the softmax so gradients remain well-behaved (raw cosine similarities are confined to `[-1, 1]`, which produces a very "flat" softmax without rescaling). The `torch.where(cosine > self.threshold, ...)` guard exists because `cos(theta + m)` is not monotonic for angles near `pi`, so beyond a threshold angle the code falls back to a linear penalty (`cosine - self.mm`) instead — a standard numerical-stability fix described in the original ArcFace paper, not an ad-hoc addition.

**Why one shared loss across all modalities matters for this project specifically:** it means the training recipes are directly comparable to each other — a difference in evaluation numbers can be attributed to data/capacity factors (class count, unfrozen-layer capacity) rather than "we used a fundamentally different, less effective training approach for that modality," which is a cleaner, more defensible position in an evaluation.

`ArcMarginProduct` is used **only during training** — at inference time (`models/*/inference.py`), only the backbone's raw embedding is used; the ArcFace head's class-center weights are training scaffolding, discarded once the checkpoint is saved.

---

## 3. Evaluation Methodology (shared across all modalities)

**Code:** `evaluation/metrics.py`, `evaluation/roc.py`, `evaluation/experiments.py`

Every modality is evaluated identically, using standard biometric-verification metrics rather than plain classification accuracy, because **verification** (is this the same person as this enrolled template — a pairwise, open-set decision) is the actual deployed task, not **classification** (which of N known training identities is this).

- **Cosine similarity** (`evaluation/metrics.py: cosine_similarity`) — the comparison score between any two embeddings, consistent with the L2-normalization applied by `BaseEmbedder`.
- **Genuine / impostor pairs** (`evaluation/experiments.py: build_genuine_impostor_scores`) — every pair of test-set samples is scored; a pair from the *same* identity is a "genuine" pair (the score the system should recognize as a match), a pair from *different* identities is an "impostor" pair (the score the system should reject).
- **FAR / FRR at a threshold** (`compute_far_frr`) — the False Accept Rate is the fraction of impostor pairs that score *above* a chosen threshold (wrongly accepted); the False Reject Rate is the fraction of genuine pairs that score *below* it (wrongly rejected). These trade off against each other as the threshold moves — the entire reason multimodal fusion (`PROJECT_REPORT.md` §3) helps is that it can reduce both simultaneously, which single-modality threshold-tuning cannot.
- **Equal Error Rate (EER)** (`compute_eer`) — the threshold at which FAR and FRR are equal (or closest to equal, searched over all observed score values); this single number is the standard way the biometrics literature summarizes a system's overall discriminative power independent of any specific deployment's accept/reject policy.
- **ROC curve and AUC** (`evaluation/roc.py`, using `sklearn.metrics.roc_curve`/`auc`) — the full True-Positive-Rate-vs-False-Positive-Rate curve across every possible threshold, and the area under it (1.0 = perfect separation, 0.5 = random guessing) — the standard way to visualize and summarize verification performance independent of any single chosen threshold.

`run_modality_experiment` (`evaluation/experiments.py`) ties all of this together into the single function every training notebook calls after fine-tuning: it builds genuine/impostor scores from the held-out test split, finds the EER and its threshold, computes accuracy at that threshold, and computes the ROC/AUC — producing exactly the numbers reported in `PROJECT_REPORT.md` §8.2 for face. This same function is designed to be reused unchanged in Phase 2 (protected-vs-unprotected embedding comparison) and Phase 3 (every multimodal fusion combination), so the evaluation methodology stays identical throughout the project rather than being redefined per phase.
