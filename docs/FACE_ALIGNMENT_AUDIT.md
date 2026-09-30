# Face pipeline audit (before alignment)

Audited at `main` @ `873c13e` (+ uncommitted evaluation work), 2026-09-25. Code is authoritative; line numbers drift, so
functions are named.

## 1. Current pipeline (authentication path)

```
RGB uint8 image (backend/utils.py::decode_biometric_sample)
 -> FacePreprocessor.preprocess (preprocessing/face.py)
      MTCNN(image_size=160, margin=0, post_process=True, device)   # facenet-pytorch; other args = library defaults
      detector(PIL image) == MTCNN.forward:
         detect() -> boxes, probs, 5 landmarks   (landmarks computed internally, then DISCARDED)
         select_largest=True (default)          -> largest box
         extract_face(img, box, 160, margin=0)  -> crop of the box, resized to 160x160 (PIL bilinear; black fill outside image)
         fixed_image_standardization            -> (x - 127.5) / 128
      tensor -> HWC -> ((x * 128) + 127.5).clip(0, 255).uint8      # back to uint8 RGB
 -> FaceEmbedder.extract_embedding (models/face/inference.py)
      (x - 127.5) / 128 -> tensor (1, 3, 160, 160) float32 -> InceptionResnetV1(classify=False) -> 512-D
      BaseEmbedder._l2_normalize (models/common/base_embedder.py)
 -> BioHash 256 bits (template_protection/biohash.py) -> Hamming comparison
```

| # | Question | Answer (source) |
|---|---|---|
| 1 | MTCNN initialization | `MTCNN(image_size=160, margin=0, post_process=True, device=self.device)`; `select_largest=True`, `keep_all=False`, thresholds `[0.6, 0.7, 0.7]`, `min_face_size=20` are library defaults (`FacePreprocessor._get_detector`) |
| 2 | Bounding boxes | `MTCNN.detect` inside `forward()` (authentication); `detector.detect(pil, landmarks=True)` in `detect_and_align` (enrollment) |
| 3 | Landmarks returned? | Authentication: **no** (computed by MTCNN, discarded by `forward`). Enrollment: yes, `(5, 2)`, order left eye, right eye, nose, mouth left, mouth right |
| 4 | How landmarks are used | **Only quality signals**: roll = eye-line angle (≤ 20°), yaw ratio = nose offset / inter-eye distance (≤ 0.20) (`detect_and_align`, `embeddings/pipelines.py::_evaluate_face_quality`). Never geometric alignment |
| 5 | Crop coordinates | the MTCNN box `(x1, y1, x2, y2)` of the largest face (confidence-filtered, single face, in enrollment); `margin=0` |
| 6 | Image size | 160 × 160 (`FACE_INPUT_SIZE`) |
| 7 | Padding / margin | margin 0; parts of the box outside the image are filled black by PIL crop |
| 8 | RGB conversion | inputs are RGB throughout (`decode_biometric_sample`; PIL from RGB array) |
| 9 | Pixel normalization | `fixed_image_standardization` then back to uint8 in the preprocessor; `(x − 127.5)/128` again in the embedder (net: one standardization of the uint8 crop) |
| 10 | Tensor into the model | `(1, 3, 160, 160)` float32 in ≈ [−1, 1] |
| 11 | Checkpoint | `models/face/saved/face_embedder.pt` (Git LFS, ~112 MB), InceptionResnetV1 VGGFace2-initialized, fine-tuned on LFW (≥ 20 images/identity), `settings.face_model_path` |
| 12 | Training vs inference preprocessing | **Match.** Training (`kaggle_kernels/face_training` cell 6/12) used `FacePreprocessor.preprocess` (bbox crop) + `(x − 127.5)/128`; inference uses the same. Training images were scikit-learn LFW crops (125×94, default slice) - inference sees full frames, but the same detector/crop code |

## 2. Enrollment path

`FacePipeline.embed_poses` → `detect_and_align` per pose → quality gates (`_evaluate_face_quality`: confidence ≥ 0.90, size
ratio ≥ 0.15, centre offset ≤ 0.35, roll ≤ 20°, yaw ≤ 0.20, sharpness ≥ 25) → embeddings of VALID poses (≥ 3 of 5,
`MIN_VALID_POSES`) → centroid (`embeddings/centroid.py`) → 4 BioHash template sets.

## 3. Other consumers

- `frontend/src/components/capture/GuidedFaceCapture.tsx`: captures poses, calls `POST /enroll/face/check-pose`, shows the backend hint (`backend/services/face_enrollment.py::POSE_HINTS`).
- Evaluation: `evaluation/ieee/extract_embeddings.py`, `robustness.py` (use `FacePipeline`, i.e. the baseline).
- Existing face metrics: `evaluation/results/face_metrics.csv` (training-identity test, not verifiable); held-out LFW evaluation in `evaluation/results/raw_vs_protected_metrics.csv`.

## 4. Compatibility issues for alignment

1. **Checkpoint/preprocessing coupling.** The deployed checkpoint was trained on bbox crops; feeding it aligned crops is
   a distribution shift. An aligned model must be (re)trained with aligned preprocessing; the old checkpoint + aligned
   preprocessing is only an ablation.
2. **Framing.** A generic template (e.g. ArcFace 112×112) frames the face differently from the bbox crops the model
   learned; the canonical template must match the model's framing to isolate geometric normalization.
3. **Face selection differs between paths** (authentication: largest face; enrollment: confidence-filtered single
   face). The aligned path must reproduce each path's selection exactly.
4. **Quality gates.** Roll/yaw are measured from landmarks of the capture; after alignment roll is ~0 by construction,
   so gates must stay on pre-alignment geometry or they stop gating.
5. **Failure modes.** Degenerate landmarks need an explicit status that API callers already handle (ValueError family)
   and that is distinguishable from NO_FACE / BLURRY.
6. **Privacy.** Landmarks are biometric-derived; they must stay transient (not stored, logged or returned).
7. **Existing enrollments.** Templates enrolled under one preprocessing will not match probes preprocessed the other
   way; switching production requires re-enrollment (templates are revocable, so this is operationally possible).
