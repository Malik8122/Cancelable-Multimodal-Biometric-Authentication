# face_aligned_v1 — literal-recipe aligned run (CONFOUNDED)

Status: **COMPLETED**, but **not used to decide** on alignment.

This run is the historical notebook recipe with only one change: the preprocessor is `FacePreprocessorAligned` instead of the
bounding-box crop. Everything else matches the reproduced baseline `training/face/runs/face-20260925T031117Z`:

- seed 42
- the same 2065/423/496 split
- InceptionResnetV1 (VGGFace2) with an ArcFace head (s=30, m=0.5)
- Adam 1e-4, batch 32, 10 epochs

Run it with:

```
python -m training.run_training face --face-alignment similarity --experiment-id face_aligned_v1
```

## Why it is confounded

The recipe trains on the images returned by `sklearn.datasets.fetch_lfw_people`. By default that function slices each
250×250 funneled image to a 125×94 region. The slice is tight around the face, and the similarity transform has to pull
in pixels from outside it, so the aligned training crops are mostly empty border. On 60 sampled images, measured by
`evaluation/ieee/face_alignment_eval.py` during development:

| image set | black-pixel fraction of the aligned crop |
|---|---|
| training images (sklearn 125×94 slice) | mean 0.391, p90 0.585 |
| held-out evaluation images (full 250×250 frame) | mean 0.010 |

The bbox crops have no border (mean 0.000) because the MTCNN box lies inside the slice. The aligned model therefore
learned from partly blank faces and is tested on complete ones, so comparing it with the baseline mixes up the
preprocessing with a train/test content mismatch.

## What replaces it

A controlled pair trained on the **full 250×250 frames** of the same 3,023 files. The loader replicates sklearn's file
order and RandomState(42) shuffle, and this is checked against `fetch_lfw_people`:

- `training/face/bbox_fullframe_v2`: `--face-alignment bbox --face-source full_frame --experiment-id face_bbox_fullframe_v2`
- `training/face/aligned_v2`: `--face-alignment similarity --face-source full_frame --experiment-id face_aligned_v2`

The v1 numbers are still reported in `evaluation/results/face_alignment_metrics.csv` (systems `C_repro_bbox` and
`D_aligned_v1`) and in `evaluation/reports/FACE_ALIGNMENT_DECISION.md`, labelled as confounded.

The checkpoint (`checkpoints/*.pt`, about 110 MB) is git-ignored and can be regenerated with the command above.
