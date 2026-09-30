# Face alignment experiment runs

This file is hand-written; `training/summarize.py` does not regenerate it. These are controlled pairs, and the deployed
checkpoint is not replaced.

| Run | Folder | Images | Preprocessing | Status | Training time | Final val acc / val EER | Closed-set test EER / AUC |
|---|---|---|---|---|---|---|---|
| `face_aligned_v1` | `aligned_v1/` | sklearn 125×94 slice | similarity-aligned | COMPLETED, **confounded** (see `aligned_v1/README.md`) | 2982 s (CPU under load) | 0.704 / 1.30 % | 0.92 % / 0.9990 |
| `face_bbox_fullframe_v2` | `bbox_fullframe_v2/` | full 250×250 frames, same 3,023 files | bounding-box crop | COMPLETED | 1075 s | 0.923 / 4.96 % | 3.19 % / 0.9856 |
| `face_aligned_v2` | `aligned_v2/` | full 250×250 frames, same 3,023 files | similarity-aligned | COMPLETED | 1121 s | 0.948 / 4.10 % | 2.14 % / 0.9887 |

Within the v2 pair, the configurations are identical except for the preprocessing:

- seed 42;
- the same kept-image set (SHA-256 `ad6a0cb5…`, 0 no-face, 0 alignment failures);
- the same 2091/426/506 split;
- the same model, loss, optimizer and 10 epochs.

Each run folder has `config.json`, `environment.json`, `training_log.csv`, `checkpoints/checkpoint_manifest.json`,
`metrics/test_verification.json` and `run_status.json`. The `.pt` files are git-ignored.

The closed-set test uses the training identities and is indicative only. The identity-disjoint held-out comparison is in
`evaluation/results/face_alignment_metrics.csv` and `evaluation/reports/FACE_ALIGNMENT_DECISION.md`.

Reproduce:

```
python -m training.run_training face --face-alignment bbox --face-source full_frame --experiment-id face_bbox_fullframe_v2
python -m training.run_training face --face-alignment similarity --face-source full_frame --experiment-id face_aligned_v2
python -m training.run_training face --face-alignment similarity --experiment-id face_aligned_v1   # confounded v1
```
