# Paper training table

Only values safe to report: configuration read from code, and NEWLY_MEASURED values from COMPLETED reproducible runs. No historical, undocumented or estimated number appears here.

| Modality | Model | Embedding | Dataset | Training | Reproduced result |
|---|---|---|---|---|---|
| Face | InceptionResNetV1 (VGGFace2-pretrained), last block + ArcFace head fine-tuned | 512-D, L2 | LFW (62 identities, >= 20 images each) | Adam 1e-4, batch 32, 10 epochs, ArcFace s=30 m=0.5, no augmentation, per-identity 70/15/15 split | 10 epochs, 893 s on 12th Gen Intel(R) Core(TM) i5-1235U (CPU); final val. accuracy 0.922; closed-set test EER 0.0140, AUC 0.9992 |
| Voice | ECAPA-TDNN (SpeechBrain), trained from scratch | 192-D, L2 | VoxCeleb1 Indian-celebrity subset (24 speakers, 4,857 utterances) | AdamW 1e-3 (wd 1e-4), cosine schedule, batch 64, up to 30 epochs (patience 5 on val. loss), ArcFace s=30 m=0.5, 80-bin log-mel, augmentation disabled | NOT_AVAILABLE |

Configuration values: `models/*/config.py`, `models/*/train.py`, `kaggle_kernels/face_training/face-embedding-training.ipynb`. Counts (24 / 4,857; 62 / 3,023) are re-counted by the runner from the datasets (see run config.json).
