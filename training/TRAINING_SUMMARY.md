# Training summary

Evidence types: **NEWLY_MEASURED** (this repository's reproducible runs, `training/<modality>/runs/`), **HISTORICAL_DOCUMENTED** (copied from repository documents, not reproducible), **NOT_AVAILABLE**. They are never merged.

## Newly measured (reproduced runs)

| Modality | Model | Dimension | Dataset | Subjects | Samples | Epochs | Best Metric | Training Time | Hardware | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| face | InceptionResnetV1 | 512 | LFW funneled | 62 | 3023 | 10 of 10 | validation_accuracy (final epoch) = 0.9220 (epoch 10) | 893 s | 12th Gen Intel(R) Core(TM) i5-1235U, 8.29 GB RAM, GPU: none, torch 2.14.0+cpu | COMPLETED |
| voice | SpeechBrain ECAPA_TDNN | 192 | Kaggle gaurav41/voxceleb1-audio-wav-files-for-india-celebrity | 24 | 4857 | 8 of 30 | NOT_AVAILABLE | 11731 s | 12th Gen Intel(R) Core(TM) i5-1235U, 8.29 GB RAM, GPU: none, torch 2.14.0+cpu | INCOMPLETE |

Only a run with status COMPLETED is a reproduced training; STOPPED / INTERRUPTED runs are partial records.

## Historical documented values (not reproducible)

| Modality | Quantity | Value | Source | Evidence |
|---|---|---|---|---|
| face | identities / images | 62 / 3,023 | docs/PROJECT_REPORT.md §8.2 | HISTORICAL_DOCUMENTED |
| face | final epoch train_loss / train_acc / val_acc | 0.1378 / 0.976 / 0.913 (epoch 10/10) | docs/PROJECT_REPORT.md §8.2 | HISTORICAL_DOCUMENTED |
| face | test accuracy@EER / EER / AUC | 0.990 / 0.010 / 0.999 | evaluation/results/face_metrics.csv; docs/PROJECT_REPORT.md §8.2 | HISTORICAL_DOCUMENTED |
| face | hardware | Kaggle kernel with GPU enabled; completed runs reported as CPU fallback | kernel-metadata.json; docs/PROJECT_REPORT.md §8.3 | HISTORICAL_DOCUMENTED |
| face | training duration | NOT_AVAILABLE | no saved notebook outputs | NOT_AVAILABLE |
| voice | speakers / utterances | 24 / 4,857 | docs/VOICE_MODEL.md | HISTORICAL_DOCUMENTED |
| voice | train / val / test | 3,389 / 717 / 751 | docs/VOICE_MODEL.md | HISTORICAL_DOCUMENTED |
| voice | epochs completed | 30 (all) | docs/VOICE_MODEL.md | HISTORICAL_DOCUMENTED |
| voice | hardware / duration | CPU fallback; 'a little under 8 hours' | docs/VOICE_MODEL.md | HISTORICAL_DOCUMENTED |
| voice | test EER / accuracy@EER / AUC | 2.29% / 97.71% / 0.9967 | evaluation/results/voice_metrics.csv | HISTORICAL_DOCUMENTED |
| voice | validation EER during training | NOT_AVAILABLE | models/voice/train.py computes validation loss only | NOT_AVAILABLE |
| voice | best epoch | NOT_AVAILABLE | no saved notebook outputs | NOT_AVAILABLE |

## All runs

- **face**: `face-20260925T031117Z` - status: COMPLETED; epochs logged: 10; training time: 893 s
- **voice**: `voice-20260925T032917Z` - status: INCOMPLETE (no run_status.json); epochs logged: 8; training time: 11731 s
