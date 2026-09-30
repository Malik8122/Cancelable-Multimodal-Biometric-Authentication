# Voice reproduction status

**Training reproduced: NO** - no run had reached COMPLETED when these documents were generated.

What was found: the training code (`models/voice/train.py`), its configuration, the dataset (under `data/`) and the deployed checkpoint are all present.

Measured on this machine: 8 epoch(s) completed, mean epoch wall time 1466 s (configured maximum 30 epochs; early stopping may end sooner). A run without run_status.json was still in progress or was stopped externally; its completed epochs are preserved in training_log.csv.

Command: `python -m training.run_training voice` (then `python -m training.summarize`)

Runs:
- `voice-20260925T032917Z` - status: INCOMPLETE (no run_status.json); epochs logged: 8; training time: 11731 s

Historical information that remains unavailable:
- validation EER during training (models/voice/train.py computes validation loss only)
- best epoch (no saved notebook outputs)
