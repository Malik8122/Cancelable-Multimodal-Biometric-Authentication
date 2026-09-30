# Voice training records

Reproduce: `python -m training.run_training voice` (from the repository root; datasets under `data/`, see `training/TRAINING_REPRODUCIBILITY_REPORT.md`).

Each run lives in `runs/<experiment_id>/` with `config.json`, `environment.json`, `training_log.csv` (written after every epoch), `checkpoints/checkpoint_manifest.json` (checkpoint files themselves are gitignored), `best_checkpoint.json`, `metrics/`, `figures/`, `run_status.json`. `latest` names the newest run. The deployed checkpoint in `models/voice/saved/` is NOT produced or replaced by these runs.

## Runs

- `voice-20260925T032917Z` - status: INCOMPLETE (no run_status.json); epochs logged: 8; training time: 11731 s
