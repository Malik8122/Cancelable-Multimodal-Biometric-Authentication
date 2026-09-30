# Face training records

Reproduce: `python -m training.run_training face` (from the repository root; datasets under `data/`, see `training/TRAINING_REPRODUCIBILITY_REPORT.md`).

Each run lives in `runs/<experiment_id>/` with `config.json`, `environment.json`, `training_log.csv` (written after every epoch), `checkpoints/checkpoint_manifest.json` (checkpoint files themselves are gitignored), `best_checkpoint.json`, `metrics/`, `figures/`, `run_status.json`. `latest` names the newest run. The deployed checkpoint in `models/face/saved/` is NOT produced or replaced by these runs.

## Runs

- `face-20260925T031117Z` - status: COMPLETED; epochs logged: 10; training time: 893 s
