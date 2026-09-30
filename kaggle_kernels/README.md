# Kaggle Kernels — real GPU training via the API

These kernels are the actual-execution counterpart to the Colab notebooks
(`notebooks/01_face_training_and_testing.ipynb`, `notebooks/04_voice_training.ipynb`): same training logic, adapted
to run unattended on Kaggle's free GPU-backed Kernels (P100/T4) via the
`kaggle` CLI, with datasets attached natively instead of downloaded manually.

```text
kaggle_kernels/
  face_training/            kernel-metadata.json + face-embedding-training.ipynb
  voice_training/           kernel-metadata.json + voice-embedding-training.ipynb
```

## One-time setup

1. Create a Kaggle API token: kaggle.com → Settings → API → **Create New
   Token**. This downloads `kaggle.json`.
2. Save it to `~/.kaggle/kaggle.json` (`C:\Users\<you>\.kaggle\kaggle.json`
   on Windows).
3. `pip install kaggle`

Each `kernel-metadata.json`'s `"id"` field has a placeholder
(`<your-kaggle-username>/...`) — replace it with your real Kaggle username
before pushing (or run `scripts/run_kaggle_kernels.py`, which fills it in
automatically from your `kaggle.json`).

## Push, monitor, and pull results

```bash
# push (starts the run on Kaggle's GPU infrastructure)
kaggle kernels push -p kaggle_kernels/face_training

# poll status until it finishes
kaggle kernels status <username>/face-embedding-training-phase-1

# pull the output (includes the repo clone with checkpoints under
# repo/models/face/saved/) once status is "complete"
kaggle kernels output <username>/face-embedding-training-phase-1 -p ./kaggle_output/face
```

Note: Kaggle derives the actual kernel slug from the *title* in
`kernel-metadata.json` if it doesn't match the `id` you set, so the `id`
fields here already include the `-phase-1` suffix Kaggle would otherwise add
automatically (avoids the "kernel title does not resolve to the specified
id" mismatch warning).

Repeat for `voice_training`.
`scripts/run_kaggle_kernels.py` automates both: push → poll → pull →
copy the resulting `.pt`/`.h5` checkpoints into `models/<modality>/saved/`
in this repo.

## Notes specific to each kernel

- **Face:** no attached dataset — LFW is downloaded inside the kernel via
  `sklearn.datasets.fetch_lfw_people` (kernel internet access is enabled in
  `kernel-metadata.json`).
- **Voice:** attaches `gaurav41/voxceleb1-audio-wav-files-for-india-celebrity` -
  a real-audio subset of VoxCeleb1 (Indian-celebrity speakers, DbCL-1.0
  licensed), not the full VoxCeleb1 corpus. The notebook auto-discovers
  `DATASET_ROOT` by searching for the mounted directory whose immediate
  children are `idNNNNN` speaker folders, rather than assuming one fixed
  nesting depth under `/kaggle/input/` - an earlier version hardcoded
  `vox1_indian/content/vox_indian/`, which turned out not to match this
  dataset's actual mount layout on a real run (only its *listed* file paths,
  not necessarily its mount structure). See `docs/DATASETS.md`'s Voice
  section and `docs/VOICE_MODEL.md` for why a subset was used and what that
  means for reported metrics.
