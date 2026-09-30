# Datasets

None of these datasets are committed to this repository (see `.gitignore`). Raw
biometric images only ever exist locally / in a Colab runtime during
preprocessing and training, per the privacy-by-design principle in
`docs/PRIVACY_AND_SECURITY.md`.

## Face — LFW (Labeled Faces in the Wild)

- **Source:** [vis-www.cs.umass.edu/lfw](http://vis-www.cs.umass.edu/lfw/), downloaded
  in `notebooks/01_face_training_and_testing.ipynb` via
  `sklearn.datasets.fetch_lfw_people` — no account or manual download needed.
- **License:** released for non-commercial research use by the University of
  Massachusetts; individual images retain the copyright/usage terms of their
  original web sources. Redistribution of derived embeddings/models trained on
  it for research purposes is standard practice in the face-recognition
  literature, but this is **not** a commercial-use license.
- **Retention:** raw images live only in the Colab runtime's `/content` during
  training; they are never written into this repository.

## Voice — VoxCeleb1 (subset)

- **Source used:** [gaurav41/voxceleb1-audio-wav-files-for-india-celebrity](https://www.kaggle.com/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity)
  on Kaggle, attached natively in `kaggle_kernels/voice_training/kernel-metadata.json`
  (no manual download, no Kaggle API token needed inside the kernel).
- **What it actually is:** a real-audio **subset** of VoxCeleb1 (Indian-celebrity
  speakers only), not the full ~1,251-speaker VoxCeleb1 corpus. The official
  full corpus ([robots.ox.ac.uk/~vgg/data/voxceleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html))
  is tens of GB split across many part-files and impractical to download
  unattended inside a single Kaggle kernel session within this capstone's
  scope.
  Exact speaker/utterance/split counts are **not hardcoded here** — they
  depend on exactly what this subset contains and are printed at kernel
  run time by `kaggle_kernels/voice_training/voice-embedding-training.ipynb`'s
  dataset cell (see `docs/VOICE_MODEL.md`).
- **License:** DbCL-1.0 (Database Contents License), as declared by the
  dataset's Kaggle listing. This is a derived subset of VoxCeleb1's audio,
  not an independently-verified redistribution grant from VoxCeleb1's
  original maintainers (University of Oxford VGG) — noted explicitly per
  this project's "do not assume datasets can be freely redistributed"
  principle.
- **Retention:** same as every other modality — raw audio only ever exists
  locally / in the Kaggle runtime during preprocessing and training; never
  committed to this repository.

## General policy

- No raw biometric image, of any modality, from any dataset, is ever committed
  to this repository or the production database (see `.gitignore` and
  `docs/PRIVACY_AND_SECURITY.md` for the enforcement mechanism).
- If you swap in a different dataset for any modality, update this file with
  its source, license, and retention notes — don't assume a new dataset is
  automatically fine to redistribute or commit.
