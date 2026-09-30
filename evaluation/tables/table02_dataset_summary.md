# Evaluation datasets

| modality | dataset | evaluation_split | identities | samples_embedded | failures_to_acquire | license | source | evidence_label |
|---|---|---|---|---|---|---|---|---|
| face | LFW (funneled) | identities with 2-19 images (disjoint from the >=20-image identities used for fine-tuning) | 1618 | 6141 | 0 | public; UMass non-commercial research use | evaluation/results/dataset_audit.csv (python -m evaluation.ieee.audit) | REAL DATA |
| voice | VoxCeleb1 subset, Indian celebrities (Kaggle gaurav41/voxceleb1-audio-wav-files-for-india-celebrity) | TEST split of the training code (per-speaker 70/15/15, seed 42): unseen utterances, speakers overlap training (closed-set) | 24 | 751 | 0 | DbCL-1.0 as declared by the mirror | evaluation/results/dataset_audit.csv (python -m evaluation.ieee.audit) | REAL DATA |
