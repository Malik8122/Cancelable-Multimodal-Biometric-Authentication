"""IEEE evaluation package: reproducible experiments on the shipped biometric pipeline.

Every artefact written by this package carries exactly one evidence label (see `common.LABELS`). Raw biometric data
and embeddings are only ever read from / cached under the gitignored `data/` directory; results contain scores only.
Entry point: `python -m evaluation.ieee.run_all`.
"""
