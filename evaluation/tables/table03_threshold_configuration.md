# Threshold configuration

| modality | metric | direction | threshold | hamming_equivalent | origin | source | evidence_label |
|---|---|---|---|---|---|---|---|
| face | estimated cosine similarity | higher is better | 0.8 | h >= 0.8176 (<= 46 of 256 bits differ) | teacher-requested; not experimentally optimized | backend/config.py:136 | CONFIGURATION |
| voice | estimated Euclidean distance | lower is better | 0.75 | h >= 0.7823 (<= 55 of 256 bits differ) | teacher-requested; not experimentally optimized | backend/config.py:141 | CONFIGURATION |
| fusion (face+voice) | mean of fusion-scale scores | higher is better | 0.759375 | - | derived = mean(0.80, 1 - 0.75^2/2); informational under ALL_REQUIRED | backend/services/modality_metrics.py | DERIVED |
