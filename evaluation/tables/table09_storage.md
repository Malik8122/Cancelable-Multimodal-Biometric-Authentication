# Storage

| experiment | users | file_bytes | rows:users | rows:protected_templates | rows:audit_logs | bytes_per_user_incremental | source | evidence_label | bytes_per_audit_row | bytes | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| enrolled_users_face_voice_4_sets | 1 | 114688 | 1 | 8 | 0 | 0 | evaluation/results/storage_analysis.csv (python -m evaluation.ieee.storage) | SOFTWARE VALIDATION |  |  |  |
| enrolled_users_face_voice_4_sets | 10 | 135168 | 10 | 80 | 0 | 2048 | evaluation/results/storage_analysis.csv (python -m evaluation.ieee.storage) | SOFTWARE VALIDATION |  |  |  |
| enrolled_users_face_voice_4_sets | 100 | 524288 | 100 | 800 | 0 | 4096 | evaluation/results/storage_analysis.csv (python -m evaluation.ieee.storage) | SOFTWARE VALIDATION |  |  |  |
| enrolled_users_face_voice_4_sets | 1000 | 4075520 | 1000 | 8000 | 0 | 3960.83 | evaluation/results/storage_analysis.csv (python -m evaluation.ieee.storage) | SOFTWARE VALIDATION |  |  |  |
| plus_1000_audit_rows | 1000 | 4575232 | 1000 | 8000 | 1000 |  | evaluation/results/storage_analysis.csv (python -m evaluation.ieee.storage) | SOFTWARE VALIDATION | 499.712 |  |  |
| template_payload_per_modality |  |  |  |  |  |  | evaluation/results/storage_analysis.csv (python -m evaluation.ieee.storage) | DERIVED |  | 32 | 256 bits packed |
| template_payload_per_user_face_voice_4_sets |  |  |  |  |  |  | evaluation/results/storage_analysis.csv (python -m evaluation.ieee.storage) | DERIVED |  | 256 |  |
| raw_embedding_float32_equivalent_face_voice |  |  |  |  |  |  | evaluation/results/storage_analysis.csv (python -m evaluation.ieee.storage) | DERIVED |  | 2816 | what storing embeddings would cost - NOT stored by this system |
