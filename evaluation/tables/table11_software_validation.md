# Software validation

| check | total | passed | failed | skipped | exit_code | seconds | source | evidence_label | result | errors | warnings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| backend pytest | 636 | 636 | 0 | 0 | 0 | 810.889 | evaluation/results/software_validation.csv (python -m evaluation.ieee.audit software) | SOFTWARE VALIDATION |  |  |  |
| coverage |  |  |  |  |  |  | evaluation/results/software_validation.csv (python -m evaluation.ieee.audit software) | SOFTWARE VALIDATION | NOT AVAILABLE (pytest-cov not installed) |  |  |
| frontend typecheck (tsc -b) |  |  |  |  | 0 | 26.8232 | evaluation/results/software_validation.csv (python -m evaluation.ieee.audit software) | SOFTWARE VALIDATION |  | 0 |  |
| frontend lint (oxlint) |  |  |  |  | 0 | 2.84096 | evaluation/results/software_validation.csv (python -m evaluation.ieee.audit software) | SOFTWARE VALIDATION |  | 0 | 25 |
| frontend build (vite build) |  |  |  |  | 0 | 41.2696 | evaluation/results/software_validation.csv (python -m evaluation.ieee.audit software) | SOFTWARE VALIDATION | success |  |  |
| frontend unit tests |  |  |  |  |  |  | evaluation/results/software_validation.csv (python -m evaluation.ieee.audit software) | SOFTWARE VALIDATION | NOT AVAILABLE (no frontend test suite in package.json) |  |  |
