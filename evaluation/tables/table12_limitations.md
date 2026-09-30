# Limitations / remaining work

| limitation | source | evidence_label |
|---|---|---|
| No labelled real-user study of the deployed system (consented participants); harness ready | evaluation/real_user_evaluation.py | FUTURE WORK |
| Voice evaluation is closed-set (test speakers also in training; 24 speakers) | evaluation/results/dataset_audit.csv | FUTURE WORK |
| Fusion uses chimeric virtual users (independence assumption; 24 voice identities limit the user count) | evaluation/results/fusion_policy_metrics.csv | FUTURE WORK |
| No presentation-attack detection; IAPMR not measured (no attack data) | evaluation/results/pad_results.csv | FUTURE WORK |
| Robustness degradations are simulated; pose and physical microphone change not measured | evaluation/results/robustness.csv | FUTURE WORK |
| No demographic analysis (no demographic labels for the evaluation sets) | evaluation/results/dataset_audit.csv | FUTURE WORK |
| Thresholds 0.80 / 0.75 are teacher-requested, not optimized on validation data | backend/config.py | FUTURE WORK |
| Latency/memory measured on one laptop CPU; no GPU or server benchmark | evaluation/reports/hardware.json | FUTURE WORK |
| Security analysis is statistical (entropy, correlation, unlinkability); no attack (inversion, hill-climbing, stolen-key) experiments | evaluation/results/template_security.csv | FUTURE WORK |
