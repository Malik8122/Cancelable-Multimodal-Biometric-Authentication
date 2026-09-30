# Fusion policies (chimeric virtual users from real data; mean ± SD over pairings)

| policy | modalities | FAR | FRR | accuracy | F1_pooled | TP/FN/FP/TN | protocol | source | evidence_label |
|---|---|---|---|---|---|---|---|---|---|
| face_only | face | 0.00% ± 0.00% | 80.35% ± 5.79% | 96.65% | 0.3285 | 283/1157/0/33120 | 20 chimeric pairings x 24 virtual users | evaluation/results/fusion_policy_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice_only | voice | 0.13% ± 0.08% | 24.38% ± 2.04% | 98.86% | 0.8465 | 1089/351/44/33076 | 20 chimeric pairings x 24 virtual users | evaluation/results/fusion_policy_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| ALL_REQUIRED(face+voice) | face+voice | 0.00% ± 0.00% | 85.35% ± 5.17% | 96.44% | 0.2556 | 211/1229/0/33120 | 20 chimeric pairings x 24 virtual users | evaluation/results/fusion_policy_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| WEIGHTED(face+voice) | face+voice | 0.00% ± 0.00% | 59.58% ± 5.98% | 97.52% | 0.5757 | 582/858/0/33120 | 20 chimeric pairings x 24 virtual users | evaluation/results/fusion_policy_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| OR(face+voice) | face+voice | 0.13% ± 0.08% | 19.38% ± 2.94% | 99.07% | 0.8779 | 1161/279/44/33076 | 20 chimeric pairings x 24 virtual users | evaluation/results/fusion_policy_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| MAJORITY(face+voice)=AND | face+voice | 0.00% ± 0.00% | 85.35% ± 5.17% | 96.44% | 0.2556 | 211/1229/0/33120 | 20 chimeric pairings x 24 virtual users | evaluation/results/fusion_policy_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
