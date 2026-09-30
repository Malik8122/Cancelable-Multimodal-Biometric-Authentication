# Embedding-model performance (raw embeddings)

| modality | score | n_genuine | n_impostor | EER | EER_95CI | ROC_AUC | TAR@FAR=1% | TAR@FAR=0.1% | operating_rule | FAR@op | FRR@op | F1@op | source | evidence_label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| face | raw_cosine | 4523 | 80900 | 7.12% | 6.54% - 7.62% | 0.9682 | 84.79% | 66.88% | raw_cosine >= 0.8 | 0.00% | 83.11% | 0.2890 | evaluation/results/raw_vs_protected_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | raw_distance | 727 | 17273 | 2.21% | 1.91% - 2.88% | 0.9980 | 94.91% | 79.92% | raw_distance <= 0.75 | 0.03% | 24.07% | 0.8591 | evaluation/results/raw_vs_protected_metrics.csv (python -m evaluation.ieee.experiments) | REAL DATA |
