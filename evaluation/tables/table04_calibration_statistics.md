# Calibration: synthetic fit and real-embedding validation

| modality | data | max_fit_residual_hamming | coefficients | RMSE_cosine | MAE_cosine | R2 | pearson_r | bias | source | evidence_label |
|---|---|---|---|---|---|---|---|---|---|---|
| face | synthetic pairs (19,600) | 0.00378 | 0.8817, 0.0274, -0.0214 | n/a | n/a | n/a | n/a | n/a | evaluation/results/biohash_metric_calibration.json (python -m scripts.calibrate_biohash_metric_mapping) | SYNTHETIC CALIBRATION |
| voice | synthetic pairs (19,600) | 0.00306 | 0.8855, 0.0191, -0.0178 | n/a | n/a | n/a | n/a | n/a | evaluation/results/biohash_metric_calibration.json (python -m scripts.calibrate_biohash_metric_mapping) | SYNTHETIC CALIBRATION |
| face | real pairs, all (n=85423) | n/a | n/a | 0.1004 | 0.0799 | 0.7735 | 0.8945 | 0.0057 | evaluation/results/calibration_real_validation.csv (python -m scripts.validate_calibration_real) | REAL DATA |
| face | real pairs, genuine (n=4523) | n/a | n/a | 0.0717 | 0.0554 | 0.8731 | 0.9391 | -0.0022 | evaluation/results/calibration_real_validation.csv (python -m scripts.validate_calibration_real) | REAL DATA |
| face | real pairs, in_calibrated_range (n=76916) | n/a | n/a | 0.0999 | 0.0795 | 0.7536 | 0.8852 | 0.0098 | evaluation/results/calibration_real_validation.csv (python -m scripts.validate_calibration_real) | REAL DATA |
| voice | real pairs, all (n=18000) | n/a | n/a | 0.0932 | 0.0736 | 0.8269 | 0.9175 | 0.0036 | evaluation/results/calibration_real_validation.csv (python -m scripts.validate_calibration_real) | REAL DATA |
| voice | real pairs, genuine (n=727) | n/a | n/a | 0.0463 | 0.0352 | 0.8435 | 0.9224 | 0.0040 | evaluation/results/calibration_real_validation.csv (python -m scripts.validate_calibration_real) | REAL DATA |
| voice | real pairs, in_calibrated_range (n=16305) | n/a | n/a | 0.0920 | 0.0727 | 0.8141 | 0.9118 | 0.0063 | evaluation/results/calibration_real_validation.csv (python -m scripts.validate_calibration_real) | REAL DATA |
