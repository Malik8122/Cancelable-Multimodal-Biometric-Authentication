# Latency (evaluation machine, CPU only)

| stage | mode | runs | median_ms | mean_ms | p95_ms | p99_ms | source | evidence_label |
|---|---|---|---|---|---|---|---|---|
| face_preprocessing_mtcnn | warm | 100 | 32.24 | 33.02 | 42.51 | 45.97 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| face_embedding_inceptionresnetv1 | warm | 100 | 39.25 | 38.01 | 44.13 | 45.04 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| face_pipeline_total | warm | 100 | 70.30 | 69.34 | 76.92 | 91.35 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| voice_preprocessing_logmel | warm | 100 | 17.97 | 17.40 | 25.83 | 29.34 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| voice_embedding_ecapa | warm | 100 | 49.77 | 50.72 | 63.77 | 67.74 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| voice_pipeline_total | warm | 100 | 69.47 | 74.43 | 94.17 | 150.21 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| hkdf_key_derivation | warm | 100 | 0.04 | 0.05 | 0.06 | 0.11 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| biohash_face_512d_256bit | warm | 100 | 42.40 | 43.53 | 50.52 | 63.87 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| biohash_voice_192d_256bit | warm | 100 | 14.85 | 15.21 | 17.97 | 19.00 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| hamming_comparison | warm | 100 | 0.01 | 0.01 | 0.01 | 0.01 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| decision_estimate | warm | 100 | 0.09 | 0.09 | 0.13 | 0.20 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| fusion_policy | warm | 100 | 0.00 | 0.00 | 0.00 | 0.01 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| db_lookup_active_template_2000_users_x4_sets | warm | 100 | 2.52 | 2.82 | 4.49 | 5.01 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| api_authenticate_face | warm | 100 | 158.25 | 167.48 | 195.85 | 198.39 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| api_authenticate_voice | warm | 100 | 127.46 | 128.98 | 141.24 | 155.24 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| api_authenticate_face_plus_voice | warm | 100 | 757.20 | 777.79 | 845.16 | 1024.97 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| cold_face_import | cold | 10 | 1177.18 | 1361.54 | 2004.15 | 2200.76 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| cold_face_model_load | cold | 10 | 3768.69 | 4433.81 | 6931.80 | 7209.03 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| cold_face_first_inference | cold | 10 | 151.98 | 209.12 | 427.13 | 467.52 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| cold_face_total | cold | 10 | 5155.10 | 6004.47 | 9351.87 | 9785.31 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| cold_voice_import | cold | 10 | 1165.56 | 1163.98 | 1184.37 | 1188.22 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| cold_voice_model_load | cold | 10 | 2039.85 | 2167.82 | 2719.04 | 2985.73 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| cold_voice_first_inference | cold | 10 | 71.69 | 74.34 | 94.37 | 107.28 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
| cold_voice_total | cold | 10 | 3261.52 | 3406.13 | 3970.63 | 4222.10 | evaluation/results/latency_benchmark.csv (python -m scripts.benchmark_latency) | REAL DATA |
