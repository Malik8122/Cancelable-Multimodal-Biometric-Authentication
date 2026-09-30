# Latency benchmark

Evidence label: REAL DATA (measured on this machine; real samples).

Hardware: os=Windows 11 (10.0.26200), python=3.13.1, cpu_logical_cores=12, ram_gb=8.3, processor=12th Gen Intel(R) Core(TM) i5-1235U, torch=2.14.0+cpu, gpu=none (CPU only), torch_threads=10

| stage | mode | runs | mean ms | median ms | SD | p95 | p99 | min | max |
|---|---|---|---|---|---|---|---|---|---|
| face_preprocessing_mtcnn | warm | 100 | 33.02 | 32.24 | 5.06 | 42.51 | 45.97 | 21.99 | 46.61 |
| face_embedding_inceptionresnetv1 | warm | 100 | 38.01 | 39.25 | 3.77 | 44.13 | 45.04 | 30.20 | 50.09 |
| face_pipeline_total | warm | 100 | 69.34 | 70.30 | 6.25 | 76.92 | 91.35 | 55.15 | 93.56 |
| voice_preprocessing_logmel | warm | 100 | 17.40 | 17.97 | 5.44 | 25.83 | 29.34 | 7.21 | 30.79 |
| voice_embedding_ecapa | warm | 100 | 50.72 | 49.77 | 4.89 | 63.77 | 67.74 | 41.43 | 68.52 |
| voice_pipeline_total | warm | 100 | 74.43 | 69.47 | 25.72 | 94.17 | 150.21 | 54.79 | 287.25 |
| hkdf_key_derivation | warm | 100 | 0.05 | 0.04 | 0.01 | 0.06 | 0.11 | 0.04 | 0.11 |
| biohash_face_512d_256bit | warm | 100 | 43.53 | 42.40 | 5.32 | 50.52 | 63.87 | 37.00 | 68.53 |
| biohash_voice_192d_256bit | warm | 100 | 15.21 | 14.85 | 1.44 | 17.97 | 19.00 | 12.66 | 19.74 |
| hamming_comparison | warm | 100 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.02 |
| decision_estimate | warm | 100 | 0.09 | 0.09 | 0.04 | 0.13 | 0.20 | 0.08 | 0.40 |
| fusion_policy | warm | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.01 |
| db_lookup_active_template_2000_users_x4_sets | warm | 100 | 2.82 | 2.52 | 0.68 | 4.49 | 5.01 | 2.29 | 5.09 |
| api_authenticate_face | warm | 100 | 167.48 | 158.25 | 21.05 | 195.85 | 198.39 | 135.72 | 231.58 |
| api_authenticate_voice | warm | 100 | 128.98 | 127.46 | 11.33 | 141.24 | 155.24 | 109.80 | 211.25 |
| api_authenticate_face_plus_voice | warm | 100 | 777.79 | 757.20 | 99.62 | 845.16 | 1024.97 | 694.79 | 1658.76 |
| cold_face_import | cold | 10 | 1361.54 | 1177.18 | 355.43 | 2004.15 | 2200.76 | 1158.73 | 2249.91 |
| cold_face_model_load | cold | 10 | 4433.81 | 3768.69 | 1315.80 | 6931.80 | 7209.03 | 3640.85 | 7278.33 |
| cold_face_first_inference | cold | 10 | 209.12 | 151.98 | 115.56 | 427.13 | 467.52 | 145.92 | 477.62 |
| cold_face_total | cold | 10 | 6004.47 | 5155.10 | 1765.04 | 9351.87 | 9785.31 | 4960.03 | 9893.67 |
| cold_voice_import | cold | 10 | 1163.98 | 1165.56 | 15.41 | 1184.37 | 1188.22 | 1136.80 | 1189.18 |
| cold_voice_model_load | cold | 10 | 2167.82 | 2039.85 | 324.38 | 2719.04 | 2985.73 | 1996.13 | 3052.40 |
| cold_voice_first_inference | cold | 10 | 74.34 | 71.69 | 13.35 | 94.37 | 107.29 | 59.63 | 110.52 |
| cold_voice_total | cold | 10 | 3406.13 | 3261.52 | 326.36 | 3970.63 | 4222.10 | 3236.80 | 4284.97 |
