# Template security properties

| modality | property | value | note | source | evidence_label |
|---|---|---|---|---|---|
| face | templates_analyzed | 1618.0000 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | mean_fraction_of_ones | 0.5003 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | min_position_fraction_of_ones | 0.4642 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | max_position_fraction_of_ones | 0.5315 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | mean_hamming_weight_bits | 128.0850 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | mean_per_bit_entropy_bits | 0.9996 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | sum_per_bit_entropy_upper_bound_bits | 255.8920 | upper bound; assumes independent bits | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | DERIVED |
| face | mean_abs_pairwise_bit_correlation | 0.0198 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | expected_mean_abs_correlation_if_independent | 0.0198 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | DERIVED |
| face | fraction_bit_pairs_abs_corr_gt_0.1 | 0.0000 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | cross_user_cross_key_mean_normalized_HD | 0.4998 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | cross_user_cross_key_sd_normalized_HD | 0.0314 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | degrees_of_freedom_daugman | 253.4840 | N = p(1-p)/sigma^2 from the cross-user cross-key HD distribution | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | DERIVED |
| face | decision_max_differing_bits | 46.0000 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | DERIVED |
| face | empirical_cross_key_collision_rate_at_threshold | 0.0000 | fraction of 19985 cross-user cross-key pairs within 46 bits | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | binomial_cross_key_collision_estimate_at_threshold | 0.0000 | binomial model with the DoF above | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | DERIVED |
| face | key_rotation_same_embedding_mean_similarity | 0.5002 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | key_rotation_same_embedding_sd_similarity | 0.0319 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | templates_analyzed | 24.0000 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | mean_fraction_of_ones | 0.4827 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | min_position_fraction_of_ones | 0.1667 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | max_position_fraction_of_ones | 0.7500 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | mean_hamming_weight_bits | 123.5830 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | mean_per_bit_entropy_bits | 0.9653 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | sum_per_bit_entropy_upper_bound_bits | 247.1090 | upper bound; assumes independent bits | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | DERIVED |
| voice | mean_abs_pairwise_bit_correlation | 0.1672 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | expected_mean_abs_correlation_if_independent | 0.1629 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | DERIVED |
| voice | fraction_bit_pairs_abs_corr_gt_0.1 | 0.5961 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | cross_user_cross_key_mean_normalized_HD | 0.4968 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | cross_user_cross_key_sd_normalized_HD | 0.0331 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | degrees_of_freedom_daugman | 227.8400 | N = p(1-p)/sigma^2 from the cross-user cross-key HD distribution | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | DERIVED |
| voice | decision_max_differing_bits | 55.0000 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | DERIVED |
| voice | empirical_cross_key_collision_rate_at_threshold | 0.0000 | fraction of 19173 cross-user cross-key pairs within 55 bits | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | binomial_cross_key_collision_estimate_at_threshold | 0.0000 | binomial model with the DoF above | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | DERIVED |
| voice | key_rotation_same_embedding_mean_similarity | 0.5029 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | key_rotation_same_embedding_sd_similarity | 0.0244 |  | evaluation/results/template_security.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | D_sys (bin 2/256) | 0.0212 |  | evaluation/results/unlinkability.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | D_sys (bin 4/256) | 0.0180 |  | evaluation/results/unlinkability.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | D_sys (bin 8/256) | 0.0176 |  | evaluation/results/unlinkability.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | D_sys (bin 4/256) - mated = the SAME sample under two keys (worst case) | 0.0192 |  | evaluation/results/unlinkability.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | D_sys (bin 2/256) | 0.0308 |  | evaluation/results/unlinkability.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | D_sys (bin 4/256) | 0.0200 |  | evaluation/results/unlinkability.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | D_sys (bin 8/256) | 0.0154 |  | evaluation/results/unlinkability.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | D_sys (bin 4/256) - mated = the SAME sample under two keys (worst case) | 0.2475 |  | evaluation/results/unlinkability.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | genuine_after_reenrollment: mean similarity / acceptance | 0.7645 / 22.00% |  | evaluation/results/revocation_summary.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | genuine_after_revocation_1: mean similarity / acceptance | 0.7640 / 22.67% |  | evaluation/results/revocation_summary.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | revoked_template_vs_new_active: mean similarity / acceptance | 0.5001 / 0.00% |  | evaluation/results/revocation_summary.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| face | within_key_genuine_before_revocation: mean similarity / acceptance | 0.7608 / 21.33% |  | evaluation/results/revocation_summary.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | genuine_after_reenrollment: mean similarity / acceptance | 0.8089 / 70.83% |  | evaluation/results/revocation_summary.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | genuine_after_revocation_1: mean similarity / acceptance | 0.8169 / 79.17% |  | evaluation/results/revocation_summary.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | revoked_template_vs_new_active: mean similarity / acceptance | 0.5036 / 0.00% |  | evaluation/results/revocation_summary.csv (python -m evaluation.ieee.experiments) | REAL DATA |
| voice | within_key_genuine_before_revocation: mean similarity / acceptance | 0.8110 / 70.83% |  | evaluation/results/revocation_summary.csv (python -m evaluation.ieee.experiments) | REAL DATA |
