# Split / Graph Structure Analysis

## Seed-level summary

| seed | train normal | train anom in raw train | test anom | test anom rate | anom 1-hop to trainN | anom 2-hop to trainN | anom dist centroid |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2086 | 242 | 4303 | 0.0973 | 0.0216 | 0.0720 | 6.0371 |
| 1 | 2097 | 231 | 4314 | 0.0975 | 0.0183 | 0.0795 | 5.8037 |
| 2 | 2091 | 237 | 4308 | 0.0974 | 0.0186 | 0.0699 | 5.7853 |
| 3 | 2099 | 229 | 4316 | 0.0976 | 0.0127 | 0.0767 | 5.8533 |
| 4 | 2108 | 220 | 4325 | 0.0978 | 0.0173 | 0.0726 | 5.8446 |

## Seed 3 deltas from non-seed3 mean

| metric | seed3 | non3 mean | delta |
|---|---:|---:|---:|
| n_train_anom | 229.000000 | 232.500000 | -3.500000 |
| n_train_normal | 2099.000000 | 2095.500000 | 3.500000 |
| n_test_anom | 4316.000000 | 4312.500000 | 3.500000 |
| test_anom_rate | 0.097568 | 0.097488 | 0.000079 |
| test_anom_adjacent_to_train_normal_ratio | 0.012743 | 0.018959 | -0.006216 |
| test_norm_adjacent_to_train_normal_ratio | 0.071668 | 0.075914 | -0.004245 |
| test_anom_within_2hop_train_normal_ratio | 0.076691 | 0.073506 | 0.003186 |
| test_norm_within_2hop_train_normal_ratio | 0.215456 | 0.217340 | -0.001885 |
| train_normal_degree.mean | 1.606479 | 1.721937 | -0.115458 |
| test_anom_degree.mean | 0.814643 | 0.807895 | 0.006748 |
| train_normal_feat_norm.mean | 11.286948 | 11.210851 | 0.076097 |
| test_anom_feat_norm.mean | 3.923000 | 3.914726 | 0.008274 |
| test_anom_dist_to_train_normal_centroid.mean | 5.853294 | 5.867677 | -0.014383 |
| test_norm_dist_to_train_normal_centroid.mean | 11.352037 | 11.364292 | -0.012255 |
