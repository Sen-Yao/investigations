# Phase 2 Route2 Stability Diagnostic (Seeds 0-4)

> Exploratory no-training diagnostic. Seeds 1-4 were executed via manual SSH with a Hermes cron watchdog rather than experiment-runner; record as protocol deviation, not formal method-validation evidence.

## Focus metrics

| signal | AUC mean±std | AP mean±std | ρ w/ margin mean±std | top5 Jaccard mean±std | AUC > margin |
|---|---:|---:|---:|---:|---:|
| margin | 0.7952±0.0071 | 0.5163±0.0221 | 1.0000±0.0000 | 1.0000±0.0000 | - |
| mat_mean | 0.8009±0.0203 | 0.5335±0.0621 | 0.6988±0.0135 | 0.5258±0.1408 | 3/5 |
| mat_entropy | 0.7777±0.0296 | 0.5024±0.0689 | 0.5900±0.0204 | 0.4718±0.1412 | 1/5 |
| mat_high08_ratio | 0.7878±0.0232 | 0.4708±0.0668 | 0.7028±0.0058 | 0.4853±0.1277 | 2/5 |
| ra_anom_ratio_diagnostic | 0.9328±0.0005 | 0.7722±0.0050 | 0.2322±0.0023 | 0.4666±0.0576 | 5/5 |
| ra_resp_mean | 0.7957±0.0108 | 0.5445±0.0143 | 0.8443±0.0066 | 0.7255±0.0403 | 3/5 |
| ra_resp_min | 0.7662±0.0146 | 0.5240±0.0174 | 0.4495±0.0119 | 0.6414±0.0519 | 0/5 |
| ra_resp_top3_mean | 0.7788±0.0043 | 0.3542±0.0191 | 0.7076±0.0074 | 0.4301±0.0342 | 0/5 |
| ra_resp_q90 | 0.7793±0.0041 | 0.3590±0.0194 | 0.7160±0.0078 | 0.4339±0.0320 | 0/5 |

## Top 15 by mean AUC

| rank | signal | AUC mean±std | AP mean±std | AUC > margin |
|---:|---|---:|---:|---:|
| 1 | ra_anom_ratio_diagnostic | 0.9328±0.0005 | 0.7722±0.0050 | 5/5 |
| 2 | mat_mean | 0.8009±0.0203 | 0.5335±0.0621 | 3/5 |
| 3 | ra_resp_mean | 0.7957±0.0108 | 0.5445±0.0143 | 3/5 |
| 4 | margin | 0.7952±0.0071 | 0.5163±0.0221 | - |
| 5 | mat_high08_ratio | 0.7878±0.0232 | 0.4708±0.0668 | 2/5 |
| 6 | ra_resp_high08_ratio | 0.7870±0.0061 | 0.4002±0.0175 | 0/5 |
| 7 | ra_resp_q75 | 0.7861±0.0062 | 0.4021±0.0112 | 0/5 |
| 8 | mat_q90 | 0.7811±0.0025 | 0.3679±0.0251 | 0/5 |
| 9 | ra_resp_q90 | 0.7793±0.0041 | 0.3590±0.0194 | 0/5 |
| 10 | ra_resp_top3_mean | 0.7788±0.0043 | 0.3542±0.0191 | 0/5 |
| 11 | mat_entropy | 0.7777±0.0296 | 0.5024±0.0689 | 1/5 |
| 12 | mat_top5_mean | 0.7731±0.0043 | 0.3441±0.0264 | 0/5 |
| 13 | ra_resp_entropy | 0.7704±0.0089 | 0.5135±0.0248 | 0/5 |
| 14 | ra_resp_min | 0.7662±0.0146 | 0.5240±0.0174 | 0/5 |
| 15 | ra_resp_max | 0.7601±0.0057 | 0.2792±0.0161 | 0/5 |

## Per-seed focus AUC/AP

| seed | margin AUC/AP | mat_mean AUC/AP | mat_entropy AUC/AP | mat_high08_ratio AUC/AP |
|---:|---:|---:|---:|---:|
| 0 | 0.7938/0.5510 | 0.8200/0.5963 | 0.8058/0.5771 | 0.8052/0.5361 |
| 1 | 0.7960/0.5194 | 0.7901/0.5804 | 0.7763/0.5641 | 0.7955/0.5292 |
| 2 | 0.7991/0.5097 | 0.7711/0.4407 | 0.7299/0.4132 | 0.7477/0.3757 |
| 3 | 0.7840/0.4904 | 0.8060/0.5092 | 0.7780/0.4620 | 0.7894/0.4374 |
| 4 | 0.8030/0.5110 | 0.8170/0.5410 | 0.7986/0.4959 | 0.8013/0.4755 |

## Interpretation

- Margin baseline: AUC 0.7952±0.0071, AP 0.5163±0.0221.
- `mat_mean`: AUC 0.8009±0.0203, AP 0.5335±0.0621, beats margin on 3/5 seeds; mean top5 Jaccard 0.526.
- `mat_entropy`: AUC 0.7777±0.0296, AP 0.5024±0.0689, beats margin on 1/5 seeds.
- `mat_high08_ratio`: AUC 0.7878±0.0232, AP 0.4708±0.0668, beats margin on 2/5 seeds.
- Diagnostic conclusion: route2 is useful for explanation/complementarity but not stable enough yet to promote directly as a method component.
