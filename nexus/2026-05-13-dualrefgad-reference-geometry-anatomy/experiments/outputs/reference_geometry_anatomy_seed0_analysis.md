# Reference Geometry Anatomy — seed0 no-training diagnostic

Generated from `reference_geometry_anatomy_s0.summary.json`.

## Protocol

- No training, no WandB, no sweep.
- Existing DualRefGAD reference construction reused.
- Anomaly labels used only for post-hoc diagnostics.
- This run was initially launched manually as a diagnostic probe; future multi-seed/formal runs must be registered/launched through `experiment-runner` or paired with an explicit watchdog.

## Baseline margin

- Test AUC: **0.7938**
- Test AP: **0.5510**
- Top-1% anomaly ratio: **0.995**
- Top-5% anomaly ratio: **0.768**
- Mean margin normal/anomaly: 0.5358 / 0.8379

## Reference purity

```json
{
  "overall": {
    "normal_ref_normal_ratio": 1.0,
    "anom_ref_anom_ratio": 0.10741130487071557,
    "anom_ref_anom_ratio_on_anom_nodes": 0.6836908690869087
  },
  "normal_nodes": {
    "normal_ref_normal_ratio": 1.0,
    "anom_ref_anom_ratio": 0.04507782193769485,
    "anom_ref_anom_ratio_on_anom_nodes": 0.0
  },
  "anom_nodes": {
    "normal_ref_normal_ratio": 1.0,
    "anom_ref_anom_ratio": 0.6836908690869087,
    "anom_ref_anom_ratio_on_anom_nodes": 0.6836908690869087
  },
  "test_nodes": {
    "normal_ref_normal_ratio": 1.0,
    "anom_ref_anom_ratio": 0.10741675332308527,
    "anom_ref_anom_ratio_on_anom_nodes": 0.6834621194515454
  }
}
```

Key immediate observation: `normal_refs` are pure by construction (`normal_ref_normal_ratio=1.0`), while anomaly refs are globally weakly pure (**0.107**) but much purer for anomaly targets (**0.684**). This strongly suggests margin works partly because `R_a` becomes label-aligned specifically around true anomalies, not because the anomaly-reference pool is globally clean.

## Signal ranking on test split

| signal | AUC | AP | top1 | top5 | Δmean(anom-normal) |
|---|---:|---:|---:|---:|---:|
| `margin` | 0.7938 | 0.5510 | 0.995 | 0.768 | 0.3021 |
| `neg_anomaly_dist_emb` | 0.7239 | 0.4302 | 0.995 | 0.625 | 1.5011 |
| `neg_normal_dist_emb` | 0.5746 | 0.1061 | 0.000 | 0.003 | 1.2151 |
| `ga_score` | 0.4356 | 0.0794 | 0.014 | 0.030 | -0.0381 |
| `normal_dist_emb` | 0.4254 | 0.0779 | 0.045 | 0.034 | -1.2151 |
| `ref_gap_emb` | 0.4254 | 0.0787 | 0.038 | 0.037 | -0.8303 |
| `hop0_attr_ref_gap_l2` | 0.3901 | 0.0743 | 0.029 | 0.038 | -3.7681 |
| `hop0_attr_normal_dist_l2` | 0.3852 | 0.0735 | 0.011 | 0.062 | -1.2486 |
| `hop1_attr_ref_gap_l2` | 0.3774 | 0.0821 | 0.061 | 0.067 | -0.6438 |
| `hop1_attr_normal_dist_l2` | 0.3733 | 0.0793 | 0.052 | 0.047 | -0.5648 |
| `normal_dist_desc_l2` | 0.3488 | 0.0700 | 0.038 | 0.065 | -2.1057 |
| `ref_gap_desc_l2` | 0.3410 | 0.0695 | 0.038 | 0.038 | -4.9922 |
| `hop2_attr_normal_dist_l2` | 0.3299 | 0.0740 | 0.045 | 0.065 | -0.7072 |
| `hop1_attr_anom_dist_l2` | 0.3185 | 0.0750 | 0.061 | 0.070 | -0.8917 |
| `hop2_attr_ref_gap_l2` | 0.3127 | 0.0724 | 0.048 | 0.037 | -1.3633 |
| `anomaly_dist_emb` | 0.2761 | 0.0637 | 0.038 | 0.047 | -1.5011 |

## Components most correlated with margin

| component | Spearman vs margin | Pearson vs margin |
|---|---:|---:|
| `orthogonal_residual_emb` | -0.9045 | -0.6322 |
| `anomaly_dist_emb` | -0.3095 | -0.1759 |
| `ref_gap_emb` | 0.1502 | 0.2171 |
| `hop0_attr_anom_dist_l2` | -0.1500 | 0.0095 |
| `normal_dist_emb` | 0.1361 | 0.2026 |
| `anom_dist_desc_l2` | -0.1336 | 0.0196 |
| `ga_score` | 0.1234 | 0.1276 |
| `hop2_attr_anom_dist_l2` | -0.1154 | 0.0269 |
| `hop1_attr_anom_dist_l2` | -0.0990 | 0.0162 |
| `hop2_attr_ref_gap_l2` | -0.0600 | 0.0331 |
| `ref_gap_desc_l2` | -0.0437 | 0.0237 |
| `hop1_attr_ref_gap_l2` | -0.0415 | 0.0212 |

## Top-k failure-case pointers

- Top-5% margin anomaly ratio: **0.768**
- First false-positive node IDs: `[15956, 23295, 7612, 26587, 22287, 30973, 18347, 30476, 30390, 30582, 19374, 25883, 26098, 30414, 29948]`
- First low-margin missed anomaly IDs: `[35409, 4223, 3195, 30256, 42473, 23585, 23519, 23003, 22682, 24111, 23057, 36082, 15430, 26803, 7403]`

## Provisional interpretation

1. Margin remains a strong no-head signal on seed0 (AUC≈0.794/AP≈0.551), so the next step should not be another learned head.
2. The large gap between global `anom_ref_anom_ratio` and anomaly-target `anom_ref_anom_ratio_on_anom_nodes` points to **target-conditional reference purity** as a key mechanism.
3. Need to inspect whether false positives are nodes whose `R_a` is impure but geometrically aligned, or normal nodes near the same normal-manifold deviation direction.
4. Need at least seeds 1-4 or a runner-registered diagnostic before treating this as stable evidence.
