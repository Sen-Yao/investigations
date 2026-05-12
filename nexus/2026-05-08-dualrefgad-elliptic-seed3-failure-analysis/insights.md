# Insights

_Status: first-pass diagnosis completed 2026-05-08._

## Initial observation

Protocol-clean no-val 5-seed result:

- AUC: 0.7455±0.0188
- AP: 0.2011±0.0238

Seed 3 is the main low outlier:

- seed 3 AUC: 0.7131
- other seeds AUC range: 0.7460 - 0.7601

## First-pass diagnosis

Source files:

- `experiments/outputs/wandb_runs_summary.json`
- `experiments/outputs/wandb_histories.json`
- `experiments/outputs/seed3_diagnosis.json`
- `experiments/outputs/seed3_first_pass_report.md`

### 1. Seed 3 is a real low outlier

Compared with the non-seed3 mean:

| metric | seed3 | non3 mean | delta |
|---|---:|---:|---:|
| final_test_auc | 0.7131 | 0.7536 | -0.0404 |
| final_test_ap | 0.1663 | 0.2099 | -0.0436 |

This confirms seed 3 is materially worse than the other four seeds.

### 2. Reference purity does **not** explain seed 3

Reference purity metrics are almost identical across seeds:

| metric | seed3 | non3 mean | delta |
|---|---:|---:|---:|
| normal_ref_normal_ratio | 1.0000 | 1.0000 | 0.0000 |
| anom_ref_anom_ratio | 0.1081 | 0.1074 | +0.0007 |
| anom_ref_anom_ratio_on_anom_nodes | 0.6795 | 0.6779 | +0.0016 |

Interpretation:

- H2 (worse normal reference purity) is not supported.
- H3 (worse deviation reference purity) is not supported by current summary-level metrics.

### 3. Score/margin collapse is not the primary explanation

Seed 3 has somewhat lower `final_score_std` and `final_sa_std`, but not a severe collapse:

| metric | seed3 | non3 mean | delta |
|---|---:|---:|---:|
| final_score_std | 2.7511 | 3.1668 | -0.4157 |
| final_sn_std | 1.6091 | 1.6844 | -0.0753 |
| final_sa_std | 1.4671 | 1.7532 | -0.2861 |
| final_margin_std | 0.4819 | 0.4699 | +0.0120 |

Interpretation:

- H4 is only weakly supported.
- There is no obvious margin collapse; `final_margin_std` is slightly higher than the non-seed3 mean.
- Lower `score_std/sa_std` may contribute, but does not by itself explain the full AUC drop.

### 4. Final epoch degradation is strongly supported

All seeds peak at epoch 0 in the logged `test_auc/test_ap` history, then degrade by epoch 50.

| seed | peak AUC | final AUC | AUC drop |
|---:|---:|---:|---:|
| 0 | 0.8048 | 0.7518 | 0.0531 |
| 1 | 0.7997 | 0.7601 | 0.0396 |
| 2 | 0.7831 | 0.7564 | 0.0266 |
| 3 | 0.7826 | 0.7131 | 0.0695 |
| 4 | 0.7798 | 0.7460 | 0.0338 |

Seed 3 has the largest AUC degradation:

```text
seed3 AUC drop = 0.0695
non3 drops = 0.0266 - 0.0531
```

Interpretation:

- H5 is strongly supported.
- The core problem is likely not reference construction quality, but the training objective/head dynamics degrading a strong initial reference signal.
- Seed 3 is the most sensitive split to this degradation.

### 5. Important methodological implication

Epoch 0 already has high AUC because fixed dual-reference evidence is informative before training:

```text
seed0 epoch0 AUC 0.8048
seed1 epoch0 AUC 0.7997
seed2 epoch0 AUC 0.7831
seed3 epoch0 AUC 0.7826
seed4 epoch0 AUC 0.7798
```

Training the current `dual_margin_two_score` head for 50 epochs reduces AUC for every seed.

This suggests the current objective is not preserving the initial reference ranking signal. The next research focus should be objective design / score calibration, not merely reference purity.

## Current conclusion

Seed 3 should be kept in the reported results. It is not an obvious data/reference bug. It is a robustness signal showing that the current head/objective can over-train or distort a strong initial reference signal.

## Updated hypothesis status

| Hypothesis | Status | Notes |
|---|---|---|
| H1 split structurally harder | open | Needs split/graph analysis |
| H2 normal reference worse | not supported | normal_ref_normal_ratio identical |
| H3 deviation reference worse | not supported at summary level | purity metrics nearly identical |
| H4 score/margin collapse | weakly supported | lower score/sa std, no margin collapse |
| H5 final epoch degradation | strongly supported | seed3 largest degradation |
| H6 expected variance | partially supported | low seed reflects objective sensitivity |

## Recommended next actions

1. Run split/graph structure analysis for H1.
2. Extract per-node final scores for seeds 0 and 3 to compare ranking errors.
3. Design an objective-preserving probe:
   - freeze reference score;
   - train only calibration layer;
   - compare no-training / short-training / full-training.
4. Do **not** use epoch-0/best-test as formal metric; use it only as diagnostic evidence.

---

## Split / graph structure analysis (H1)

Source files:

- `experiments/scripts/analyze_split_graph_seed3.py`
- `experiments/outputs/split_graph_analysis.json`
- `experiments/outputs/split_graph_analysis.md`

### Summary

| seed | train normal | train anomaly in raw train | test anomaly | test anomaly rate | anomaly 1-hop to train normal | anomaly 2-hop to train normal | anomaly centroid distance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2086 | 242 | 4303 | 0.0973 | 0.0216 | 0.0720 | 6.0371 |
| 1 | 2097 | 231 | 4314 | 0.0975 | 0.0183 | 0.0795 | 5.8037 |
| 2 | 2091 | 237 | 4308 | 0.0974 | 0.0186 | 0.0699 | 5.7853 |
| 3 | 2099 | 229 | 4316 | 0.0976 | 0.0127 | 0.0767 | 5.8533 |
| 4 | 2108 | 220 | 4325 | 0.0978 | 0.0173 | 0.0726 | 5.8446 |

### Key finding

Seed 3 is **not** globally harder by simple split composition:

- anomaly rate is nearly identical across seeds;
- test anomaly degree is normal;
- feature distance to train-normal centroid is normal;
- 2-hop coverage is normal.

The only mildly unusual signal is lower 1-hop contact between test anomalies and labeled train normals:

```text
seed3 test_anom_adjacent_to_train_normal_ratio = 0.0127
non3 mean = 0.0190
```

This may make seed 3 less directly supported by labeled-normal anchors, but it is not a complete explanation because 2-hop coverage remains normal:

```text
seed3 2-hop anomaly coverage = 0.0767
non3 mean = 0.0735
```

### H1 status

H1 is **weakly supported only at the local 1-hop level**, but not supported as a broad split difficulty explanation.

Combined with the previous WandB analysis, the stronger explanation remains:

```text
current training objective degrades a strong initial reference ranking signal;
seed 3 is more sensitive to that degradation, possibly because it has fewer direct 1-hop anomaly-to-train-normal contacts.
```

## Updated conclusion after A

The diagnosis now points away from data leakage / reference purity / gross split imbalance.

Most plausible mechanism:

1. fixed dual-reference signal is already strong at epoch 0;
2. current `dual_margin_two_score` training objective distorts that ranking over 50 epochs;
3. seed 3 has slightly weaker direct graph support from labeled train normals, making it more vulnerable to degradation.

Next best step is B: an objective-preserving probe.

---

## Handoff to training degradation investigation

After first-pass WandB analysis and split/graph structure analysis, seed 3 is no longer considered the root cause. It is a symptom of a broader training degradation problem.

Follow-up investigation:

```text
investigations/2026-05-08-dualrefgad-training-degradation/
```

Core handoff statement:

```text
Seed 3 is not the root cause; it is the most visible symptom of broader training degradation.
```
