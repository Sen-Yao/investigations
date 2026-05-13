# Reference Response Distribution — seed0 analysis

## Protocol

No-training Phase-2 diagnostic. Existing DualRefGAD reference construction and frozen/random-init VecGAD token encoder path are reused. Labels are diagnostic-only.

## Main result

Route 2 is **worth continuing** on seed0.

The scalar margin baseline is strong:

- Margin AUC/AP: **0.7938 / 0.5510**
- Margin top1/top5 anomaly ratio: **0.995 / 0.768**

But multi-reference response distribution contains additional signal:

- `mat_mean` AUC/AP: **0.8200 / 0.5963**
- `mat_mean` vs margin Spearman: **0.708**
- `mat_mean` top5 Jaccard with margin: **0.705**
- `mat_mean` top5 anomaly ratio: **0.839** vs margin **0.768**

This passes the route-2 gate: AUC > 0.70, AP above margin, and ranking/top-k not identical to margin.

## Signal table

| signal | AUC | AP | ρ vs margin | top5 Jaccard | top1 | top5 | Δmean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ra_anom_ratio_diagnostic` | 0.9322 | 0.7741 | 0.231 | 0.545 | 0.975 | 0.938 | 0.6381 |
| `mat_mean` | 0.8200 | 0.5963 | 0.708 | 0.705 | 0.995 | 0.839 | 0.2778 |
| `mat_entropy` | 0.8058 | 0.5771 | 0.602 | 0.659 | 0.982 | 0.830 | 0.0588 |
| `mat_high08_ratio` | 0.8052 | 0.5361 | 0.709 | 0.659 | 0.839 | 0.821 | 0.3451 |
| `margin` | 0.7938 | 0.5510 | 1.000 | 1.000 | 0.995 | 0.768 | 0.3021 |
| `ra_resp_mean` | 0.7873 | 0.5604 | 0.849 | 0.761 | 1.000 | 0.786 | 0.3645 |
| `mat_min` | 0.7861 | 0.5575 | 0.322 | 0.653 | 0.993 | 0.823 | 0.6862 |
| `ra_resp_q75` | 0.7845 | 0.4207 | 0.801 | 0.612 | 0.620 | 0.697 | 0.2440 |
| `mat_q90` | 0.7825 | 0.4036 | 0.612 | 0.420 | 0.649 | 0.604 | 0.0595 |
| `ra_resp_high08_ratio` | 0.7807 | 0.4074 | 0.754 | 0.438 | 0.611 | 0.630 | 0.3851 |
| `mat_positive_ratio` | 0.7805 | 0.2846 | 0.475 | 0.161 | 0.373 | 0.367 | 0.1015 |
| `ra_pair_cos_mean` | 0.7198 | 0.3616 | 0.198 | 0.474 | 0.697 | 0.631 | 0.0573 |
| `ra_centroid_dist_mean` | 0.2790 | 0.0635 | -0.201 | 0.000 | 0.029 | 0.030 | -1.6704 |

## Candidate signals passing gate

```json
[
  {
    "signal": "ra_anom_ratio_diagnostic",
    "auc": 0.9322117218895682,
    "ap": 0.7741488636453064,
    "spearman_with_margin": 0.23083237381418367,
    "top5_jaccard_with_margin": 0.5445337059028991,
    "top1_ratio": 0.9751131221719457,
    "top5_ratio": 0.9380370872908186
  },
  {
    "signal": "mat_mean",
    "auc": 0.8199782421984899,
    "ap": 0.5963057223372972,
    "spearman_with_margin": 0.7076649223871567,
    "top5_jaccard_with_margin": 0.7047031611410949,
    "top1_ratio": 0.995475113122172,
    "top5_ratio": 0.8385345997286295
  },
  {
    "signal": "mat_entropy",
    "auc": 0.8058028222138455,
    "ap": 0.5770615469458463,
    "spearman_with_margin": 0.6019242519595817,
    "top5_jaccard_with_margin": 0.6586646661665416,
    "top1_ratio": 0.9819004524886877,
    "top5_ratio": 0.8299412030755314
  },
  {
    "signal": "mat_high08_ratio",
    "auc": 0.805239515207261,
    "ap": 0.5361303147574905,
    "spearman_with_margin": 0.7092910340864427,
    "top5_jaccard_with_margin": 0.6586646661665416,
    "top1_ratio": 0.8393665158371041,
    "top5_ratio": 0.8208955223880597
  },
  {
    "signal": "ra_resp_mean",
    "auc": 0.7872603936716007,
    "ap": 0.5604448214887894,
    "spearman_with_margin": 0.8485241086875303,
    "top5_jaccard_with_margin": 0.7610513739545998,
    "top1_ratio": 1.0,
    "top5_ratio": 0.7860696517412935
  },
  {
    "signal": "mat_min",
    "auc": 0.7861200918463829,
    "ap": 0.5575308144881934,
    "spearman_with_margin": 0.322029165822653,
    "top5_jaccard_with_margin": 0.6530841121495327,
    "top1_ratio": 0.9932126696832579,
    "top5_ratio": 0.8231569425599277
  },
  {
    "signal": "ra_resp_q75",
    "auc": 0.7845023722892945,
    "ap": 0.4207055498922212,
    "spearman_with_margin": 0.8008744079397008,
    "top5_jaccard_with_margin": 0.6121035362741524,
    "top1_ratio": 0.6199095022624435,
    "top5_ratio": 0.6965174129353234
  },
  {
    "signal": "mat_q90",
    "auc": 0.7825128645209984,
    "ap": 0.40359392760606183,
    "spearman_with_margin": 0.6122867472817313,
    "top5_jaccard_with_margin": 0.4204946996466431,
    "top1_ratio": 0.6493212669683258,
    "top5_ratio": 0.6037991858887382
  },
  {
    "signal": "ra_resp_high08_ratio",
    "auc": 0.7807294828645092,
    "ap": 0.40735472356369606,
    "spearman_with_margin": 0.7540391096228111,
    "top5_jaccard_with_margin": 0.4375812743823147,
    "top1_ratio": 0.6108597285067874,
    "top5_ratio": 0.6295793758480326
  },
  {
    "signal": "mat_positive_ratio",
    "auc": 0.7805116534406146,
    "ap": 0.2845898710006168,
    "spearman_with_margin": 0.4748725244248391,
    "top5_jaccard_with_margin": 0.16093462851142032,
    "top1_ratio": 0.3733031674208145,
    "top5_ratio": 0.36680235187697874
  }
]
```

## Interpretation

1. **Mean-pooled scalar margin is losing information.** `mat_mean` averages the full normal-anchor × anomaly-ref response matrix `M[i,j] = cos(h-rn_i, ra_j-rn_i)`. It improves seed0 AUC from ~0.794 to ~0.820 and AP from ~0.551 to ~0.596.
2. **The improvement is not just monotonic margin calibration.** `mat_mean` Spearman with margin is ~0.708 and top5 Jaccard is ~0.705, so it changes ranking materially.
3. **Distribution shape matters.** `mat_entropy`, `mat_high08_ratio`, `mat_min`, and `mat_q90` also pass the independence gate. This supports the hypothesis that response distribution contains non-collapsed signal.
4. **Diagnostic label-purity upper bound is very strong but not deployable.** `ra_anom_ratio_diagnostic` AUC ~0.932 is label-based diagnostic only; it shows that target-conditional `R_a` purity is highly explanatory, but cannot be used directly.
5. **The deployable direction is likely matrix-response scoring**, not another learned residual head. Candidate fixed formulas should combine margin with response-matrix summary stats, especially `mat_mean`/entropy/high-ratio.

## Next decision

Do not open a formal sweep yet. First run seeds 1-4 for the no-training response distribution diagnostic under a monitored/runner-compatible path. If `mat_mean` remains consistently above margin and top-k overlap remains non-identical, open a new method-validation investigation for a fixed no-head score.
