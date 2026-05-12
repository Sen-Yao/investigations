# Insights

## 2026-05-06 Initial framing

The research focus has shifted from "why does training degrade?" to "why does pseudo anomaly generation fail to provide positive learning signal?"

Current evidence:

- Full train_all on Elliptic under `target_ref_guided` is weak: best_test_auc around 0.6836±0.0229.
- Frozen geometry + head-only is much stronger: best_test_auc 0.7514±0.0243.
- However, pseudo separability remains almost random: pseudo_auc 0.5154±0.0002.

Working interpretation:

```text
R_a/R_n reference geometry is useful as a ranking prior,
but converting it into local displacement positives
normal + beta * mean(R_a - R_n)
does not produce a reliable positive class for BCE learning.
```

The next step is a pseudo quality audit, not another performance sweep.

## 2026-05-06 D1/D2/D4/D6 beta-quality audit, seed=0

Mini audit sweep finished successfully:

- Job: `exp_20260506_103506_pseudo_quality_beta_seed0`
- Sweep: `72wxwvwi`
- WandB: https://wandb.ai/HCCS/VoxG/sweeps/72wxwvwi
- Runs: 5/5 finished, failed=0
- Dataset: Elliptic
- Seed: 0
- beta: [0.05, 0.1, 0.2, 0.4, 0.8]
- encode_batch_size: 256

### Key metrics

| beta | pseudo_auc | pseudo_margin | normal-pseudo L2 | pseudo→anom dist | pseudo→normal dist | pseudo kNN16 anomaly ratio | pseudo-real dir cos |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.4997 | -0.0000 | 0.0500 | 3.8464 | 0.0468 | 0.0338 | 0.0505 |
| 0.10 | 0.4995 | -0.0001 | 0.1000 | 3.8468 | 0.0903 | 0.0340 | 0.0585 |
| 0.20 | 0.4990 | -0.0001 | 0.2000 | 3.8513 | 0.1746 | 0.0350 | 0.0766 |
| 0.40 | 0.4979 | -0.0002 | 0.4000 | 3.8723 | 0.3378 | 0.0365 | 0.1066 |
| 0.80 | 0.4956 | -0.0005 | 0.8000 | 3.9462 | 0.6579 | 0.0405 | 0.1569 |

### Reference / geometry diagnostics

| metric | value |
|---|---:|
| R_a pairwise cosine | 0.9005 |
| R_n pairwise cosine | 0.8687 |
| mean direction vs individual direction cosine | 0.5849 |
| mean direction norm | 2.8629 |
| ref_delta_dist_auc | 0.6089 |
| ref_delta_dist_ap | 0.1141 |
| ref_cos_delta_auc | 0.4930 |
| ref_cos_delta_ap | 0.0914 |
| epoch0_auc | 0.4877 |
| epoch0_ap | 0.0863 |

### Interpretation

1. Current pseudo positives are not separable from normal positives under the random/frozen head: `pseudo_auc≈0.50` for all beta values.
2. Increasing beta increases the geometric displacement exactly as expected (`normal-pseudo L2 = beta`), but does not improve pseudo separability. In fact pseudo_auc slightly decreases as beta grows.
3. Pseudo samples remain much closer to normal manifold than anomaly manifold. Even at beta=0.8:
   - pseudo→normal distance = 0.6579
   - pseudo→anomaly distance = 3.9462
   - kNN16 anomaly ratio around pseudo = 0.0405
4. Pseudo-real direction alignment is weak. It increases with beta only because pseudo moves further along the chosen direction, but remains low (`0.05 → 0.16`).
5. This strongly supports the hypothesis that local displacement pseudo generation does not move normal nodes into the real anomaly manifold.
6. `R_a` references themselves are highly self-similar (`pairwise cosine≈0.90`), so the immediate problem is not obvious multi-modal cancellation in this seed/config. The larger issue is that the generated point remains near the normal manifold.
7. A simple reference score `dist_to_Rn - dist_to_Ra` reaches AUC 0.6089, above random but below head-only best-test. `cos(Ra)-cos(Rn)` is near random.

### Conclusion

For Elliptic seed=0, increasing beta makes pseudo samples farther from their source normals but still does not place them near real anomalies or create a useful positive class. The current pseudo generation behaves like a weak local displacement artifact rather than realistic anomaly synthesis.

Next step:

- Audit alternative pseudo construction strategies, especially interpolation/mixup toward individual R_a refs rather than mean-direction displacement.
- Compare local displacement vs direct reference scoring vs multi-direction pseudo quality before doing any training sweep.

## 2026-05-06 Exploration constraint: minimize hyperparameters

The next pseudo-generation exploration should avoid introducing new continuous hyperparameters where possible. The preferred direction is to test whether the retrieved reference structure itself can define positives or scores, rather than creating positives through tunable displacement.

This means prioritizing:

1. direct reference scoring,
2. using `R_a` references/prototypes directly as positive evidence,
3. only later considering tunable interpolation/displacement if zero-hyperparameter alternatives fail.

## 2026-05-06 Zero/few-hyperparameter strategy audit, seed=0

Strategy comparison sweep finished successfully:

- Job: `exp_20260506_110718_pseudo_quality_strategy_seed0`
- Sweep: `evzwj9jd`
- WandB: https://wandb.ai/HCCS/VoxG/sweeps/evzwj9jd
- Runs: 3/3 finished, failed=0
- Strategies: `local_displacement`, `ra_mean_positive`, `ra_individual_positive`

| strategy | pseudo_auc | pseudo_margin | normal-pseudo L2 | pseudo→anom dist | pseudo→normal dist | kNN16 anomaly ratio | pseudo-real dir cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| local_displacement | 0.4990 | -0.0001 | 0.2000 | 3.8513 | 0.1746 | 0.0350 | 0.0766 |
| ra_mean_positive | 0.5013 | -0.0005 | 2.8835 | 3.7090 | 1.6842 | 0.0338 | 0.1753 |
| ra_individual_positive | 0.4939 | -0.0011 | 4.2345 | 3.5840 | 0.0344 | 0.0426 | 0.5357 |

Interpretation:

1. None of the tested zero/few-hyperparameter positive strategies creates a separable pseudo task under the current random/frozen head: pseudo_auc remains around 0.5.
2. `ra_mean_positive` moves positives farther from source normals than local displacement, but still does not increase kNN anomaly ratio.
3. `ra_individual_positive` has high pseudo-real direction cosine because the positive is literally a retrieved R_a ref, but many R_a refs are still near the normal manifold in embedding space (`pseudo→normal dist` is extremely small). This suggests R_a retrieval is anomaly-informative for anomalous targets but contaminated/normal-like globally when used as positives for all normal sources.
4. Directly using R_a refs as positives is therefore not enough. A selector/gating mechanism may be needed, but this risks introducing hyperparameters unless based on existing reference scores.

Caveat:

- `pseudo_normal_anom_dist_ratio` for `ra_individual_positive` is numerically unstable because many retrieved refs have near-zero distance to some normal point. Use raw distances and kNN anomaly ratio instead.

Conclusion:

The failure is not only beta/local-displacement magnitude. Naively replacing generated positives with R_a prototypes or individual R_a refs also fails to define a clean positive class. The next low-hyperparameter direction should focus on score/calibration or reference-derived gating rather than treating all R_a refs as positive samples.

## Link to follow-up investigation

The pseudo anomaly generation diagnosis motivates a follow-up investigation:

```text
2026-05-06-fixed-dual-reference-guided-gt-learning
```

Key transition:

```text
Do not learn/update dual-reference and do not convert R_a into pseudo anomaly labels.
Treat dual-reference as fixed precomputed evidence, and study how GT should learn anomaly judgment from it under 5% labeled-normal supervision.
```
