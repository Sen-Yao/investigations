# Stage-3 Residual ABCD Diagnostics — Result Analysis

> Sweep: `vtkl5ykv`  
> Job: `exp_20260513_142903_dualrefgad_stage3_residual_abcd_diagnost`  
> WandB: <https://wandb.ai/HCCS/DualRefGAD/sweeps/vtkl5ykv>  
> Dataset: `elliptic`  
> Seeds: `[0,1,2,3,4]`  
> Status: 5/5 finished, 0 failed

## Executive conclusion

The residual correction route should be treated as a **negative diagnostic finding**. The learned correction does train, but it does not discover a stable baseline-independent anomaly ranking signal. The final score is almost a monotonic transform of the original margin:

```text
score = margin + corr ≈ a + 0.788 * margin + small_residual
```

AUC changes by only `6.29e-05±2.72e-04` and AP changes by `-4.46e-04±2.50e-03`. This is far below a meaningful 5-seed effect and is directionally unstable for AP.

## Main performance table

| Metric | margin-only | score = margin + corr | delta |
|---|---:|---:|---:|
| AUC | 0.7952±0.0071 | 0.7953±0.0071 | 6.29e-05±2.72e-04 |
| AP | 0.5165±0.0220 | 0.5161±0.0221 | -4.46e-04±2.50e-03 |

**Insight:** the mean AUC delta is technically positive but only `6.3e-05`; AP is negative on average. This is not a real improvement. The correction route does not pass the pre-defined stop-loss rule.

## ABCD diagnosis

| Hypothesis | Evidence | Verdict |
|---|---|---|
| A. Global shift / calibration | `corr_mean=-0.1767±0.0037`, `corr_std=0.1073±0.0015`; mean magnitude dominates node-specific variation. | Raised |
| B. Margin-linked / monotonic | `spearman_corr_margin=-0.9957±0.0010`, `R²(corr~margin)=0.8455±0.0060`. | Raised |
| C. Too weak to change ranking | `corr_std/margin_std=0.2305±0.0055`, `rank_flip_rate=0.0086±0.0008`, `top5_jaccard=0.9996±0.0005`. | Raised |
| D. No anomaly separation | `corr_cohen_d=-0.5256±0.0204`, `neg_corr_auc=0.7910±0.0068`. Correction has some class-correlated signal, but it is anti-oriented as `corr` and mostly mirrors margin. | Not the primary failure |

## Ranking geometry

| Metric | 5-seed mean±std | Interpretation |
|---|---:|---|
| `spearman_score_margin` | 0.9985±0.0002 | Final score ranking is almost identical to margin. |
| `pearson_score_margin` | 0.9935±0.0003 | Final score is near-linear in margin. |
| `top1_jaccard_margin_score` | 0.9753±0.0298 | Top-1% candidates mostly unchanged. |
| `top5_jaccard_margin_score` | 0.9996±0.0005 | Top-5% candidates are essentially identical. |
| `rank_flip_rate_sampled` | 0.0086±0.0008 | Fewer than 1% sampled node pairs flip ranking. |

**Insight:** even when the correction is nonzero, it does not move the candidate set. This explains why AUC/AP remain essentially unchanged.

## Linear decomposition

Using the diagnostic approximation:

```text
corr ≈ a + b * margin + residual
b = Pearson(corr, margin) * std(corr) / std(margin)
```

we get:

- `b ≈ -0.212`
- `score slope = 1 + b ≈ 0.788`
- `residual std / margin std ≈ 0.091`

This means the correction mostly compresses margin instead of adding independent evidence. Because the score slope remains positive, the final ranking is preserved; because the residual term is small, it cannot rescue top-k ordering.

## Per-seed table

| seed | margin_auc | score_auc | ΔAUC | margin_ap | score_ap | ΔAP | score-margin ρ | top5 Jaccard |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7938 | 0.7938 | +4.46e-05 | 0.5510 | 0.5512 | +1.38e-04 | 0.9987 | 1.0000 |
| 1 | 0.7960 | 0.7963 | +2.51e-04 | 0.5192 | 0.5164 | -2.81e-03 | 0.9982 | 1.0000 |
| 2 | 0.7991 | 0.7987 | -4.02e-04 | 0.5112 | 0.5144 | +3.19e-03 | 0.9984 | 1.0000 |
| 3 | 0.7840 | 0.7843 | +2.07e-04 | 0.4904 | 0.4905 | +5.70e-05 | 0.9987 | 0.9991 |
| 4 | 0.8030 | 0.8032 | +2.14e-04 | 0.5108 | 0.5079 | -2.81e-03 | 0.9983 | 0.9991 |

## Scientific interpretation

The normal-only residual objective successfully learns a correction, but the easiest solution under the current loss is to suppress high normal scores in a way that is strongly tied to the margin itself. This creates calibration and margin compression, not a new anomaly signal. The result reconciles with the earlier observation that margin-only is strong while learned heads degrade or fail to improve: the dual-reference margin already captures the useful ranking geometry available to this head, and the learned residual mostly reparameterizes it.

This does **not** mean all reference-learning directions are dead. It specifically falsifies the additive residual patch as a final method narrative. Future work should not keep tuning this correction head. If we continue, the mechanism must be redesigned around reference construction, residualized/orthogonal evidence, or a different normal-manifold inconsistency score.

## Decision

Close the additive residual probe route as a negative result under the current setup. Do not present `margin + correction` as the method. Use this experiment as evidence that a learnable head on top of margin is mostly surface-level calibration.
