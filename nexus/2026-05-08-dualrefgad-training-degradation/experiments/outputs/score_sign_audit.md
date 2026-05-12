# Score Sign / Semantic Audit

## Setup

Diagnostic reruns on HCCS-88 using modified output schema:

- seed 0, epoch 50
- seed 3, epoch 50
- `head_mode=dual_margin_two_score`
- `val_rate=0.0`
- `wandb=false`

Files:

- `sign_audit_seed0_epoch50.json`
- `sign_audit_seed3_epoch50.json`

## Seed 0

| score | AUC | AP |
|---|---:|---:|
| margin | 0.7938 | 0.5510 |
| -normality_logit (= final anomaly score) | 0.7518 | 0.2174 |
| sa_minus_sn | 0.7334 | 0.1963 |
| sa | 0.7252 | 0.1972 |
| -sn | 0.7215 | 0.1766 |
| sn | 0.2785 | 0.0641 |
| -sa | 0.2748 | 0.0638 |
| sn_minus_sa | 0.2666 | 0.0632 |
| normality_logit | 0.2482 | 0.0617 |
| -margin | 0.2062 | 0.0574 |

## Seed 3

| score | AUC | AP |
|---|---:|---:|
| margin | 0.7840 | 0.4904 |
| -normality_logit (= final anomaly score) | 0.7131 | 0.1663 |
| sa_minus_sn | 0.6737 | 0.1410 |
| -sn | 0.6639 | 0.1370 |
| sa | 0.6631 | 0.1360 |
| -sa | 0.3369 | 0.0689 |
| sn | 0.3361 | 0.0691 |
| sn_minus_sa | 0.3263 | 0.0677 |
| normality_logit | 0.2869 | 0.0644 |
| -margin | 0.2160 | 0.0579 |

## Interpretation

### 1. Final sign is not globally flipped

`-normality_logit` is much better than `normality_logit`:

- seed0: 0.7518 vs 0.2482
- seed3: 0.7131 vs 0.2869

Therefore, the final evaluation sign is directionally correct.

### 2. The margin term alone is much stronger than the learned final score

The strongest score is `margin`:

- seed0: margin AUC 0.7938 vs final AUC 0.7518
- seed3: margin AUC 0.7840 vs final AUC 0.7131

This is critical: the hand-crafted geometric margin contains most of the useful ranking signal, and the learned `sn/sa` terms degrade it.

### 3. Learned sublogits are weaker and can dilute the margin

`sa_minus_sn` is lower than margin:

- seed0: 0.7334
- seed3: 0.6737

Final anomaly score is:

```text
-final = sa - sn + margin
```

If `sa - sn` is noisy or misaligned, it can damage the strong margin ranking.

### 4. Seed3 is especially damaged by learned terms

Margin remains strong for seed3:

```text
margin AUC = 0.7840
```

But final drops to:

```text
final AUC = 0.7131
```

So seed3 is not lacking the reference geometric signal; the learned score terms damage that signal more strongly.

## Conclusion

H2 score sign mismatch is only partially supported:

- final sign is correct;
- but component semantics are mismatched: `margin` is the reliable anomaly score, while learned `sn/sa` terms reduce performance.

Updated root cause:

```text
The current head combines a strong geometric margin with learned sn/sa terms that are not aligned with anomaly ranking. Training makes those learned terms influential enough to degrade the final score.
```

## Next step

The next probe should test constrained alternatives:

1. margin-only score;
2. margin + learned scalar calibration;
3. margin + small residual regularized toward zero;
4. full `sa - sn + margin` as current control.
