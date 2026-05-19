# Insights

## Seed0 result — inconclusive / weak positive diagnostic, not a method promotion

The seed0 normal-only Matrix AE probe completed successfully.

| item | value |
|---|---:|
| decision | `INCONCLUSIVE` |
| strongest scalar baseline | `neg_mat_mean` |
| scalar AUC / AP | 0.6377 / 0.1405 |
| best AE | `ae_mse_latent8` |
| AE AUC / AP | 0.6028 / 0.1140 |
| AE ΔAUC vs scalar best | -0.0349 |
| AE Spearman vs margin | -0.4560 |
| AE top5 Jaccard vs margin | 0.0036 |

### Interpretation

- Matrix AE has some signal (AUC≈0.603), but it does **not** beat simple scalar matrix orientation on seed0.
- The strongest scalar here is negative matrix mean, not margin, suggesting orientation/regime effects remain important.
- AE has very low top-k overlap with margin, so it may be complementary, but the effect size is currently too weak to promote.
- Next best step is not a full method sweep yet. Recommended follow-up: compare seed1/2 cheaply or inspect degree/rejection/regime stratification before adding model capacity.

### Conclusion boundary

This is a runner-registered seed0 diagnostic. It is valid as exploratory evidence, not final scientific validation.
