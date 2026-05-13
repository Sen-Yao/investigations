# Insights — DualRefGAD Normal-only Residual Probe

## Initial framing

This investigation exists because the earlier DualRefGAD evidence points in two directions at once. The fixed dual-reference margin contains real ranking signal, but learned heads can degrade it. That contradiction should not be resolved by adding another unconstrained head; it should be resolved by a diagnostic probe that asks whether the baseline leaves residual signal under a protocol-clean normal-only setup.

## Diagnostic-only status

The additive residual form is not considered elegant enough to be the final method narrative. Its role is diagnostic: test whether margin-only has a stable blind spot. If the probe works, the output should be inspected and reformulated; if it does not work, the negative result should stop this branch.

## Interpretation rule

A successful result requires stable improvement and evidence of ranking-geometry change. A correction that simply suppresses known-normal scores is valid training behavior, but it is not a new anomaly detection principle unless it changes anomaly ranking in a stable way.

## Relation to prior investigation

`2026-05-09-semisupervised-negative-signal-for-dualrefgad` established that pseudo-negative or contrastive signals need a theoretical basis. This investigation narrows that lesson into a concrete diagnostic route: residual signal must be grounded in normal-manifold deviation or reference inconsistency, not in arbitrary proxy compatibility.


## Final result — additive residual route closed

The 5-seed Stage-3 ABCD diagnostic on `elliptic` finished under sweep `vtkl5ykv`. The learned correction does not provide stable improvement over margin-only:

| Metric | margin-only | score = margin + corr | delta |
|---|---:|---:|---:|
| AUC | `0.7952±0.0071` | `0.7953±0.0071` | `6.29e-05±2.72e-04` |
| AP | `0.5165±0.0220` | `0.5161±0.0221` | `-4.46e-04±2.50e-03` |

The important result is not only “no gain”; it is the geometry of the failure:

- `spearman(score, margin)=0.9985±0.0002`: final ranking is almost identical to margin.
- `top5_jaccard=0.9996±0.0005`: top candidate set is essentially unchanged.
- `spearman(corr, margin)=-0.9957±0.0010`, `R²(corr~margin)=0.8455±0.0060`: correction is mostly a function of margin.
- Linear decomposition gives `score ≈ a + 0.788 * margin + small_residual`; the residual component is too small to rescue ranking.

### Scientific interpretation

The normal-only objective can learn to suppress high normal scores, but the easiest solution is margin compression / calibration. This is not a new anomaly detection principle. It matches the broader pattern from the earlier investigation: the useful signal is already in dual-reference margin geometry, while learned heads tend to reparameterize or degrade it.

### Decision

Close the additive residual route. Do not present `margin + correction` as the method. If we continue DualRefGAD, the next mechanism should target reference construction, normal-manifold deviation, or multi-reference distributional inconsistency directly.
