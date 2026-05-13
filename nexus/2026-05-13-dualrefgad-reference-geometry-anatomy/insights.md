# Insights — DualRefGAD Reference Geometry Anatomy

## Initial position

The previous additive residual probe closed the `margin + correction` route. The correction head learned margin compression/calibration, not baseline-independent anomaly ranking signal.

This investigation therefore starts from a stricter premise: **do not add a learned head until margin/reference geometry has been fully dissected.**

## Working intuition

The useful signal likely lives in reference geometry itself. The unknown is whether scalar margin already captures almost all of it, or whether the response vector over multiple references contains distributional inconsistency that scalar margin collapses away.

## Expected decision output

The investigation should end with a decision table, not just plots:

| Route | Continue? | Evidence |
|---|---|---|
| reference construction | TBD | |
| normal-manifold deviation | TBD | |
| multi-reference distributional inconsistency | TBD | |
| learned residual head | No | Closed by prior investigation |

## Current status

Created. Phase 1 anatomy script and data export are pending.

## 2026-05-13 — Seed0 no-training reference geometry anatomy

Artifacts:
- `experiments/scripts/reference_geometry_anatomy.py`
- `experiments/outputs/reference_geometry_anatomy_s0.summary.json`
- `experiments/outputs/reference_geometry_anatomy_s0.per_node.csv`
- `experiments/outputs/reference_geometry_anatomy_s0.arrays.npz`
- `experiments/outputs/reference_geometry_anatomy_seed0_analysis.md`

### Key numbers

- Margin test AUC/AP: **0.7938 / 0.5510**.
- Margin top1/top5 anomaly ratio: **0.995 / 0.768**.
- Normal refs purity: **1.000**.
- Anomaly refs global purity: **0.107**.
- Anomaly refs purity on anomaly target nodes: **0.684**.

### Interpretation

Seed0 supports a target-conditional reference-purity story: `R_a` is not globally clean, but when the target is a true anomaly, selected anomaly-side references become much more anomaly-enriched. That makes the scalar margin strong without implying that a learned residual head has independent signal.

### Process note

This was a manual no-training diagnostic probe, not a runner-compliant formal experiment. It is acceptable only as Phase-1 debugging/anatomy evidence. Future multi-seed diagnostics should either be registered through `experiment-runner` (`probe`/`single-run` path if available) or paired with an explicit Hermes watchdog so completion/failure is reported promptly.

## 2026-05-13 — Route 2 seed0: multi-reference response distribution

Artifacts:
- `experiments/scripts/reference_response_distribution.py`
- `experiments/outputs/reference_response_distribution_s0.summary.json`
- `experiments/outputs/reference_response_distribution_s0.per_node.csv`
- `experiments/outputs/reference_response_distribution_s0.arrays.npz`
- `experiments/outputs/reference_response_distribution_seed0_analysis.md`

### Key result

Route 2 is positive on seed0. `mat_mean`, the mean of the full normal-anchor × anomaly-ref response matrix, improves over scalar margin:

- Margin AUC/AP: **0.7938 / 0.5510**.
- `mat_mean` AUC/AP: **0.8200 / 0.5963**.
- `mat_mean` Spearman vs margin: **0.708**.
- `mat_mean` top5 Jaccard vs margin: **0.705**.
- `mat_mean` top5 anomaly ratio: **0.839** vs margin **0.768**.

### Interpretation

Scalar mean-pooled margin is not sufficient: the full response matrix contains ranking signal that is both stronger on seed0 and not rank-identical to margin. This supports continuing `multi-reference distributional inconsistency` as Phase 2.

Diagnostic-only `ra_anom_ratio_diagnostic` is a strong upper-bound/explanatory variable (AUC **0.9322**) but is not deployable because it uses labels. It confirms that target-conditioned `R_a` purity explains much of the mechanism.

### Decision

Run seeds 1-4 for this no-training diagnostic before designing a fixed formula. If `mat_mean` remains stable, open a method-validation investigation for no-head response-matrix scoring.

