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

