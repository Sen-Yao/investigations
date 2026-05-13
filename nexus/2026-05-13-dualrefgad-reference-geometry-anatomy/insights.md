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
