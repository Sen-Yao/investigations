# Insights — MatGuardGT

## Initial position

MatGuardGT is not an attempt to replace `mat_mean` blindly. It treats `mat_mean` as a strong positive control and weak high-confidence ranking teacher, while asking whether a tensor/entry-vector GT can learn structured corrections under normal-only supervision.

## Interpretation rules

- `mat_mean` is not ground truth; it is a guardrail / positive control.
- Full pointwise regression to `mat_mean` is not the first-stage objective.
- True anomaly labels are diagnostic-only and must not select epochs, loss weights, or hyperparameters.
- Multi-center ROCC is a tested ablation, not the default inductive commitment.
- If a variant improves only by copying `mat_mean`, report it as imitation, not a method contribution.
- If a variant diverges from `mat_mean` but fails AUC/AP and top-k autopsy, report target mismatch rather than useful complementarity.
