# RIFT-GT — Response-Informed Flow Transformer for C-LEG3

> Investigation for the feasibility of using the C-LEG3 response matrix as a GT input and learning a no-leakage normal-response-manifold readout.

## Internal codename

**RIFT-GT** = **Response-Informed Flow Transformer**.

Rationale:

- **Response-Informed**: the method starts from the C-LEG3 response matrix, not raw node features alone.
- **Flow**: preserves the earlier insight that directional relations between embeddings/reference responses matter, not only scalar magnitude.
- **Transformer / GT**: the learnable reader is a Graph/Set/Reference-Pair Transformer over response tokens.

## Starting point

Fixed upstream regime:

- C-LEG3 / `old_exact_080_regime` response matrix;
- `normal_k=4`, `anom_k=16` as the first canonical setting;
- labels: known-normal labels are allowed for normal-manifold training; true anomaly labels are diagnostic-only and must not be used for training, checkpoint selection, or hyperparameter tuning.

## Research question

Can a GT-style response-matrix reader learn a stable, no-leakage anomaly score from C-LEG3 response matrices that at least matches `mat_mean` and ideally captures complementary directional/reference-flow information beyond scalar averaging?

## Scope

RIFT-GT should start minimal and expand only through gates:

1. **RIFT-R0 / Set reader**: no row/column position encoding; response entries treated as a set/distribution. Goal: recover mat-mean-level signal under a data-driven normal-manifold objective.
2. **RIFT-R1 / Reference-pair reader**: add row/column/rank/role embeddings. Goal: test whether normal-anchor and anomaly-reference grouping matters.
3. **RIFT-R2 / Flow-aware reader**: add vectorized response/displacement readout heads that keep direction information instead of collapsing immediately to L2 energy.
4. **RIFT-R3 / Extended objective**: only after R0-R2 gates pass, consider more complex losses such as angular compactness, covariance/Mahalanobis energy, multi-center normal manifolds, or view consistency.

## Non-goals

- Do not train directly to match `margin` or `mat_mean` as a teacher.
- Do not introduce pseudo anomalies as the first training signal.
- Do not use test AUC/AP or true anomaly top-K for early stopping or hyperparameter tuning.
- Do not claim grid/image semantics unless row/column order semantics pass permutation stress tests.

## No-leakage metrics

This investigation follows `research/references/dualrefgad-riftgt-no-leakage-metrics.md`. If reports mention early stopping/monitoring indicators, they must define:

- center drift;
- trimmed-unlabeled subset Jaccard stability;
- reference/token dropout ranking stability;
- known-normal energy plateau;
- representation collapse / covariance-rank check;
- score tail mass stability;
- permutation stress test.

## Current status

Created after the RIFT-GT method discussion and before code implementation. Previous unfinished operational items were checked: the ABCD watchdog/publisher cron IDs recorded in the prior investigation are already absent from the scheduler, no tracked Hermes background process is running, and the `~/investigations` git tree was cleaned/committed before starting this investigation.
