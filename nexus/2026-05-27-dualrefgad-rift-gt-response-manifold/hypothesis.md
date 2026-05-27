# Hypotheses — RIFT-GT

## H1 — Input signal hypothesis

The C-LEG3 response matrix contains nontrivial anomaly-ranking signal because `mat_mean` already matches or slightly exceeds `margin` under the strict reproduction audit. Therefore a learned GT reader should not be treated as learning from garbage input.

Verification:

- Reproduce/freeze C-LEG3 response matrices.
- Compare `margin`, `mat_mean`, and direction-aware scalar controls before any learned reader.

## H2 — No-position recovery hypothesis

A no-position GT/set reader can recover mat-mean-level distributional signal if the objective is aligned with normal-response-manifold learning.

Verification:

- RIFT-R0 with no row/column/rank embeddings.
- Train only with known-normal labels plus trimmed majority-normal unlabeled nodes.
- Use no-leakage diagnostics for early stopping.
- Evaluate AUC/AP only after training as diagnostic evidence.

## H3 — Position/role encoding hypothesis

Row/column/rank/role encodings can improve beyond mat-mean if reference grouping contains reusable semantics.

Verification:

- RIFT-R1 with row-only, column-only, row+column, and rank/role ablations.
- Permutation stress tests to distinguish true role use from brittle reference-order memorization.

## H4 — Vectorized-flow hypothesis

A scalar L2 distance from a single center may discard directional response information. RIFT should preserve direction in the readout embedding and optionally score with angular/radial decompositions once the minimal one-class objective is stable.

Verification:

- Compare radial energy, angular deviation, covariance/Mahalanobis energy, and multi-center variants as post-R0/R1 extensions.
- Keep the first runnable objective simple; complex losses require ablation justification.

## H5 — No-leakage controllability hypothesis

Without validation labels, training can still be controlled by center drift, trimmed-subset stability, dropout ranking stability, known-normal energy plateau, collapse checks, score-tail stability, and permutation stress.

Verification:

- Log all no-leakage metrics per epoch.
- Do not use test AUC/AP or true anomaly top-K for checkpoint/hyperparameter selection.
