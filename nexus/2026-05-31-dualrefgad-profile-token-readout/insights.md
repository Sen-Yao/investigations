# Insights — DualRefGAD Profile-Token Readout

## Initial scientific position

Direction 2A is opened because the previous scalar-entry RIFT-GT route failed both philosophically and empirically. Philosophically, a response-matrix entry is a scalar rather than a rich token. Empirically, scalar-entry R0/R1/P0 readers underperformed the C-LEG3 `mat_mean` positive control.

The new investigation does not reject RIFT-GT. It narrows and corrects it: the reader should first operate on response-profile tokens, where each token is already a vector object with clear semantics.

## Interpretation rule

- If profile tokens beat or complement `mat_mean`, the response matrix contains learnable structure beyond scalar averaging.
- If profile tokens only copy `mat_mean`, they may still help explain the signal but should not be promoted as a method contribution.
- If profile tokens underperform scalar summaries, do not add larger heads immediately; first check whether the profile branch, normalization, objective, or reference regime is mismatched.

## Current no-go statements

- No true anomaly labels for training, early stopping, or hyperparameter selection.
- No raw node feature fusion in the first probe.
- No image/grid semantics claim without order-stability evidence.
- No broad k sweep unless the reference generator changes.

## 2026-06-01 closure insight

The profile-token ROCC route is demoted as a standalone route. Its negative result should be interpreted narrowly:

- It does **not** invalidate the response matrix, because `mat_mean` remained a strong positive control.
- It does **not** prove low-Spearman learned scores are useful complements, because node-level diagnostics showed that P0-only true positives are much fewer than mat-mean-only true positives and are unstable across seeds.
- It **does** show that merely correcting scalar-token philosophy at the row/column profile level is insufficient when the training objective remains ROCC-style compactness.

The successor should therefore change both the token object and the learning signal. The next route is MatGuardGT: a tensor/entry-vector GT reader trained with high-confidence `mat_mean` pairwise ranking distillation and strong known-normal low-score constraints, while treating multi-center ROCC, collapse barrier, and reference dropout as optional ablations rather than default assumptions.
