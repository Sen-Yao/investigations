# Insights — RIFT-GT

## Current methodological decision

The method should be developed as **RIFT-GT**:

> C-LEG3 response matrices are treated as response-informed reference-flow objects. A GT reader maps each matrix into a normal-response representation. The first training objective is a no-leakage robust one-class compactness objective over known normals and trimmed unlabeled likely-normal nodes. Later extensions may preserve or explicitly score angular/directional deviation once the base objective is stable.

## Important caution

The first scalar loss `||z-c||_2^2` is a stabilizing training handle, not the final claim that all anomaly information is radial magnitude. RIFT-GT must keep vectorized response information in the GT embedding before the scalar energy is computed. If R0/R1 only reproduces mat-mean, it is not enough; later gates should test whether direction-aware residuals or angular deviations provide complementary signal.

## Code-development reminders

- Define `z_v = GT_theta(T_v; E_pos)` explicitly, where `T_v` is the tokenized response matrix and `E_pos` is optional row/column/rank/role encoding.
- Log no-leakage metrics from the first implementation.
- Do not use diagnostic AUC/AP for early stopping.
- Keep R0 minimal; do not add multi-loss complexity until failure mode is observed.

## 2026-05-31 closure insight

The scalar-entry RIFT-GT route is closed as the main next step. R0/R1/P0 experiments did not match the C-LEG3 `mat_mean` positive control. This does not reject response-matrix learning; it rejects the current entry-as-token ontology plus ROCC-style scalar-entry reader as the priority route.

The important interpretation is methodological: a single response value `M_ij(v)` is too poor to serve as a Transformer token. A better first correction is to make the token itself a response profile vector, e.g. a row profile over all anomaly references or a column profile over all normal references. The next investigation should test profile-token readout before any larger entry-token architecture is revived.

