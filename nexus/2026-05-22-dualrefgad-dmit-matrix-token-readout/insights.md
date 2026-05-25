# Insights — D-MiT Matrix Token Readout

## Current stance

D-MiT is approved as a naming frame, not yet as a final method claim. The first scientific claim is deliberately modest:

> There may be learnable readout space inside a fixed C-LEG3 response matrix, but the burden of proof is high because `mat_mean` is already strong.

## Non-goals

- Do not claim D-MiT is a new final method before V0/V1/V2 controls finish.
- Do not claim response matrix has natural image-like geometry unless V2 passes order/permutation controls.
- Do not introduce complex multi-term losses in the first probe.
- Do not use true anomaly labels for training, early stopping, or checkpoint selection.

## Interpretation ladder

1. **V0 fails and tracks `mat_mean`**: token reader adds no learnable space; close or redesign.
2. **V0 close to `mat_mean` but lower overlap**: distributional complementary signal exists; inspect top-k autopsy.
3. **V1 > V0**: reference identity matters; continue with identity-aware reader.
4. **V2 > V1 and survives order controls**: grid/position semantics may be discussed cautiously.
5. **V2 > V1 only in canonical order and fails permutation controls**: positional encoding is likely overfitting reference order.

## Reporting requirements

The final V0 report must include:

- exact V0 definition and no-position boundary;
- training data and pseudo anomaly generation protocol;
- comparison with `mat_mean`, `margin`, and robust hand-crafted readouts;
- pseudo-pair constraint satisfaction vs real anomaly AUC/AP;
- Spearman/top-k overlap with scalar baselines;
- seed-level table and aggregate mean ± std;
- failure-mode autopsy, especially whether V0 rescues anomalies or removes false positives differently from `mat_mean`.
