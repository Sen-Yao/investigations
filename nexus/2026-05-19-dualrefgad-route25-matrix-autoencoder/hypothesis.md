# Hypotheses

## H1 — full matrix contains normal-only pattern signal

Normal nodes share a response-matrix geometry that cannot be reduced to margin/mean. A small AE trained only on labeled normals should reconstruct normal matrices better than anomaly matrices.

Evidence motivating H1:

- Prior Route2 anatomy found matrix-derived statistics (`mat_mean`, `mat_high08_ratio`) can sometimes approach margin-level signal.
- Additive scalar correction failed, suggesting new signal—if any—may live in a distributional pattern rather than in a scalar residual head.

## H0 — full matrix does not add useful signal

The apparent Route2 signal is mostly captured by margin orientation, degree/rejection regimes, or unstable reference selection. Matrix AE reconstruction error will underperform scalar baselines or correlate strongly with degree/rejection.

## Interpretation branches

1. **AE beats scalar baseline**: promote to denoising AE / regime-conditioned decoder.
2. **AE close but complementary**: investigate ensemble or two-stage route.
3. **AE weak**: drop full-matrix modeling and prioritize R_a repair or orientation/regime diagnostics.
4. **AE degree-correlated**: treat as structural regime probe, not anomaly signal.
