# Hypotheses

## H1 — full matrix contains normal-only pattern signal

Normal nodes share a response-matrix geometry that cannot be reduced to margin/mean. A small AE trained only on labeled normals should reconstruct normal matrices better than anomaly matrices.

Evidence motivating H1:

- Prior Route2 anatomy found matrix-derived statistics (`mat_mean`, `mat_high08_ratio`) can sometimes approach margin-level signal.
- Additive scalar correction failed, suggesting new signal—if any—may live in a distributional pattern rather than in a scalar residual head.

## H0 — full matrix does not add useful signal

The apparent Route2 signal is mostly captured by margin orientation, degree/rejection regimes, or unstable reference selection. Matrix AE reconstruction error will underperform scalar baselines or correlate strongly with degree/rejection.


## 2026-05-21 — Derived hypothesis after the regime comparison evidence

The earlier H1/H0 split is still useful, but the new evidence suggests a more precise derived hypothesis:

## H1' — response-matrix signal depends on sequence-generation regime

The historical high `mat_mean≈0.80` effect seems to require the old strong sequence-generation/reference regime. When the regime changes, the same matrix family can flip orientation and `neg_mat_mean` becomes the useful scalar family instead.

Evidence:

- old exact / old refs alignment recover `mat_mean≈0.80`;
- current refs and Stage A shift toward `neg_mat_mean`;
- Matrix AE instability localizes more to reference/orientation/regime than to initialization or checkpoint selection;
- AMRF is not yet a stable replacement for scalar summaries.

## H2' — fixed C should be the next control experiment, not another generator change

Before changing the sequence generator again, the next science step should fix the historically strong C regime and study response-matrix readout families there. This will separate “signal in the matrix” from “signal caused by changing the generator.”

## Interpretation branches

1. **If fixed C keeps `mat_mean≈0.80` and scalar readouts stay strong**: the old regime is a stable asset and should be preserved.
2. **If fixed C still makes orientation flip or weaken**: the current signal is regime-sensitive enough that readout may need to be tied explicitly to regime/reference quality.
3. **If no readout beats the best scalar family under fixed C**: the matrix family may already be adequately summarized by simple statistics, and learnable heads are unnecessary.
4. **If A/B-only results are unstable while C is stable**: A/B should remain diagnostic-only failure cases, not a method direction.
