# DualRefGAD Profile-Token Readout

> Investigation for Direction 2A: correcting the response-matrix token ontology by treating row/column response profiles as vector tokens rather than treating each scalar matrix entry as an independent Transformer token.

## Internal codename

**RIFT-Profile** = Response-Informed Flow Transformer with profile-level tokens.

Alternative local shorthand: **RP-MiT** = Response Profile Matrix Token readout.

## Starting point

This investigation starts from three predecessor facts:

1. The C-LEG3 / `old_exact_080_regime` response matrix is a strong positive-control signal object. In the fixed setting, `mat_mean` reaches around AUC 0.81 and AP 0.56 across five seeds.
2. The scalar-entry RIFT-GT route underperformed this positive control. R0/R1/P0 scalar-entry readers stayed around AUC 0.70–0.73 and did not justify more capacity/loss complexity.
3. A single response entry is a scalar, not a semantically rich token. The next correction should change the token object before adding architectural complexity.

## Research question

If a node-level response matrix is represented as row or column response-profile tokens, can a small Transformer-style reader recover or exceed the C-LEG3 `mat_mean` signal while providing complementary, auditable ranking information?

## Scope

First-stage scope is deliberately narrow:

- Freeze upstream encoder and reference construction.
- Use C-LEG3 / `old_exact_080_regime` as the fixed reference regime.
- Start with row-profile tokens: each normal-reference row becomes one vector token over anomaly references.
- Add column-profile tokens only after the row branch has passed basic implementation and baseline checks.
- Compare against scalar baselines before claiming any learned-reader contribution.

## Non-goals

- Do not restart broad reference-length sweeps.
- Do not train on true anomaly labels.
- Do not use diagnostic AUC/AP for checkpoint selection or hyperparameter tuning.
- Do not claim image-like grid semantics for the response matrix.
- Do not introduce raw node embedding `h_v`, reference embeddings, or large fusion heads in the first gate.
- Do not revive scalar-entry RIFT-GT by adding more capacity unless profile-token probes fail for a clearly diagnosed reason.

## First continuation object

Prepare a runner-registered probe for row-profile token readout:

- input object: fixed response matrix per node;
- token object: one row profile per normal reference;
- projection: anomaly-reference dimension to small model dimension;
- reader: one-layer, two-head small Transformer/GT;
- objective: normal-only / no-leakage monitor; exact loss to be finalized before launch;
- diagnostics: AUC/AP, AP/top-K, Spearman/top-K overlap with `mat_mean` and margin, degree/hubness confound, row/column permutation stress.

## Current status

Skeleton created on 2026-05-31 after closing the scalar-entry RIFT-GT route as the main next step. No experiment has been launched yet.
