# Hypotheses — DualRefGAD Profile-Token Readout

## H1 — Token ontology correction hypothesis

A row or column response profile is a valid Transformer token because it is a vector with interpretable coordinates. A single response entry is only a scalar and should be treated as an attribute or coordinate, not as the token object itself.

Verification:

- Implement row-profile tokenization without raw node features or reference embeddings.
- Confirm token shape and projection are explicit in the experiment config/report.
- Compare against the closed scalar-entry RIFT-GT results as a negative predecessor.

## H2 — Profile-shape signal hypothesis

Anomalies are not only separated by global response magnitude. They may also differ in the shape of their response profile across anomaly-side references or normal-side references.

Verification:

- Compare row-profile reader against `mat_mean`, margin, row mean, row standard deviation, row quantile summaries, and flatten-MLP.
- Run permutation stress: row shuffle, column shuffle, and row+column shuffle where applicable.

## H3 — Complementarity hypothesis

A profile-token reader is useful if it either beats the strongest scalar baseline or remains close while changing the ranking meaningfully.

Verification:

- Report 5-seed AUC/AP.
- Report Spearman correlation and top-K overlap with `mat_mean`, margin, and the strongest scalar summary.
- Continue only if improvement is meaningful or complementarity is clear.

## H4 — No-shortcut hypothesis

Any improvement must not mainly come from degree/hubness proxies, reference-order memorization, or implicit label leakage.

Verification:

- Keep true anomaly labels evaluation-only.
- Include degree/hubness/rejection proxies when available.
- Use no-leakage monitoring for checkpoint selection.
- Treat order-sensitive gains as suspect unless permutation controls justify the semantics.

## H5 — Minimal-reader-first hypothesis

Before adding node features, reference embeddings, complex losses, or larger heads, a tiny profile-token reader should establish whether learnable space exists beyond scalar summaries.

Verification:

- Start with one-layer, two-head small reader.
- Compare with flatten-MLP to separate sequence/token benefit from generic capacity.
- If tiny reader fails and flatten-MLP also fails, return to reference construction or scalar mechanism decomposition rather than scaling architecture.
