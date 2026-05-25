# Hypotheses — D-MiT Matrix Token Readout

## H1 — Set-level response distribution may contain learnable signal

If the 64 response entries are treated as an unordered set, a low-capacity Transformer/readout may learn distributional evidence beyond a fixed `mat_mean`: upper-tail support, consensus, dispersion, robust pooling, or interactions between strong and weak entries.

Validation: V0 / Set-D-MiT should be compared against `mat_mean`, `margin`, robust hand-crafted readouts, and pseudo-constraint-only baselines across 5 seeds.

## H2 — Reference identity is not assumed in V0

V0 intentionally has no row/column/reference identity. If V0 works, the signal should be interpreted as full-matrix distributional signal, not matrix geometry.

Validation: V1 can later add reference identity. If V1 improves over V0, identity matters; if not, signal is probably distributional.

## H3 — Image-like 2D position is a high-risk assumption

V2 / Grid-D-MiT can only be promoted if row/column order has stable scientific meaning or order controls show the gain is not a fragile ordering artifact.

Validation: compare canonical order, per-node/reference permutation, fixed random order, and possibly sorted-by-score order before claiming grid structure.

## H4 — Two textual constraints are necessary but not sufficient

Known normals low and pseudo anomalies higher than their source normal can train a scorer, but they may only shape pseudo-data rather than real anomaly ranking.

Validation: report both pseudo-pair constraint satisfaction and real test AUC/AP; do not use true anomaly labels for training or early stopping.

## H5 — Strong scalar baselines define the burden of proof

`mat_mean` is already a strong positive control under C-LEG3. V0 does not need to immediately and stably beat it to remain scientifically useful, but it must reveal either complementary ranking or interpretable failure modes.
