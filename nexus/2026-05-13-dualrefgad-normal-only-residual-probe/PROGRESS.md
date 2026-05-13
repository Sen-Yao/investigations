# PROGRESS — DualRefGAD Normal-only Residual Probe

## Day 1: 2026-05-13 — Investigation created

**Activity:** Created investigation archive and framed the route as a diagnostic probe, not a final method.

- Location: `~/investigations/nexus/2026-05-13-dualrefgad-normal-only-residual-probe/`
- Related prior investigation: `2026-05-09-semisupervised-negative-signal-for-dualrefgad`
- Current status: waiting for user confirmation before HCCS-88 5-seed diagnostic run.

**Interpretation Rule:**

> If the residual probe does not produce stable improvement over margin-only, drop the route. If it does, inspect what the residual learns and redesign it as a unified scoring principle rather than keeping an additive patch.

## Pending

- [ ] Run 5-seed diagnostic on HCCS-88 after confirmation.
- [ ] Record per-seed AUC/AP deltas, correction statistics, Spearman, top-k overlap, and selected epochs.
- [ ] Decide: close route, inspect unstable signal, or redesign mechanism.
