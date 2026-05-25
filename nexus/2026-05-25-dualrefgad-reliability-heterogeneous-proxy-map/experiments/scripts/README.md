# Scripts

Probe scripts for `2026-05-25-dualrefgad-reliability-heterogeneous-proxy-map`.

Implementation preference:

- Reuse upstream C-LEG3 decomposition helpers from `../2026-05-21-dualrefgad-constraint-calibrated-reference-relation/experiments/scripts/` when possible.
- Do not fork scientific logic unless the change is explicitly documented in `PROGRESS.md`.
- Preserve strict split discipline: pass `data_split_seed=seed`, avoid threaded global RNG around data loading/reference construction, and save split fingerprints.
