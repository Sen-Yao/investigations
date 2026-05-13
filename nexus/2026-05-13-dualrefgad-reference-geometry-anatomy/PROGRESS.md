# PROGRESS — DualRefGAD Reference Geometry Anatomy

## Day 1: 2026-05-13 — Investigation created

**Activity:** Opened a new mechanism-autopsy investigation after the additive residual route was closed.

- Location: `~/investigations/nexus/2026-05-13-dualrefgad-reference-geometry-anatomy/`
- Prior negative result: `2026-05-13-dualrefgad-normal-only-residual-probe`
- Current phase: **Phase 1 — Anatomy without training**
- Formal sweep status: **not started / intentionally out of scope for Phase 1**

## Initial decision

Do not train another correction head. First dissect the existing margin/reference geometry:

1. normal-reference distance;
2. anomaly-reference distance;
3. margin decomposition;
4. reference purity;
5. hop/descriptor contribution;
6. top-k failure cases;
7. reference response vector distribution.

## Pending

- [ ] Inspect current DualRefGAD code/data outputs to identify where reference responses and margin are computed.
- [ ] Create `experiments/scripts/reference_geometry_anatomy.py` or equivalent export/analyzer.
- [ ] Run a no-training diagnostic on elliptic.
- [ ] Save summary JSON/CSV and plots under `experiments/outputs/` and `experiments/plots/`.
- [ ] Update `insights.md` with decision table.

## Constraints

- No learned head in Phase 1.
- No formal sweep until a fixed no-head score is justified by anatomy evidence.
- Any future AUC/AP formal run must use `experiment-runner` and 5-seed mean±std.
- Anomaly labels are diagnostic-only; no training or selection leakage.
