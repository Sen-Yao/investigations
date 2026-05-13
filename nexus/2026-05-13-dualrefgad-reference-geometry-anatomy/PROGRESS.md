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

- [x] Inspect current DualRefGAD code/data outputs to identify where reference responses and margin are computed.
- [x] Create `experiments/scripts/reference_geometry_anatomy.py` or equivalent export/analyzer.
- [x] Run a no-training diagnostic on elliptic seed0.
- [x] Save summary JSON/CSV/NPZ under `experiments/outputs/` (plots still pending).
- [ ] Update `insights.md` with decision table.

## Constraints

- No learned head in Phase 1.
- No formal sweep until a fixed no-head score is justified by anatomy evidence.
- Any future AUC/AP formal run must use `experiment-runner` and 5-seed mean±std.
- Anomaly labels are diagnostic-only; no training or selection leakage.


## Activity: 2026-05-13 — Seed0 no-training anatomy completed

**Status:** completed successfully after two debugging fixes.

### Command/process notes

- Initial failure 1: remote default Python was conda `base` without numpy. Fixed by using `/data/linziyao/miniconda3/envs/DualRefGAD/bin/python`.
- Initial failure 2: subset purity calculation incorrectly reused project `reference_purity()` on sliced refs. Fixed by adding `subset_reference_purity()`.
- Final run: manual no-training diagnostic, not a formal runner-managed experiment.
- Feedback issue: user correctly flagged that manual runs lack `experiment-runner` monitoring. Future long diagnostics must use runner registration if possible or an explicit Hermes watchdog/cron.

### Outputs

- `experiments/outputs/reference_geometry_anatomy_s0.summary.json`
- `experiments/outputs/reference_geometry_anatomy_s0.per_node.csv`
- `experiments/outputs/reference_geometry_anatomy_s0.arrays.npz`
- `experiments/outputs/reference_geometry_anatomy_seed0_analysis.md`

### Seed0 key results

- Margin test AUC/AP: **0.7938 / 0.5510**.
- Margin top1/top5 anomaly ratio: **0.995 / 0.768**.
- Normal refs purity: **1.000**.
- Anomaly refs global purity: **0.107**.
- Anomaly refs purity on anomaly target nodes: **0.684**.

### Immediate interpretation

Seed0 suggests the margin signal may be driven by target-conditional anomaly-reference enrichment: anomaly refs are globally noisy, but become substantially anomaly-enriched for true anomaly targets. Next step is to inspect false positives/missed anomalies and decide whether to run seeds 1-4 under a monitored/runner-compatible path.
