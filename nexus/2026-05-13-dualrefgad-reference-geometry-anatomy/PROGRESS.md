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

## Activity: 2026-05-13 — Phase 2 route 2 seed0 response distribution diagnostic

**Status:** completed foreground with immediate feedback.

### Protocol

- No training, no WandB, no formal sweep.
- Foreground run on HCCS-88 using `/data/linziyao/miniconda3/envs/DualRefGAD/bin/python` to avoid missing-env failures.
- Runtime: **230.5s**.
- Outputs pulled back to investigation archive.

### Outputs

- `experiments/scripts/reference_response_distribution.py`
- `experiments/outputs/reference_response_distribution_s0.summary.json`
- `experiments/outputs/reference_response_distribution_s0.per_node.csv`
- `experiments/outputs/reference_response_distribution_s0.arrays.npz`
- `experiments/outputs/reference_response_distribution_seed0_analysis.md`

### Key results

- Margin AUC/AP: **0.7938 / 0.5510**.
- `mat_mean` AUC/AP: **0.8200 / 0.5963**.
- `mat_mean` Spearman vs margin: **0.708**.
- `mat_mean` top5 Jaccard vs margin: **0.705**.
- `mat_mean` top5 anomaly ratio: **0.839** vs margin **0.768**.

### Interpretation

Route 2 passes the seed0 continuation gate: response-matrix summaries contain signal beyond scalar margin. Next step should be seeds 1-4 no-training response diagnostics under a monitored/runner-compatible path, then decide whether to open a method-validation investigation.


## Activity: 2026-05-13 — Phase 2 route 2 stability diagnostic (seeds 0-4)

**Status:** completed as exploratory no-training diagnostic.

### Protocol note / deviation

Seeds 1-4 were executed through manual SSH rather than `experiment-runner`. A Hermes `cron` no-agent watchdog was added after user review to monitor progress and was removed after completion. This run is retained as Phase 2 diagnostic evidence, but **must not be treated as formal method-validation evidence**.

### Outputs

- `experiments/outputs/reference_response_distribution_s1.summary.json`
- `experiments/outputs/reference_response_distribution_s2.summary.json`
- `experiments/outputs/reference_response_distribution_s3.summary.json`
- `experiments/outputs/reference_response_distribution_s4.summary.json`
- `experiments/outputs/reference_response_distribution_stability_s0_s4.md`
- `experiments/outputs/reference_response_distribution_stability_s0_s4.json`

### 5-seed focus results

| signal | AUC mean±std | AP mean±std | AUC > margin |
|---|---:|---:|---:|
| margin | 0.7952±0.0071 | 0.5163±0.0221 | - |
| mat_mean | 0.8009±0.0203 | 0.5335±0.0621 | 3/5 |
| mat_entropy | 0.7777±0.0296 | 0.5024±0.0689 | 1/5 |
| mat_high08_ratio | 0.7878±0.0232 | 0.4708±0.0668 | 2/5 |
| ra_anom_ratio_diagnostic | 0.9328±0.0005 | 0.7722±0.0050 | 5/5 |

### Interpretation

Seed0 was too optimistic. Across five seeds, `mat_mean` remains slightly better than margin on average and has lower top-k overlap with margin, but it only beats margin on **3/5** seeds and has noticeably higher variance. `mat_entropy` and `mat_high08_ratio` are not stable improvements.

Route2 remains useful as a mechanism/explanation probe: response matrix summaries expose information not rank-identical to scalar margin. However, the evidence is **not stable enough to directly promote a fixed response-matrix score as a method component**. If continued, the next step should be a clean runner-managed method-validation investigation with pre-declared fixed formulas and no label tuning.
