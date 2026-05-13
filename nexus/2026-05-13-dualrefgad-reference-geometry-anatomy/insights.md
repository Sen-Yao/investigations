# Insights — DualRefGAD Reference Geometry Anatomy

## Initial position

The previous additive residual probe closed the `margin + correction` route. The correction head learned margin compression/calibration, not baseline-independent anomaly ranking signal.

This investigation therefore starts from a stricter premise: **do not add a learned head until margin/reference geometry has been fully dissected.**

## Working intuition

The useful signal likely lives in reference geometry itself. The unknown is whether scalar margin already captures almost all of it, or whether the response vector over multiple references contains distributional inconsistency that scalar margin collapses away.

## Expected decision output

The investigation should end with a decision table, not just plots:

| Route | Continue? | Evidence |
|---|---|---|
| reference construction | TBD | |
| normal-manifold deviation | TBD | |
| multi-reference distributional inconsistency | TBD | |
| learned residual head | No | Closed by prior investigation |

## Current status

Created. Phase 1 anatomy script and data export are pending.

## 2026-05-13 — Seed0 no-training reference geometry anatomy

Artifacts:
- `experiments/scripts/reference_geometry_anatomy.py`
- `experiments/outputs/reference_geometry_anatomy_s0.summary.json`
- `experiments/outputs/reference_geometry_anatomy_s0.per_node.csv`
- `experiments/outputs/reference_geometry_anatomy_s0.arrays.npz`
- `experiments/outputs/reference_geometry_anatomy_seed0_analysis.md`

### Key numbers

- Margin test AUC/AP: **0.7938 / 0.5510**.
- Margin top1/top5 anomaly ratio: **0.995 / 0.768**.
- Normal refs purity: **1.000**.
- Anomaly refs global purity: **0.107**.
- Anomaly refs purity on anomaly target nodes: **0.684**.

### Interpretation

Seed0 supports a target-conditional reference-purity story: `R_a` is not globally clean, but when the target is a true anomaly, selected anomaly-side references become much more anomaly-enriched. That makes the scalar margin strong without implying that a learned residual head has independent signal.

### Process note

This was a manual no-training diagnostic probe, not a runner-compliant formal experiment. It is acceptable only as Phase-1 debugging/anatomy evidence. Future multi-seed diagnostics should either be registered through `experiment-runner` (`probe`/`single-run` path if available) or paired with an explicit Hermes watchdog so completion/failure is reported promptly.

## 2026-05-13 — Route 2 seed0: multi-reference response distribution

Artifacts:
- `experiments/scripts/reference_response_distribution.py`
- `experiments/outputs/reference_response_distribution_s0.summary.json`
- `experiments/outputs/reference_response_distribution_s0.per_node.csv`
- `experiments/outputs/reference_response_distribution_s0.arrays.npz`
- `experiments/outputs/reference_response_distribution_seed0_analysis.md`

### Key result

Route 2 is positive on seed0. `mat_mean`, the mean of the full normal-anchor × anomaly-ref response matrix, improves over scalar margin:

- Margin AUC/AP: **0.7938 / 0.5510**.
- `mat_mean` AUC/AP: **0.8200 / 0.5963**.
- `mat_mean` Spearman vs margin: **0.708**.
- `mat_mean` top5 Jaccard vs margin: **0.705**.
- `mat_mean` top5 anomaly ratio: **0.839** vs margin **0.768**.

### Interpretation

Scalar mean-pooled margin is not sufficient: the full response matrix contains ranking signal that is both stronger on seed0 and not rank-identical to margin. This supports continuing `multi-reference distributional inconsistency` as Phase 2.

Diagnostic-only `ra_anom_ratio_diagnostic` is a strong upper-bound/explanatory variable (AUC **0.9322**) but is not deployable because it uses labels. It confirms that target-conditioned `R_a` purity explains much of the mechanism.

### Decision

Run seeds 1-4 for this no-training diagnostic before designing a fixed formula. If `mat_mean` remains stable, open a method-validation investigation for no-head response-matrix scoring.


## 2026-05-13 — Route 2 stability: seeds 0-4

Artifacts:
- `experiments/outputs/reference_response_distribution_stability_s0_s4.md`
- `experiments/outputs/reference_response_distribution_stability_s0_s4.json`
- `experiments/outputs/reference_response_distribution_s{1,2,3,4}.*`

### Key numbers

| signal | AUC mean±std | AP mean±std | AUC > margin |
|---|---:|---:|---:|
| margin | **0.7952±0.0071** | **0.5163±0.0221** | - |
| `mat_mean` | **0.8009±0.0203** | **0.5335±0.0621** | **3/5** |
| `mat_entropy` | 0.7777±0.0296 | 0.5024±0.0689 | 1/5 |
| `mat_high08_ratio` | 0.7878±0.0232 | 0.4708±0.0668 | 2/5 |
| `ra_anom_ratio_diagnostic` | 0.9328±0.0005 | 0.7722±0.0050 | 5/5 |

### Updated interpretation

The seed0 positive result does not fully survive stability validation. `mat_mean` is still the best deployable route2 candidate and is slightly above margin on mean AUC/AP, but the improvement is unstable: it wins only 3/5 seeds and loses badly on seed2.

This changes the conclusion from "Route2 likely method candidate" to:

> Multi-reference response distribution is a useful explanatory lens and may contain complementary ranking information, but current simple matrix summaries are not stable enough to become a standalone method component.

The label-dependent `ra_anom_ratio_diagnostic` remains very strong and stable, confirming the target-conditioned anomaly-reference purity mechanism, but it is diagnostic-only and cannot be used for scoring.

### Decision table update

| Route | Continue? | Evidence |
|---|---|---|
| reference construction / target-conditioned `R_a` purity | Yes, as mechanism | Diagnostic upper bound `ra_anom_ratio_diagnostic` AUC 0.9328±0.0005; Phase 1 anomaly-target `R_a` purity enrichment |
| multi-reference distributional inconsistency | Cautious / explanation-first | `mat_mean` AUC 0.8009±0.0203 vs margin 0.7952±0.0071, but only 3/5 seed wins |
| fixed no-head response-matrix score | Not yet | Needs clean runner-managed method-validation with pre-declared formulas; exploratory evidence alone insufficient |
| learned residual/correction head | No | Prior residual investigation showed compression/calibration, not independent signal |

### Process note

Seeds 1-4 were run manually over SSH with a Hermes cron watchdog after user review, not via `experiment-runner`. This is acceptable as exploratory diagnostic evidence only. Future multi-seed validation runs should be runner-managed or explicitly registered as non-formal watchdog diagnostics before launch.
