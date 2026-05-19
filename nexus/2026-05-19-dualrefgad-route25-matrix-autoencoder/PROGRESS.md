# Progress

## 2026-05-19 — Investigation created

- Created independent investigation for Route2.5 Matrix Autoencoder.
- User approved execution of seed0 probe.
- Planned runner-registered probe using `experiment-runner profile=probe`.

### Probe summary

- Input: flattened response matrix `M ∈ R^(4×16)`.
- Train: labeled-normal nodes only.
- Evaluate: AUC/AP on test split, labels diagnostic-only.
- Baselines: margin / -margin / matrix scalar summaries / degree / rejection.


## 2026-05-19 — Seed0 probe completed

Runner job: `exp_20260519_165831_route25_matrix_ae_elliptic_seed0`  
Remote: HCCS-25 GPU0  
Status: finished, exit code 0  
Runtime: 42.3s

### Key result

- Decision: `INCONCLUSIVE`
- Strongest scalar baseline: `neg_mat_mean` AUC=0.6377, AP=0.1405
- Best AE: `ae_mse_latent8` AUC=0.6028, AP=0.1140
- AE Spearman with margin: -0.4560
- AE top5 Jaccard with margin: 0.0036

### Immediate interpretation

Seed0 does **not** promote Matrix AE yet: best AE underperforms the best scalar baseline by 0.0349 AUC. It is above the DROP threshold but not close enough to beat/replace scalar summaries. Low top-k overlap suggests it may still be a complementary diagnostic, but this single seed is insufficient for a method claim.

## 2026-05-19 — Seed1-4 probes completed and 5-seed aggregation

Runner jobs:
- `exp_20260519_170908_route25_matrix_ae_elliptic_seed1` — finished, HCCS-25 GPU1
- `exp_20260519_170909_route25_matrix_ae_elliptic_seed2` — finished, HCCS-25 GPU2
- `exp_20260519_170909_route25_matrix_ae_elliptic_seed3` — finished, HCCS-25 GPU3
- `exp_20260519_170909_route25_matrix_ae_elliptic_seed4` — finished, HCCS-25 GPU4

All outputs were pulled back into `experiments/outputs/`; logs were pulled into `experiments/logs/`. Aggregate JSON: `experiments/outputs/route25_matrix_ae_elliptic_5seed_aggregate.json`.

### 5-seed aggregate

- Best AE AUC: 0.6115 ± 0.0567
- Best AE AP: 0.1511 ± 0.0696
- Best scalar AUC: 0.6500 ± 0.0111
- Best scalar AP: 0.1897 ± 0.0461
- AE - scalar ΔAUC: -0.0385 ± 0.0647
- AE - scalar ΔAP: -0.0387 ± 0.0951
- Best AE Spearman with margin: -0.4218 ± 0.1051
- Best AE top5 Jaccard with margin: 0.0075 ± 0.0061
- Per-seed decisions: PROMOTE=2, DROP=2, INCONCLUSIVE=1

### Updated interpretation

The multi-seed result does **not** support promoting Matrix AE as a stable method component. Although seed1 and seed3 individually cross the probe's PROMOTE rule, seed2 and seed4 drop below threshold and the 5-seed mean underperforms the strongest scalar matrix summaries by ΔAUC=-0.0385. The signal is seed-unstable and weaker than cheap scalar response-matrix statistics.

Conclusion boundary: this is valid runner-registered exploratory evidence for Route2.5 diagnosis, not a full SOTA experiment. It argues against spending a full method sweep on this exact normal-only Matrix AE head without repairing the representation/regime problem first.

## 2026-05-19 — Phase 1 AE instability audit completed

Runner job: `exp_20260519_175609_route25_matrix_ae_phase1_instability_aud`  
Remote: HCCS-25 GPU0  
Status: finished, exit code 0  
Runtime: 114.6s

### Audit protocol

- Reused the Route2.5 frozen-encoder response-matrix construction.
- For each split seed `{0,1,2,3,4}`, reran AE training with AE init seeds `{0,1,2,3,4}`.
- Audited latent dimensions `{8,16}`; labels remained diagnostic-only.
- Output: `experiments/outputs/route25_matrix_ae_phase1_instability_audit.json`.

### Phase 1 aggregate

- Best repeat AE AUC: 0.6143 ± 0.0499
- Best repeat AE ΔAUC vs scalar: -0.0357 ± 0.0571
- Scalar AUC: 0.6500 ± 0.0100
- Mean within-split AE AUC std: 0.0033 ± 0.0019
- Promote-repeat splits: 2/5
- Drop-repeat splits: 2/5
- Decision: `SPLIT_REFERENCE_INSTABILITY_DOMINATES__DO_NOT_PROMOTE`

### Interpretation

Repeating AE initializations does **not** repair Route2.5 Matrix AE. Within a fixed split/reference construction, AE AUC is very stable (mean within-split std ≈ 0.0033), but across split seeds the best-repeat AE remains unstable and underperforms scalar summaries on average. This localizes the failure more to split/reference/representation regime instability than to random AE initialization noise.

Conclusion boundary: Phase 1 strengthens the negative method decision for this exact Matrix AE head. It does not reject response-matrix features globally; it says the next repair should target reference distribution / orientation regime before adding learnable capacity.

## 2026-05-19 — Phase 2 learning-strategy posthoc diagnostic completed

Runner job: `exp_20260519_184959_route25_matrix_ae_phase2_learning_strate`  
Execution: local zero-new-training posthoc probe over existing Phase-1 repeated-AE output  
Status: finished, exit code 0  
Output: `experiments/outputs/route25_matrix_ae_phase2_learning_strategy_posthoc.json`.

### Audit protocol

- No new model training and no GPU use; this is a low-cost posthoc diagnostic over Phase-1 results.
- Tested whether cheap learning-strategy choices could repair Matrix AE instability:
  - deployable validation-loss selector;
  - fixed latent dimension `{8,16}`;
  - label-oracle best-AUC selector as an upper-bound autopsy only.
- Labels remain diagnostic-only; the label-oracle selector is explicitly **not deployable**.

### Phase 2 aggregate

- Scalar baseline AUC: 0.6500 ± 0.0100
- Oracle best-AUC selector AE AUC: 0.6143 ± 0.0499; ΔAUC vs scalar: -0.0357 ± 0.0571
- Validation-loss selector AE AUC: 0.6099 ± 0.0488; ΔAUC vs scalar: -0.0401 ± 0.0556
- Validation-loss selector promote splits: 1/5; drop splits: 2/5
- Fixed latent=8 ΔAUC: -0.0406 ± 0.0586; promote 1/5; drop 2/5
- Fixed latent=16 ΔAUC: -0.0357 ± 0.0571; promote 2/5; drop 2/5
- Spearman-like rank correlation of validation loss with AUC across all runs: -0.0618
- Decision: `NO_CHEAP_LEARNING_STRATEGY_REPAIR__KEEP_DROP_DECISION`

### Interpretation

Cheap learning-strategy fixes do **not** repair Route2.5 Matrix AE. Even the label-oracle best-AUC selector remains below scalar response-matrix summaries on average, and the deployable validation-loss selector is worse. Fixed latent capacity also fails. The negative decision is therefore not just “we selected the wrong AE checkpoint/latent”; the failure remains localized to reference/orientation/representation regime rather than AE training policy.

