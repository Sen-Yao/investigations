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
