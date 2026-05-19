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

