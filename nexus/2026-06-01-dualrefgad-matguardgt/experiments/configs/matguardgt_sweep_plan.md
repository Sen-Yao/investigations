# MatGuardGT Sweep Plan — 500 to 1000 Run Budget

> Planning artifact only. Do not launch until a runner-validatable YAML and implementation script pass smoke tests.

## Fixed protocol

- Dataset/regime: C-LEG3 / old-exact response-matrix construction, same split/seeds as predecessor probes.
- Seeds: `[0, 1, 2, 3, 4]` for every promoted config.
- Labels: true anomaly labels diagnostic-only for AUC/AP/top-k autopsy; not used for training, early stopping, or hyperparameter selection.
- Baselines to log in every run: `mat_mean`, margin, old P0 reproduction when feasible, rowcol-profile ROCC baseline.
- Primary monitors: teacher-pair violation rate, Spearman with `mat_mean`, Top-1%/5% Jaccard with `mat_mean`, known-normal score mean/q90, effective rank, embedding variance, reference-dropout stability, `mat_mean` FN repaired vs TP destroyed.

## Token/readout candidates

Reader families:

1. `rowcol_profile_gt`: row and column profile tokens, retained as vector-token baseline.
2. `entry_vector_gt`: each `(normal_ref, anomaly_ref)` response entry becomes a vector token.
3. `entry_vector_gt_relpos`: entry-vector token plus row/column rank/position/type embeddings.
4. `entry_vector_gt_stats`: entry-vector token plus row/column/global rank-stat descriptors.

Readout heads:

1. `score_head_pool`: pooled token embedding -> scalar score.
2. `energy_head`: embedding energy / low-score normal constraint head.
3. `attn_pool_score`: attention pooling -> scalar score.

Initial recommendation: keep GT depth at 1 layer, 2 heads first; test d_model `[32, 64]` before larger settings.

## Loss families

Base loss for all main variants:

- `L_N_score`: known-normal low-score constraint.
- `L_mat_rank`: high-confidence pairwise ranking distillation from `mat_mean`.

Optional ablations:

- `L_barrier`: collapse barrier only when embedding variance/effective-rank drops below threshold.
- `L_ref_cons`: reference/token dropout consistency.
- `L_MC`: multi-center normal manifold, test-only; do not make it default.

Excluded in this stage:

- interval guardrail `|s_GT - s_mat| < epsilon`;
- residual final score `s_mat + alpha*r_theta`;
- pseudo-anomaly / hard-negative contrast.

## Suggested 720-run formal grid

Runs = 5 seeds × 144 configs = 720.

Factorization:

- reader_family: 4
- d_model: 2 (`32`, `64`)
- pooling/readout: 2 (`mean_pool_score`, `attn_pool_score`)
- mat_rank_margin_delta: 3 (`q70`, `q80`, `q90` high-confidence pair thresholds)
- lambda_rank: 3 (`0.1`, `0.3`, `1.0`)
- aux_profile: 2 (`none`, `barrier_refdrop`)

Total: 4 × 2 × 2 × 3 × 3 × 2 × 5 = 720 runs.

Rationale:

- Keeps the first stage centered on teacher ranking and token design.
- Tests barrier/reference dropout together as a stabilizer pack first to keep the grid small.
- Deliberately excludes multi-center from the main 720-run grid.

## Optional +180 run multi-center sidecar

Runs = 5 seeds × 36 configs = 180.

Only after the 720-run grid shows that base MatGuardGT is runnable and not collapsed:

- reader_family: top 2 from main grid
- d_model: top 1 or `64`
- pooling/readout: top 1 or `attn_pool_score`
- lambda_rank: top 2
- mc_prototypes: 3 (`2`, `4`, `8`)
- lambda_mc: 3 (`0.05`, `0.1`, `0.25`)

Example reduced count: 2 × 1 × 1 × 2 × 3 × 3 × 5 = 180.

Grand total: 900 runs.

## Smaller 540-run fallback

If budget or runtime looks tight:

- reader_family: 3 (`rowcol_profile_gt`, `entry_vector_gt`, `entry_vector_gt_stats`)
- d_model: 2
- readout: 2
- mat_rank_margin_delta: 3
- lambda_rank: 3
- aux_profile: 1 (`none` only)

Total: 3 × 2 × 2 × 3 × 3 × 1 × 5 = 540 runs.

Then run a 90–180 run auxiliary stabilizer sidecar for barrier/refdrop/multicenter only on the best 2 reader families.

## Promotion gates

Strong pass:

- 5-seed AUC or AP mean exceeds `mat_mean`, and at least 4/5 seeds do not degrade on the chosen metric.
- Top-1% precision of `mat_mean` is not destroyed.
- Teacher-pair violation rate is lower than normal-only baseline.
- `mat_mean` FN repaired count exceeds `mat_mean` TP destroyed count.

Weak pass:

- Does not exceed `mat_mean`, but shows stable FN repair with limited TP destruction and nontrivial token/readout evidence.

Fail / demote:

- Pure imitation: Spearman with `mat_mean` ≈ 1 without AUC/AP or top-k autopsy gain.
- Target mismatch: Spearman near 0 plus weak AUC/AP.
- Stabilizer trap: barrier/refdrop improves no-leakage monitors but worsens anomaly ranking.
