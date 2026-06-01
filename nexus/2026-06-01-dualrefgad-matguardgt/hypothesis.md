# Hypotheses — MatGuardGT

## H1 — High-confidence `mat_mean` ranking distillation fixes objective mismatch

Given a strong response-matrix scalar positive control `s_mat(v)`, training a GT reader with high-confidence pairwise ranking constraints should align learned scores with reliable anomaly-ordering relations without forcing pointwise imitation.

Falsification:

- teacher-pair violation rate remains high;
- Spearman with `mat_mean` remains near zero and AUC/AP remain below old ROCC/profile baselines;
- or Spearman approaches 1 while AUC/AP do not improve, indicating thin imitation.

## H2 — Entry-vector tokens preserve local reference-pair structure better than row/column profile tokens

Each response-matrix entry token should be vector-valued, containing the raw relation value plus row/column statistics, local ranks, type/position encodings, and optional reliability descriptors. This may preserve element-level relation patterns that row/column profile projection washed out.

Falsification:

- entry-vector GT under the same loss performs no better than rowcol-profile GT;
- permutation stress shows the reader ignores stable matrix semantics;
- learned scores track only `mat_mean` with no improvement or local correction.

## H3 — Known-normal low-score is necessary; multi-center ROCC is only an ablation

Known-normal low-score constraints should be retained as the clean supervised signal. Multi-center normal manifold may help but carries strong inductive bias and hyperparameter load, so it should be tested, not assumed.

Falsification:

- normal-low-score-only variants collapse or fail to separate known normals;
- multi-center variants dominate robustly across seeds without hurting `mat_mean` alignment.

## H4 — Barrier/reference-dropout are auxiliary stabilizers, not main objectives

Collapse barrier and reference-dropout consistency may improve stability, but they should not define the anomaly target.

Falsification:

- adding these terms improves no-leakage monitors but consistently reduces AUC/AP or increases teacher-pair violations.
