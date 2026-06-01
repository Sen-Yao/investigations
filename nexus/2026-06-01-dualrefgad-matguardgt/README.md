# DualRefGAD MatGuardGT Investigation

> Successor investigation after profile-token ROCC demotion. Goal: test whether a tensor / entry-vector Graph Transformer reader can learn useful response-matrix structure under high-confidence `mat_mean` pairwise ranking distillation and known-normal low-score constraints.

## Lineage

Predecessors:

- `2026-05-21-dualrefgad-constraint-calibrated-reference-relation`: established C-LEG3 / old-exact response matrix and `mat_mean` positive control.
- `2026-05-27-dualrefgad-rift-gt-response-manifold`: scalar-entry RIFT-GT / old P0 variants; useful incremental but below `mat_mean`.
- `2026-05-31-dualrefgad-profile-token-readout`: row/row-column profile-token ROCC sweep; demoted because profile readers remained below `mat_mean`; P0 complementarity was too sparse for direct fusion.

## Research question

Can MatGuardGT train a tensor/entry-vector GT reader that preserves high-confidence `mat_mean` ordering while learning additional response-matrix structure from known-normal labels and reference perturbations?

## Scope

Included:

- tensor / entry-vector token design;
- rowcol-profile reader as controlled baseline;
- high-confidence pairwise `mat_mean` ranking distillation;
- known-normal low-score constraint as the primary supervised signal;
- optional tests for multi-center normal manifold, collapse barrier, and reference dropout consistency;
- loss-balance sweep within a 500–1000 run budget.

Excluded for the first stage:

- interval guardrail against `mat_mean`;
- residual-over-`mat_mean` final score;
- pseudo-anomaly generation / hard-negative contrast;
- true anomaly labels for training, early stopping, or hyperparameter selection.

## First promotion gate

A useful first-stage MatGuardGT variant should:

1. Beat or match `mat_mean` AUC/AP under 5-seed mean, with at least 4/5 seeds non-degrading on the selected metric.
2. Reduce high-confidence `mat_mean` teacher-pair violation rate relative to an unguarded normal-only reader.
3. Preserve top-1% / top-5% `mat_mean` precision rather than damaging the strongest baseline region.
4. Show nontrivial but not total alignment with `mat_mean`: Spearman near 1 indicates imitation, near 0 with weak metrics indicates teacher signal failed.
5. Provide score-level autopsy: `mat_mean` FN repaired vs `mat_mean` TP destroyed.
