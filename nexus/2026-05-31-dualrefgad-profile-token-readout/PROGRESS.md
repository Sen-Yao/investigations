# Progress — DualRefGAD Profile-Token Readout

## 2026-05-31 — Investigation skeleton created

Created this investigation as the next step after closing the scalar-entry RIFT-GT route.

Immediate lineage:

- `2026-05-21-dualrefgad-constraint-calibrated-reference-relation` established C-LEG3 / old-exact response matrix and `mat_mean` as the positive control.
- `2026-05-27-dualrefgad-rift-gt-response-manifold` tested scalar-entry RIFT-GT variants and closed them as the main route because they remained below `mat_mean`.
- The new question is not whether to add more capacity; it is whether the token object should be corrected to row/column response profiles.

Current state:

- Documentation skeleton created.
- No experiment has been launched.
- Next step is to prepare a runner-validatable row-profile probe config and script, then request/record explicit launch approval before training.

## Planned first gate

Gate 0: implementation/design preflight only.

- Define exact row-profile token shape.
- Define normalization using train-normal statistics only.
- Define no-leakage checkpoint monitor.
- Define scalar and capacity controls.
- Register or prepare the job in `created` state before launch if experiment-runner is used.

Gate 1: row-profile reader smoke/probe.

- Fixed C-LEG3 / old-exact response matrices.
- Row-profile token reader.
- 5 seeds only after smoke validation passes.
- Diagnostic labels evaluation-only.

## 2026-06-01 — Closure: profile-token ROCC demoted; successor MatGuardGT opened

Closure status: **closed as a standalone ROCC/profile-token route; not promoted as the next main method route**.

Evidence archived or referenced:

- Formal 1120-run profile-token sweep config: `experiments/configs/profile_token_rocc_1120_sweep.yaml`.
- Single-run implementation: `experiments/scripts/profile_token_rocc_single_run.py`.
- Aggregated working evidence from the 1120-run sweep: best profile-token reader remained below the fixed response-matrix positive control `mat_mean`.
  - `mat_mean` baseline: AUC `0.8104 ± 0.0068`, AP `0.5593 ± 0.0279`.
  - best profile-token reader: AUC about `0.6783 ± 0.0694`, with near-zero / unstable Spearman against `mat_mean`.
- Follow-up exploratory score-complementarity diagnostic tested `mat_mean + old P0` rank fusion using exported node-level scores for seed0/seed1.
  - At the true-anomaly top-k budget, average `P0 TP / mat_mean FN` was about `87.5`, while `mat_mean TP / P0 FN` was about `1296.5`.
  - At top 1%, `mat_mean` precision was about `0.994`, while P0 top-1% precision was about `0.010`.
  - Simple and light nonlinear fusion did not show stable gain over `mat_mean`; P0-only TP exists but is too sparse and unstable to justify direct fusion.

Scientific decision:

1. The response matrix remains a valid signal source because the fixed `mat_mean` positive control is strong and stable.
2. Profile-token ROCC failed primarily as an objective/readout route, not as proof that the response-matrix signal is empty.
3. Low Spearman with `mat_mean` is not sufficient evidence of useful complementarity; exported-score diagnostics showed a large imbalance between P0-only true positives and mat-mean-only true positives.
4. Do not continue by increasing profile-token reader capacity or by adding heavier ROCC-style compactness objectives.
5. Handoff to a successor route: use `mat_mean` as a high-confidence ranking teacher/guardrail, and train a tensor/entry-vector GT reader under known-normal low-score constraints plus optional auxiliary losses.

Handoff to new investigation:

- New investigation: `2026-06-01-dualrefgad-matguardgt`.
- Role: explore **MatGuardGT** — tensor/entry-vector GT with high-confidence pairwise `mat_mean` ranking distillation.
- Boundary: this investigation becomes predecessor evidence and negative/control baseline. It should not be extended with broader ROCC/profile-token sweeps unless the successor identifies a specific missing control.
