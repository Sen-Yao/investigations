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
