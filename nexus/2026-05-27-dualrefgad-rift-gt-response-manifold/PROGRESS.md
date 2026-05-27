# Progress — RIFT-GT Response Manifold

## 2026-05-27 — Investigation created

Created new Nexus investigation for the method internally named **RIFT-GT** — Response-Informed Flow Transformer.

Immediate context:

- The previous response-matrix evidence says C-LEG3 `mat_mean` is a strong positive control, so the response matrix is not garbage.
- V0 Set-D-MiT failed because its pseudo-supervision objective did not transfer to real anomaly ranking, not because the matrix had no signal.
- The user accepted the broad method flow: C-LEG3 response matrix → GT reader → optional position/role encoding → ROCC/normal-response-manifold loss → final anomaly score.
- The user raised two important corrections before code:
  1. vectorized-response / direction information may be underrepresented if scoring uses only `||z-c||_2^2`;
  2. `z_v = GT_theta(...)` must be explicitly defined before implementation.

Operational cleanup before creating this investigation:

- Checked current scheduler: the ABCD watchdog/publisher cron IDs recorded in the previous investigation are already absent; no additional removal needed.
- Checked background processes: no tracked Hermes background processes are running.
- Checked `~/investigations` git status: two modified probe utility scripts were py_compile-verified and committed as `0214ed7 Optimize DualRefGAD probe utility scripts`.
- Added reusable no-leakage metric definitions to `research/references/dualrefgad-riftgt-no-leakage-metrics.md` and linked them from the research skill.

Next steps before code:

1. Publish a full method report explaining RIFT-GT for readers without prior context.
2. In that report, answer the vectorized-direction concern and define the GT readout formally.
3. After report approval, implement RIFT-R0 minimal probe through `experiment-runner`.
