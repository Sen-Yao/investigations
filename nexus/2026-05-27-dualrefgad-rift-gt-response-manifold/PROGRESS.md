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

## 2026-05-31 — Closure: scalar-entry RIFT-GT demoted

Closure status: **closed as a scalar-entry tokenization investigation; not promoted as the next main route**.

Evidence archived under `experiments/`:

- RIFT-R0 no-position scalar-entry reader finished. Best recorded variant: `r0_mean_lr1e3`, AUC≈0.702, AP≈0.242, while C-LEG3 `mat_mean` was AUC≈0.810 and AP≈0.559.
- RIFT-R1 row/column/rank/role scalar-entry identity variants finished. Best recorded variant: `r1_rowcol_role_attn_lr1e3_lu10`, AUC≈0.719, AP≈0.335, still below `mat_mean`.
- P0 ROCC-MC finished. Best recorded AUC variant: `p0_rocc_mc_k2_attn_rowcol_rank_lu25`, AUC≈0.726; best AP variant among inspected summaries: `p0_rocc_mc_k4_mean_rowcol_rank_lu25`, AP≈0.421. Both remain below `mat_mean`.
- Across these probes, the learned scalar-entry readers usually show low Spearman correlation with `mat_mean`, meaning they are not merely copying the scalar baseline; however, their diagnostic AUC/AP remain substantially lower, so complementarity is not yet usable as a deployable anomaly score.

Scientific decision:

1. The C-LEG3 response matrix remains valid as a positive-control signal object. The failure is not “matrix has no signal.”
2. The scalar-entry tokenization route is now demoted: treating each `M_ij(v)` entry as a token creates a weak token ontology and empirically failed to recover the strong `mat_mean` signal.
3. Further work should not continue by adding more loss terms, deeper readers, or broader R1/R2 scalar-entry sweeps.
4. The next investigation should correct the token object itself: use row/column response profiles as vector tokens, then test whether profile-level reference relation carries signal beyond scalar aggregation.

Handoff to new investigation:

- New investigation: `2026-05-31-dualrefgad-profile-token-readout`.
- Role: tokenization-correction probe for the broader RIFT-GT family.
- Boundary: old scalar-entry RIFT-GT evidence becomes a negative control / cautionary predecessor, not an active main route.

