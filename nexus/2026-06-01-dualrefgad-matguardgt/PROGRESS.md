# Progress — DualRefGAD MatGuardGT

## 2026-06-01 — Investigation skeleton created

Created as the successor to `2026-05-31-dualrefgad-profile-token-readout` after profile-token ROCC and direct `mat_mean + P0` fusion were demoted.

User-approved first-stage constraints:

- Use **high-confidence pairwise ranking distillation** from `mat_mean`.
- Do **not** introduce interval guardrail yet.
- Do **not** introduce residual-over-`mat_mean` final score yet.
- Keep known-normal low-score constraints seriously.
- Treat multi-center ROCC as a test-only ablation because its inductive bias and hyperparameter load are high.
- Also test collapse barrier and reference-dropout consistency.

Immediate next steps:

1. Implement MatGuardGT script with rowcol-profile and entry-vector token readers.
2. Define loss variants and 500–1000 run sweep budget.
3. Validate with experiment-runner before launch.
4. Run a smoke probe before any formal sweep.
