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

## 2026-06-01 — HCCS-25 smoke implementation and preflight

Implemented the first self-contained `MatGuardGT` single-run prototype:

- script: `experiments/scripts/matguardgt_single_run.py`
- smoke sweep config: `experiments/configs/matguardgt_smoke_sweep.yaml` = 5 seeds × 8 variants = 40 runs
- main sweep config: `experiments/configs/matguardgt_main_960_sweep.yaml` = 5 seeds × 24 variants × 2 d_model × 2 lr = 960 runs

Design choices implemented in code:

- response-matrix tokens are vector profiles (`row`, `col`, or `rowcol`), not scalar entries;
- `mat_mean` is used only as a high-confidence pairwise ranking teacher;
- no residual-over-`mat_mean` path is implemented;
- known-normal low-score constraint is a primary loss term;
- barrier and reference-dropout are sidecar ablations, default disabled.

Validation completed:

- local `python3 -m py_compile` passed;
- runner `validate --profile sweep` passed for both smoke and main configs;
- remote HCCS-25 `py_compile` passed under `/home/linziyao/.conda/envs/DualRefGAD/bin/python`;
- HCCS-25 single-seed smoke completed successfully.

Smoke command summary:

- host: HCCS-25
- output: `experiments/outputs/smoke_seed0_variant0.json`
- log: `experiments/logs/smoke_seed0_variant0.log`
- matrix shape: `[46564, 16, 4]`
- smoke setting: seed 0, variant 0, 2 epochs, 2 steps/epoch
- diagnostic MatGuardGT AUC/AP: `0.6198 / 0.1421`
- diagnostic `mat_mean` AUC: `0.7263`
- MatGuardGT-minus-`mat_mean` AUC: `-0.1065`
- Spearman(MatGuardGT, `mat_mean`): `-0.0131`

Interpretation:

- This smoke is code-path validation only, not method-quality evidence.
- It confirms that the MatGuardGT input object, vector-token GT reader, pairwise teacher loss, known-normal low-score constraint, output JSON, and remote HCCS-25 runtime are operational.
- Because the smoke is intentionally tiny, the lower diagnostic AUC does not yet adjudicate the method.

Next gate:

- Launch the 40-run smoke sweep first through `experiment-runner` if user approves.
- Only if the 40-run sweep has stable completion and non-degenerate score diagnostics should we launch the 960-run main sweep.
