# PROGRESS — Constraint-Calibrated Reference Relation

## 2026-05-21 — Investigation created

Timestamp: 2026-05-21 20:48 

Created a new investigation from the discussion around:

- two normal-only textual constraints;
- `D_psi` as C-LEG3 / old-exact PCA residual;
- pseudo anomaly / GGAD-like risk;
- no `W` in the first final score;
- need for R0–R4 gates before claiming a method.

### Context confirmed

- Investigation archive root: `/home/openclawvm/investigations/nexus/`.
- Current evidence source: `2026-05-19-dualrefgad-route25-matrix-autoencoder`.
- Strong control output: `route25_stage_a_old_setting_alignment_probe.json`.
- Existing decomposition probe output: `route25_leg3_response_matrix_decomposition_probe.json`.
- Current-regime k scan output: `route25_ref_length_mat_mean_scan.json`.

### First-step decision

The first executable step is not full pseudo-anomaly training. It is a runner-validatable gate:

> Fix C-LEG3 / old_exact reference regime and audit whether scalar, row, column, entry-normal-manifold, and PCA residual families leave learnable space beyond `margin` / `mat_mean`.

Reason: if R0/R4 already explain the signal, R1–R3 training would likely become a GGAD-like or margin-calibration exercise.

### Prepared artifacts

- `README.md` — investigation framing and boundaries.
- `hypothesis.md` — hypotheses H1–H5.
- `insights.md` — current evidence and interpretation rules.
- `experiments/configs/step1_cleg3_decomposition_gate.yaml` — runner-validatable probe config.
- `experiments/scripts/README.md` — execution artifact note; first step reuses the trusted Route2.5 decomposition script instead of duplicating science logic.


## 2026-05-21 — Step-1 config validation and HCCS-25 preflight

Timestamp: 2026-05-21 20:44 CST

### Local validation

- `python3 -m py_compile` passed for the reused decomposition probe script.
- `experiment.py validate --profile probe` passed for `experiments/configs/step1_cleg3_decomposition_gate.yaml`.
- Validator warning: tokenization/reference modes are implicit in the script variant definition; acceptable for this gate because the config fixes `variants=old_exact_080_regime`, but the final report should state the C-LEG3 mode mapping explicitly.

### HCCS-25 live preflight

- Experiment-runner alias query `--remote-host HCCS-25` failed because the SSH alias is not resolvable in this environment.
- Loaded `hccs-cluster` and used the documented direct HCCS-25 route for a read-only probe.
- Nezha WebSocket probe failed with `InvalidMessage: did not receive a valid HTTP response`; this is recorded as monitor-source failure, not machine failure.
- Direct SSH succeeded: host `VM-2-25-ubuntu`.
- GPU state: 8 × RTX 2080 Ti, all showing 0 MiB used, 11011 MiB free, 0% util at probe time.
- Remote project/data: `~/DualRefGAD` present; `~/DualRefGAD/dataset/elliptic.mat` present.

### Execution readiness boundary

The next action can register and launch the pure probe through the runner-registered probe pattern, using HCCS-25's direct SSH route or an experiment-runner-compatible alias. This session intentionally stops at preparation unless the user explicitly approves actual experiment start.


## 2026-05-21 — Runner job prepared, not launched

Registered a runner job in `created` state for the first-step pure probe:

- Job ID: `exp_20260521_205308_step1_cleg3_decomposition_gate_prepared`
- Profile/kind: `probe` / `probe`
- Status: `created` only; no remote experiment process was started.
- Remote host label: `HCCS-25-direct`
- Planned GPU set: `0,1,2,3,4,5,6,7`
- Planned log: `/home/linziyao/DualRefGAD/logs/step1_cleg3_decomposition_gate.log`
- Planned summary: `/home/linziyao/DualRefGAD/experiments/outputs/step1_cleg3_decomposition_gate.json`

Launch requires explicit user approval. Recommended launch pattern: sync/reuse the trusted script on HCCS-25, remote `py_compile`, run one Hermes-tracked SSH command with `--devices 0,1,2,3,4,5,6,7`, then update this runner job from `created` to `running/finished` through job_store.


## 2026-05-21 — Report published

Published planning report:

- URL: https://report.senyao.org/reports/2026/05/21/dualrefgad-constraint-calibrated-rrl-plan-2026-05-21.html
- Report repo commits: `1c6b7b5` (report), `f726ece` (metadata refresh)
- Publisher verification: Cloudflare Access service-token HTTP 200.

The report states that the job is prepared but not launched, and asks for explicit approval before starting the first-step probe.


## 2026-05-22 — Step-1 C-LEG3 decomposition gate finished

Runner job `exp_20260521_205308_step1_cleg3_decomposition_gate_prepared` finished successfully.

- Remote exit code: 0
- Runtime: 217.2 sec
- Output pulled to: `experiments/outputs/step1_cleg3_decomposition_gate.json`
- Progress pulled to: `experiments/outputs/step1_cleg3_decomposition_gate.progress.json`
- Log pulled to: `experiments/logs/step1_cleg3_decomposition_gate.log`
- Status: `finished`
- Errors: `[]`

### Main result

The winner is `mat_mean` for all 5 seeds. Aggregate decision: `SIMPLE_SCALAR_SUMMARY_STILL_DOMINATES`.

| family | AUC mean ± std | interpretation |
|---|---:|---|
| scalar_matrix_summary (`mat_mean`) | 0.8168 ± 0.0054 | strongest and most stable |
| row_normal_anchor (`row_mean_min`) | 0.8093 ± 0.0038 | close, complementary, but not better |
| centroid_margin (`margin`) | 0.7928 ± 0.0054 | weaker than `mat_mean` in this fixed C-LEG3 gate |
| entry_normal_manifold | 0.7827 ± 0.0076 | not enough to justify PCA-residual-only scorer |
| column_anomaly_reference | 0.7743 ± 0.0027 | weaker |
| proxy_diagnostic (`degree`) | 0.3296 ± 0.0005 | not a degree artifact |

Complementarity checks:

- Spearman(best, margin): 0.7074 ± 0.0036
- Top-5% Jaccard(best, margin): 0.5792 ± 0.0425
- Abs Spearman(best, degree): 0.0874 ± 0.0182
- Diagnostic anomaly ratio in anomaly references: 0.1089 ± 0.0015

### Scientific reading

This gate says the C-LEG3 response matrix itself is strong: the simplest full-matrix mean beats centroid margin on every seed. Row-anchor summaries are close and less同序 with margin, but they do not beat `mat_mean`. Entry-normal-manifold/PCA residual scores do not explain or exceed the main signal. Therefore R1-R3 pseudo-anomaly training should not be launched as a presumed method contribution yet; if launched, it must be framed as a second gate that tries to beat a strong `mat_mean` baseline, not as a default next step.


## 2026-05-22 — Step-2 mat_mean vs margin failure autopsy finished

Runner job `exp_20260522_142856_step2_mat_mean_margin_failure_autopsy` finished successfully.

- Runtime: 199.1 sec
- Output: `experiments/outputs/step2_mat_mean_margin_failure_autopsy.json`
- Progress: `experiments/outputs/step2_mat_mean_margin_failure_autopsy.progress.json`
- Log: `experiments/logs/step2_mat_mean_margin_failure_autopsy.log`
- Status: `finished`
- Errors: `0`

### Aggregate metrics

| score | AUC mean ± std | AP mean ± std |
|---|---:|---:|
| `mat_mean` | 0.8134 ± 0.0105 | 0.5608 ± 0.0440 |
| `margin` | 0.7980 ± 0.0060 | 0.5199 ± 0.0285 |

Decision: `MAT_MEAN_REORDERS_BOUNDARY_BY_RESCUING_ANOMALIES_AND_REMOVING_MARGIN_FALSE_POSITIVES`.

Top-K protocol: K equals the number of test anomalies per seed; labels are diagnostic-only for autopsy categories.

| category | count mean ± std | short meaning |
|---|---:|---|
| rescued anomalies: mat only true positives | 360.2 ± 21.9 | anomalies missed by margin top-K but recovered by mat_mean top-K |
| introduced false positives: mat only normal | 1065.0 ± 115.9 | normal nodes newly introduced by mat_mean top-K |
| lost anomalies: margin only true positives | 206.2 ± 73.0 | anomalies kept by margin top-K but dropped by mat_mean |
| removed false positives: margin only normal | 1219.0 ± 91.7 | normal nodes removed when switching from margin to mat_mean |

### Mechanistic contrast

- Rescued anomalies have high anomaly-reference diagnostic purity: 0.8530 ± 0.0163, low response dispersion (`mat_std`: 0.1273 ± 0.0072), and very low rejection/residual proxy values. They look like clean reference-supported anomalies that margin did not rank high enough.
- Removed margin false positives have very low anomaly-reference diagnostic purity: 0.0772 ± 0.0222, large response dispersion (`mat_std`: 0.4424 ± 0.0067), and large row/column ranges. They look like centroid-margin artifacts: margin is near 1.0, but pairwise matrix is heterogeneous and weak on average.
- Lost anomalies also have margin near 1.0 but `mat_mean` much lower; their anomaly-reference purity is still high-ish (0.7300 ± 0.0646) but matrix dispersion/range is large. This suggests mat_mean penalizes heterogeneous anomaly-reference support; this is a real cost, not only a cleanup.
- Introduced false positives have high rejection/residual proxies and low anomaly-reference purity; mat_mean still introduces many normal nodes. So mat_mean improves AUC by net boundary reordering, not by eliminating all false positives.

### Scientific interpretation

`mat_mean` improves over `margin` because it can down-rank many high-margin normal false positives whose pairwise response matrix is heterogeneous and supported by low-purity anomaly references. It also rescues hundreds of anomalies that have cleaner anomaly-reference support. However, it sacrifices some high-margin anomalies whose matrix support is heterogeneous. Therefore the next useful research question is not "train a pseudo-anomaly head now", but "can we preserve the rescued-anomaly benefit while preventing heterogeneous true anomalies from being over-penalized?"


## 2026-05-25 — LSD oracle autopsy probe prepared; remote launch blocked

Prepared runner-registered pure probe `lsd_oracle_autopsy_probe` for Learning Signal Discovery.

Local artifacts created:

- `experiments/scripts/lsd_oracle_autopsy_probe.py`
- `experiments/configs/lsd_oracle_autopsy_probe.yaml`
- `experiments/lsd_oracle_autopsy_probe_BLOCKED.md`

Validation:

- Local `python3 -m py_compile` passed.
- `experiment.py validate --profile probe` passed.

Remote read-only preflight on HCCS-25 direct route passed before launch attempt:

- Host: `VM-2-25-ubuntu`
- Project: `~/DualRefGAD` present
- Dataset: `~/DualRefGAD/dataset/elliptic.mat` present, 36M
- Python: `/home/linziyao/.conda/envs/DualRefGAD/bin/python`
- Runtime: torch `2.0.0+cu117`, CUDA available, 8 GPUs
- GPU state: GPUs 0-7 all 0 MiB used / 0% util at preflight

Runner job registered but not launched:

- Job ID: `exp_20260525_171733_lsd_oracle_autopsy_probe`
- Profile/kind: `probe` / `probe`
- Status: `created`
- Planned remote log: `/home/linziyao/DualRefGAD/investigations/2026-05-21-dualrefgad-constraint-calibrated-reference-relation/experiments/logs/lsd_oracle_autopsy_probe.log`
- Planned remote output: `/home/linziyao/DualRefGAD/investigations/2026-05-21-dualrefgad-constraint-calibrated-reference-relation/experiments/outputs/lsd_oracle_autopsy_probe.json`
- Planned remote progress: `/home/linziyao/DualRefGAD/investigations/2026-05-21-dualrefgad-constraint-calibrated-reference-relation/experiments/outputs/lsd_oracle_autopsy_probe.progress.json`

Blocked point: the minimal remote sync / remote mkdir / remote py_compile command was denied by tool policy (`BLOCKED: Command denied by user. Do NOT retry this command.`). No remote write occurred, no Hermes background SSH process was started, and no watchdog cron was created.

## 2026-05-25 — LSD oracle autopsy probe launched and finished

Runner-registered pure probe `lsd_oracle_autopsy_probe` was launched on HCCS-25 after the earlier remote-sync block was resolved by explicit continuation.

Execution handles:

- Job ID: `exp_20260525_171733_lsd_oracle_autopsy_probe`
- Profile/kind: `probe` / `probe`
- Remote host: `HCCS-25-direct` (`VM-2-25-ubuntu`)
- Hermes background session: `proc_b45f7c86382c`
- Status: `finished`
- Progress: `5/5` seeds
- Runtime: ~188 sec

Artifacts pulled back by pure-probe watchdog:

- Output: `experiments/outputs/lsd_oracle_autopsy_probe.json`
- Progress: `experiments/outputs/lsd_oracle_autopsy_probe.progress.json`
- Log: `experiments/logs/lsd_oracle_autopsy_probe.log`
- Watchdog script: `/home/openclawvm/.hermes/scripts/lsd_oracle_autopsy_probe_watchdog_watchdog.py`

Protocol boundary:

- Frozen C-LEG3 / old-exact response regime.
- No training.
- Anomaly labels used only for report-only AUC/AP and oracle-autopsy categories.
- Labels did not enter loss, early stopping, checkpoint selection, or method claim.

Aggregate baseline result in this probe:

| score | AUC mean ± std | AP mean ± std |
|---|---:|---:|
| `margin` | 0.7826 ± 0.0101 | 0.4898 ± 0.0439 |
| `mat_mean` | 0.7845 ± 0.0308 | 0.5000 ± 0.0649 |
| `mat_mean - margin` | +0.0018 ± 0.0269 AUC | +0.0102 ± 0.0619 AP |

Per-seed note: the positive-control relationship is weaker and less stable than the earlier Step-1/Step-2 outputs; seed 0 and seed 3 have `mat_mean < margin`, while seeds 1/2/4 have `mat_mean > margin`. Treat this as a protocol/regime comparison warning, not as a final contradiction.

Oracle-autopsy highlights:

- `mat_mean` rescues true anomalies: `330.0 ± 38.4` per seed.
- `mat_mean` also introduces false positives: `1226.0 ± 132.2` per seed.
- `margin`-only false positives removed by `mat_mean`: `1382.2 ± 83.1` per seed; these have high matrix dispersion (`mat_std ≈ 0.4506`) and low anomaly-reference diagnostic ratio (`≈ 0.0738`).
- True anomalies lost by `mat_mean`: `173.8 ± 112.6` per seed; these show high heterogeneous matrix support (`mat_std ≈ 0.3679`) and nontrivial anomaly-reference ratio (`≈ 0.7119`).
- Spearman(`mat_mean`, `margin`) ≈ `0.6996 ± 0.0153`: the two scores are related but not identical.
- Spearman(delta, degree) ≈ `-0.0241 ± 0.0130`: the delta is not explained by degree.

Runner gates recorded by the JSON:

- `label_boundary`: PASS
- `teacher_positive_control`: PASS, but scientifically weak/unstable because mean AUC gain is small and has high seed variance.
- `not_margin_only`: PASS
- `anti_degree_shortcut`: PASS

Immediate interpretation: Learning Signal Discovery remains viable, but the next step should focus on stabilizing row/column reliability and heterogeneous-support handling rather than simply distilling raw `mat_mean`.

## 2026-05-25 — C-LEG3 strict reproduction audit finished

Runner-registered strict reproduction audit `cleg3_strict_reproduction_audit` finished on HCCS-25.

Execution handles:

- Job ID: `exp_20260525_184627_cleg3_strict_reproduction_audit`
- Profile/kind: `probe` / `probe`
- Remote host: `HCCS-25-direct` (`VM-2-25-ubuntu`)
- Hermes background session: `proc_45c8492b42c3`
- Status: `finished`
- Progress: `5/5` seeds
- Output: `experiments/outputs/cleg3_strict_reproduction_audit.json`
- Progress: `experiments/outputs/cleg3_strict_reproduction_audit.progress.json`
- Log: `experiments/logs/cleg3_strict_reproduction_audit.log`

Strict protocol:

- Fixed C-LEG3 / old-exact response regime.
- No training and no oracle classifier/head.
- Only `margin` and `mat_mean` are audited as primary scores.
- Sequential execution, single device, explicit `data_split_seed=seed` to avoid global RNG/thread interference.
- Formula check requires `mat_mean == response_matrix.mean(axis=(1,2))`.

Aggregate comparison:

| stage | `mat_mean` AUC mean ± std | `margin` AUC mean ± std | `mat_mean - margin` AUC |
|---|---:|---:|---:|
| Step-1 decomposition gate | 0.8168 ± 0.0054 | 0.7928 ± 0.0054 | +0.0241 |
| Step-2 failure autopsy | 0.8134 ± 0.0105 | 0.7980 ± 0.0060 | +0.0154 |
| LSD oracle autopsy | 0.7845 ± 0.0308 | 0.7826 ± 0.0101 | +0.0018 |
| Strict reproduction audit | 0.8009 ± 0.0182 | 0.7952 ± 0.0064 | +0.0057 |

Per-seed strict result:

| seed | strict `mat_mean` AUC | strict `margin` AUC |
|---:|---:|---:|
| 0 | 0.8200 | 0.7938 |
| 1 | 0.7901 | 0.7960 |
| 2 | 0.7711 | 0.7991 |
| 3 | 0.8060 | 0.7840 |
| 4 | 0.8170 | 0.8030 |

Audit conclusion:

1. The `mat_mean` formula is strictly reproduced: all five seeds have direct matrix-mean max absolute difference `0.0`.
2. The LSD aggregate drop is not caused by a changed `mat_mean` formula. The likely cause is protocol/RNG drift: earlier scripts used global random state around `load_mat()` under multi-threaded seed execution, while the strict audit pins `data_split_seed=seed` and runs sequentially.
3. Strict audit does not prove exact per-seed equality to Step-1/Step-2 because those historical outputs did not save split fingerprints and may themselves include concurrent global-RNG effects. It establishes a new auditable baseline with split fingerprints.
4. Scientific reading: cite Step-1/Step-2 as historical evidence, cite strict audit as the reproducible baseline, and avoid citing LSD aggregate as clean contradiction without explaining split/RNG protocol drift.
