# Progress — Reliability / Heterogeneous Proxy Map

## 2026-05-25 20:07 CST — Investigation created

Created new Nexus investigation for the short pure probe requested by SenYao:

- Investigation: `2026-05-25-dualrefgad-reliability-heterogeneous-proxy-map`
- Upstream: `2026-05-21-dualrefgad-constraint-calibrated-reference-relation`
- First probe: `cleg3_oracle_to_proxy_candidate_signal_map`

Scope:

- runner-registered pure probe;
- no training;
- frozen C-LEG3 / `old_exact_080_regime`;
- built-in row/column reliability and heterogeneous-support no-training readouts;
- anomaly labels diagnostic-only for oracle categories and report metrics.

Immediate next step: implement probe config/script and validate with `experiment.py validate --profile probe` before any remote launch.

## 2026-05-25 20:29 CST — `cleg3_oracle_to_proxy_candidate_signal_map` finished
Runner-registered pure probe finished successfully.
- Job ID: `exp_20260525_202222_cleg3_oracle_to_proxy_candidate_signal_m`
- Hermes session: `proc_1dfab57a1c28`
- Host/GPU: HCCS-25 / GPU0
- Status: `finished`; errors: `0`; seeds: `0,1,2,3,4`
- Output: `experiments/outputs/cleg3_oracle_to_proxy_candidate_signal_map.json`
- Progress: `experiments/outputs/cleg3_oracle_to_proxy_candidate_signal_map.progress.json`
- Log: `experiments/logs/cleg3_oracle_to_proxy_candidate_signal_map.log`

Protocol invariants:

- Frozen C-LEG3 / `old_exact_080_regime`, `normal_k=4`, `anom_k=16`.
- Sequential seed execution with `data_split_seed=seed` to avoid `load_mat()` global RNG/thread drift.
- No training; anomaly labels diagnostic-only for AUC/AP and oracle top-K categories.
- Formula check: `mat_mean == response_matrix.mean(axis=(1,2))`, max abs diff `0.0000 ± 0.0000`.

Aggregate baseline:

- `margin` AUC: `0.7952 ± 0.0064`
- `mat_mean` AUC: `0.8009 ± 0.0182`

Oracle category counts at top-K where K equals the number of test anomalies:

- `introduced_false_positives_mat_only_normal`: `1227.6 ± 146.5`
- `lost_anomalies_margin_only_true_positive`: `225.0 ± 201.3`
- `removed_false_positives_margin_only_normal`: `1361.8 ± 95.8`
- `rescued_anomalies_mat_only_true_positive`: `359.2 ± 60.4`

Top no-training candidate readouts by AUC:

- `consensus_minus_fragmentation`: AUC `0.7993 ± 0.0184`, Spearman vs `mat_mean` `0.9674 ± 0.0009`, recovered lost anomalies `12.6 ± 10.9`, reintroduced removed FPs `31.0 ± 10.9`
- `trimmed_mean_10_90`: AUC `0.7980 ± 0.0199`, Spearman vs `mat_mean` `0.9920 ± 0.0003`, recovered lost anomalies `26.8 ± 16.0`, reintroduced removed FPs `160.2 ± 40.1`
- `mixture_support`: AUC `0.7980 ± 0.0090`, Spearman vs `mat_mean` `0.9008 ± 0.0041`, recovered lost anomalies `111.0 ± 162.6`, reintroduced removed FPs `245.2 ± 52.8`
- `top2_row_mean`: AUC `0.7887 ± 0.0070`, Spearman vs `mat_mean` `0.8641 ± 0.0070`, recovered lost anomalies `128.2 ± 174.9`, reintroduced removed FPs `485.6 ± 52.8`
- `top25_entry_mean`: AUC `0.7883 ± 0.0016`, Spearman vs `mat_mean` `0.6428 ± 0.0081`, recovered lost anomalies `141.0 ± 193.2`, reintroduced removed FPs `612.2 ± 57.6`

Decision: `CANDIDATE_READY_FOR_SHALLOW_RELIABILITY_GATE_REVIEW`. This does **not** mean a trainable head is already approved; it means the pure-probe evidence is strong enough to design the next shallow reliability gate around `consensus_minus_fragmentation` / reliability-aware support, while treating `trimmed_mean_10_90` as likely too close to raw `mat_mean` and `mixture_support` as useful but FP-expensive.

## 2026-05-26 12:59:24 — Layer 0 fixed-formula gate probe completed

- Job: `exp_20260526_124955_cleg3_layer0_fixed_formula_gate_probe`
- Status: `finished`
- Decision: `LAYER0_FIXED_GATE_HAS_USABLE_PROXY_SIGNAL_REVIEW_REQUIRED` — Layer 0 固定门控有可用 proxy 信号，但仍需人工 review；可以考虑进入 Layer 1 label-free shallow gate，不应直接升级为高容量 head。
- Best diagnostic strategy: `L0_reliability_gate_a0.5_b0.5` (reliability_blend)
- Best AUC: `0.8047`; delta vs mat_mean: `+0.0038`
- Best AP: `0.5240`; delta vs mat_mean: `-0.0095`
- mat_mean AUC: `0.8009 ± 0.0182`
- Protocol: no training; fixed formulas only; labels diagnostic-only.

## 2026-05-26 13:52:22 — Layer 1 label-free shallow gate probe completed

- Job: `exp_20260526_133750_cleg3_layer1_label_free_shallow_gate_pro`
- Status: `finished`
- Decision: `LAYER1_LABEL_FREE_GATE_NOT_PROMOTED_USE_AS_DIAGNOSTIC` — Layer 1 不应被提升为方法组件；它只能作为 reliability proxy 诊断，下一步应回到 reference / fragmentation 机制。
- Best Layer 1: `L1_lfgate_q0.05_qf0.1_aa0.5_am0.1_l20.01`
- mat_mean AUC/AP: `0.8009` / `0.5335`
- Best Layer 1 AUC/AP: `0.8037` / `0.5341` if available
- Continuation: metric=False, rho=True, shortcut=True
- Report predecessor: https://report.senyao.org/reports/2026/05/26/dualrefgad-layer1-next-step-decision-2026-05-26.html

## 2026-05-26 — Investigation closure

Closure decision recorded in `insights.md`: this investigation has answered the reliability / heterogeneous proxy-map question and should stop here. The next scientific unit should be a separate **DualRefGAD Learning Signal Recovery** investigation, because the question has shifted from reading the existing response matrix to recovering a reliable, trainable, label-free target.

