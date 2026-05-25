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
