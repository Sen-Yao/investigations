# Insights — Reliability / Heterogeneous Proxy Map

## Current scientific position

本探究的核心判断是：当前掌握的信息足以启动一个 runner-registered pure probe，但不足以直接启动正式训练。最可靠的下一步是把 oracle categories 映射成无标签 proxy，并在同一个短 probe 内测试 row/column reliability 与 heterogeneous-support handling 的无训练 readout。

## Interpretation rule

候选学习信号必须同时满足三类约束：

1. **机制约束**：能解释 `mat_mean` 为什么移除 margin false positives，以及为什么会丢失 heterogeneous true anomalies；
2. **无标签约束**：proxy 本身不能依赖 anomaly label；label 只用于 autopsy 和报告；
3. **非同序约束**：不能只是 `margin` / `mat_mean` / degree / rejection 的单调重写。

## Baseline citation rule

未来讨论 C-LEG3 `margin` / `mat_mean` 行为时，优先引用 strict reproduction audit：

- `mat_mean` AUC：`0.8009 ± 0.0182`
- `margin` AUC：`0.7952 ± 0.0064`
- formula invariant：`mat_mean == response_matrix.mean(axis=(1,2))`，5 seed `max_abs_diff=0.0`
- protocol invariant：显式 `data_split_seed=seed`，顺序执行，保存 split fingerprints

Step-1/Step-2 仍可作为 positive-control/autopsy 证据，但要说明其 split fingerprint 不如 strict audit 可审计。

## Expected decision after first probe

第一 probe 结束后只允许三类结论：

1. **继续**：某个无训练 reliability/heterogeneous readout 通过 continuation gate，进入 shallow reliability gate；
2. **改 proxy**：oracle categories 可分，但当前 readout 不够好，需要重设 proxy；
3. **回退 reference constructor**：oracle categories 与无标签 proxy 对不上，说明当前 reference construction 本身不稳定，训练 gate 没有意义。

## 2026-05-25 probe result — proxy map is useful; do not jump to large head

The first oracle-to-proxy pure probe finished cleanly across 5 seeds. It reproduced the strict baseline exactly (`margin` AUC `0.7952 ± 0.0064`, `mat_mean` AUC `0.8009 ± 0.0182`, formula diff `0.0`).

Main result: the best candidate was not a high-capacity model but a hand-designed no-training readout, `consensus_minus_fragmentation`, with AUC `0.7993 ± 0.0184`. It is near raw `mat_mean`, less identical than trimmed mean (`Spearman≈0.967` vs `0.992`), and has a clearer mechanism: it rewards multi-reference consensus while penalizing fragmentation.

Important boundary: `consensus_minus_fragmentation` recovers fewer lost anomalies than `mixture_support`, but it also reintroduces far fewer removed false positives. `mixture_support` is more aggressive for heterogeneous true anomalies but FP-expensive. Therefore the next gate should not train a generic matrix head; it should test a shallow reliability/consensus gate with explicit false-positive reintroduction control.

Oracle-to-proxy evidence: lost anomalies differ from removed false positives in `consensus_minus_fragmentation` by about `+0.0885 ± 0.0215`, while rescued anomalies differ from introduced false positives by about `+0.0378 ± 0.0069`. This means heterogeneous-support handling is not hopeless: there is a no-label proxy direction, but it must be guarded against margin/mat_mean shortcut behavior and FP reintroduction.

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

