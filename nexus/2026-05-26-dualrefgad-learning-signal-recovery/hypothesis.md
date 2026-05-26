# Hypotheses — DualRefGAD Learning Signal Recovery

## H1 — 当前瓶颈是 trainable target recovery，不是 head 容量不足

Layer 1 label-free shallow gate 只有弱提升（上游报告记录 ΔAUC 约 +0.0028、ΔAP 约 +0.0006），并且 top-K autopsy 显示救回 anomaly 的同时引回更多 false positives。因此继续加 head 容易学习 shortcut；必须先恢复一个可靠目标 `T(v)`。

**验证方式：** Phase A/B/C 先在固定公式层检查候选目标能否改善 AP 或降低 FP reintroduction，并报告与 `margin` / `mat_mean` / degree / rejection 的相关性。

## H2 — fragmentation 必须按来源分解，而不是作为整体惩罚项

异常和 false positive 都可能表现为异质支持。若把 fragmentation 作为单一负信号，会继续丢掉 heterogeneous true anomalies。

**验证方式：** Phase B 拆出 normal-side dispersion、deviation-side dispersion、pair interaction sparsity、hub dominance、regime conditional components，并比较它们在 recovered anomaly 与 reintroduced FP 上的方向差异。

## H3 — 可靠 reference relation 应具备 dropout stability 与非 hub-dominance

若强响应主要由少数 reference pair 或 hub/reference shortcut 主导，它不应成为学习目标。

**验证方式：** Phase C 对 reference dropout、row/column contribution、effective reference count、pair concentration、descriptor-regime drift 做 seed-level 诊断。

## H4 — Phase D 只判断 target readiness，不直接训练大 head

即使 Phase B/C 找到候选 `T(v)`，Phase D 也只定义浅层 label-free objective 的准备度：目标来源、训练边界、loss 构造、负例/正常约束、可审计性与反证条件。

**验证方式：** 输出 readiness decision：`READY_FOR_SHALLOW_TARGET_LEARNING` / `FIXED_FORMULA_ONLY` / `REFERENCE_CONSTRUCTOR_BACKOFF`，并给出 evidence chain。

## H5 — 成功标准以 AP / FP reintroduction / anomaly retention 为核心

AUC 微升不够；GAD 关心 top-ranked 区域。一个候选若只改善全局排序但不改善 AP 或 top-K 质量，不应推进。

**验证方式：** 所有候选报告 AUC/AP、ΔAP、rescued anomaly retention、FP reintroduction、top-K anomaly density、Spearman/top-K overlap 与 shortcut audit。
