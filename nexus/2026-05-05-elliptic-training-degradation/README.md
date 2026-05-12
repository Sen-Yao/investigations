# Elliptic Training Degradation Diagnosis

**创建日期**: 2026-05-05  
**状态**: Planning  
**主题**: 诊断为什么 Elliptic 上 `target_ref_guided` 的 epoch-0 / early reference geometry 较强，但 BCE pseudo anomaly 训练后 final AUC/AP 明显变差。

## 背景

前序探究：`2026-04-30-reference-guided-pseudo-anomaly`

已完成正式 Elliptic sweep：

- Sweep ID: `w47qp7ah`
- WandB: https://wandb.ai/HCCS/VoxG/sweeps/w47qp7ah
- Objective: `target_ref_guided`
- Dataset: `elliptic`
- Grid: `pseudo_beta ∈ {0.1, 0.2, 0.3}`, `seed ∈ {0,1,2,3,4}`
- Status: 15/15 finished, failed=0

结果显示：

| pseudo_beta | final AUC | final AP | best_test_auc | best_test_ap |
|---:|---:|---:|---:|---:|
| 0.1 | 0.5183±0.0514 | 0.0935±0.0097 | 0.6725±0.0348 | 0.1472±0.0224 |
| 0.2 | **0.5359±0.0618** | **0.0997±0.0200** | **0.6836±0.0229** | **0.1530±0.0149** |
| 0.3 | 0.5317±0.0745 | 0.0988±0.0168 | 0.6772±0.0249 | 0.1506±0.0186 |

关键现象：

```text
best_test_auc ≈ 0.67-0.68
final AUC     ≈ 0.52-0.54
```

说明当前训练过程并没有稳定放大 reference geometry，反而在后续 epoch 中破坏了初始可分性。

## 核心问题

为什么 Elliptic 上 `target_ref_guided` 的 reference geometry 在 epoch 0 / early epoch 有较强异常排序能力，但经过 BCE pseudo anomaly training 后 final ranking 明显退化？

## 本 investigation 的目标

本 investigation 不直接提出新 objective，而是先定位退化来源：

1. 退化来自 encoder/reference geometry 被训练破坏？
2. 退化来自 classifier/head 对 pseudo anomaly 的过拟合？
3. 退化来自 pseudo anomaly direction 与真实异常方向不一致？
4. 退化是否表现为 model score 与 original reference score 的 rank correlation 下降？

## 非目标

暂时不做：

- 更细 pseudo_beta sweep；
- 新增复杂 loss / margin / temperature；
- 新增 decoder / VecGAD-style reconstruction path；
- 直接追 SOTA 性能。

本阶段目标是解释训练退化机制。
