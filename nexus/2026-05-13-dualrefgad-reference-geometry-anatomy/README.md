# 2026-05-13-dualrefgad-reference-geometry-anatomy

## 研究问题

前序探究 `2026-05-13-dualrefgad-normal-only-residual-probe` 已经关闭 additive residual 路线：`margin + correction` 在 elliptic 5-seed 上几乎不改变排序，learned correction 主要表现为 margin compression / calibration，而不是新的 anomaly ranking signal。

因此本探究不再继续堆 learnable head，而是转向 **reference geometry anatomy**：解剖 DualRefGAD 的 `margin-only` 信号到底来自哪里、失败在哪里，以及是否存在 margin scalar 没表达出来的 reference response distributional signal。

核心问题：

> DualRefGAD 的 margin-only 信号到底来自哪类 reference geometry？它的 top-k failure case 是否暴露出可被 distributional inconsistency / normal-manifold deviation 利用的剩余结构？

## 定位

这是一次 **mechanism autopsy / 机制解剖型探究**，不是新方法 sweep。

第一阶段只做无训练诊断：

- 不启动 formal sweep；
- 不训练 learned head；
- 不追求 SOTA；
- 不使用 anomaly label 做模型选择；
- anomaly label 仅用于离线诊断评估。

## 必做诊断项

| 诊断项 | 要回答的问题 | 预期产出 |
|---|---|---|
| normal-reference distance | anomaly 是否远离 normal reference？ | normal-side score AUC/AP、分布、top-k label composition |
| anomaly-reference distance | anomaly 是否靠近 anomaly-like reference？ | anomaly-side score AUC/AP、分布、top-k label composition |
| margin decomposition | margin 有效性来自哪一边？ | margin 与 normal/anomaly side 的相关性、贡献比例 |
| reference purity | reference set 里是否混了错误结构？ | reference 邻域 purity、prototype response pattern |
| hop/descriptor contribution | 哪些 descriptor 真正贡献排序？ | per-descriptor / per-hop AUC/AP、相关性、消融式诊断 |
| top-k failure case | margin 排名前列的 false positives 是什么类型？ | FP/FN 节点结构特征、reference response pattern |

## 额外重点：reference response vector analysis

不要只看 scalar margin。对每个节点保留并分析：

```text
node -> normal_refs response vector
node -> anomaly_refs response vector
```

派生指标：

- response mean / std / entropy / skewness；
- top-reference concentration；
- normal/anomaly response gap；
- response vector variance；
- margin residualized distributional features；
- top-k FP/FN 的 response signature。

目标是判断是否存在一类比 scalar margin 更有解释力的 **multi-reference distributional inconsistency**。

## 成功标准

本探究成功不等于立即刷出更高 AUC/AP，而是给出明确 decision table：

| 后续路线 | 是否继续 | 判据 |
|---|---|---|
| reference construction | TBD | reference purity / reference side contribution 是否暴露瓶颈 |
| normal-manifold deviation | TBD | normal-side distance 是否强且有未被 margin 表达的残差信号 |
| multi-reference distributional inconsistency | TBD | response vector 统计是否区分 FP/FN 或 anomaly/normal |
| learned head / residual | No | 前序探究已关闭 |

## 阶段计划

### Phase 1 — Anatomy without training

1. 找到或补充脚本，导出 per-node normal/anomaly reference responses、margin、label、descriptor/hop 信息。
2. 计算 scalar side diagnostics：normal-side、anomaly-side、margin 的 AUC/AP 与相关性。
3. 计算 response vector diagnostics：mean/std/entropy/concentration/skewness 等。
4. 分析 top-k FP/FN：结构、descriptor、reference response signature。
5. 写入 `experiments/outputs/reference_geometry_anatomy_summary.json` 和 `insights.md`。

### Phase 2 — Decide whether to design no-head score

只有 Phase 1 发现 margin 未表达的 distributional signal 时，才设计 fixed-formula score，例如：

- `distributional_inconsistency_score`
- `normal_manifold_deviation_score`
- `reference_response_entropy_score`

如果 Phase 1 显示 margin 已经吃干榨净，则收束为更简洁的 reference-margin 叙事，不继续发明复杂模块。

## 关联探究

- `2026-05-13-dualrefgad-normal-only-residual-probe`: additive residual route closed；证明 learned correction 主要是 margin compression。
- `2026-05-09-semisupervised-negative-signal-for-dualrefgad`: 前置 DualRefGAD / semi-supervised negative signal 探究。

---
*Created: 2026-05-13 | Nexus Agent*
