# DualRefGAD Constraint-Calibrated Reference Relation

> 创建时间：2026-05-21  
> 状态：🟡 活跃 / 新探究  
> 上游证据：Route2/Route2.5 response matrix、C-LEG3 / old-exact 对齐、normal-only residual probe、Matrix AE negative finding  
> 第一步实验：C-LEG3 固定 regime 下的“约束校准前门禁”与 response-matrix 可学习空间审计

## 1. 中心问题

当前证据显示，DualRefGAD 的强信号并不来自更大的可学习头，而主要来自 reference relation 本身：在 C-LEG3 / old-exact reference regime 下，response matrix 的 `mat_mean` 可恢复约 0.80 AUC；而 current reference regime 下，`mat_mean` 方向反转，`neg_mat_mean` 才有约 0.65–0.67 AUC。

因此，本探究不再把“换一个 matrix head”作为默认路线，而是问一个更窄的问题：

> 在固定 C-LEG3 reference regime 后，能否用两条 normal-only 约束，把无权重 reference relation score 校准成稳定、可解释、非 GGAD 换皮的异常分数？

这里的两条文字约束是：

1. 对已知正常节点，最终异常分数应低；
2. 如果引入伪异常生成，则伪异常的分数应高于其源正常节点。

## 2. 研究边界

本探究第一阶段只研究“学习信号是否成立”，不声明完整方法贡献。

- `D_psi` 第一版直接等同于 C-LEG3 / old-exact 里的 PCA 正常子空间残差，不包装为新模块。
- 最终分数暂不加入 `W` 或额外 reliability gate，先验证无权重 reference relation score 能否被约束校准。
- 伪异常生成存在 GGAD-like 风险，因此必须做门禁对照，不能先写成贡献。
- anomaly labels 只允许用于诊断评估、AUC/AP/autopsy，不参与训练、早停或 checkpoint 选择。

## 3. 关键上游证据

| 证据 | 结论 | 对本探究的约束 |
|---|---|---|
| normal-only residual probe | `margin + correction` 几乎只是 margin compression，路线关闭 | 不能再做 additive patch |
| Matrix AE probe | 普通 normal-only Matrix AE 5-seed 低于 scalar baseline | 不继续堆 AE 容量 |
| current-regime k scan | `normal_k/anom_k` 扫描不能修复方向反转，最好约 0.67 AUC | 不重复 broad k scan |
| old-setting alignment | C-LEG3 / old refs 恢复约 0.80 AUC，且 GT 1 层对照也接近 | 第一阶段固定 C-LEG3 reference regime |
| LEG3 decomposition | row/column/entry 与 margin 不完全同序，存在可学习空间候选 | 第一实验先审计哪些 family 有学习价值 |

## 4. 第一阶段假设

- **H1：无权重 relation score 已经提供强正控制。** 若 R0 已接近 C-LEG3 `mat_mean/margin` 上界，则新增训练必须证明不是单调校准。
- **H2：PCA 残差方向可作为伪异常生成的正控制。** 如果 residual-guided pseudo anomaly 优于随机方向和 GGAD-like 扰动，才可能继续方法化。
- **H3：若 R1 与 R3 难以区分，则伪异常路线应降级为 GGAD-like 对照，而非贡献。**
- **H4：若 R4（PCA 残差直接打分）已解释全部增益，则训练头没有必要。**

## 5. 第一阶段实验门禁

第一阶段按 R0–R4 组织：

- **R0：无训练 relation score** — 只用 C-LEG3 response matrix 的 scalar/row/column/entry family。
- **R1：PCA 残差伪异常 + 两条约束** — 训练低容量 scorer，使 known-normal 分数低，residual-guided pseudo anomaly 高。
- **R2：随机方向伪异常** — 判断是否“任何扰动都有效”。
- **R3：GGAD-like 扰动对照** — 判断是否只是 GGAD 换皮。
- **R4：PCA 残差直接打分** — 判断残差范数/方向本身是否已经解释全部收益。

第一步只准备并优先执行 R0/R4 + decomposition gate；R1–R3 在 R0/R4 结果证明仍有学习空间后再启动。

## 6. 成功/停止规则

继续到 R1–R3 的最低条件：

1. C-LEG3 固定 regime 下，非训练 family 存在至少一个与 `margin` 不完全同序的候选：Spearman 不应接近 1，top-k overlap 不应接近 1；
2. 候选 family AUC/AP 接近或超过 `mat_mean/margin`，或能解释某些 seed 的互补 top-k；
3. R4 不能完全解释候选收益，否则训练伪异常 scorer 没有必要。

停止或降级条件：

- R1 与 R2/R3 表现无差异：说明不是 residual direction 的机制优势；
- R1 只学到 `margin` 或 `D_psi` 的单调函数：降级为校准/换皮；
- pseudo anomaly 分数提升但真实 anomaly AUC/AP 不动：说明约束只塑造训练分布，不改善异常排序。

---
*Created: 2026-05-21 | Hermes / research-investigation skill*
