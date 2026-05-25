# Hypotheses — Constraint-Calibrated Reference Relation

## H1 — 先固定 reference regime，而不是继续换生成器

C-LEG3 / old-exact reference regime 是当前 response matrix 强信号的正控制。只有在固定该 regime 后，学习信号的成败才可解释；否则任何提升/下降都可能来自 reference construction 改变。

**验证方式：** 第一实验固定 `old_exact_080_regime`，`normal_k=4`，`anom_k=16`，不扫 broad k，不改变 reference generator。

## H2 — 两条文字约束是合理的训练信号，但不是自动贡献

约束：known-normal 分数低；residual-guided pseudo anomaly 分数高于源 normal。它们符合 normal-only semi-supervised 协议，但必须证明不是普通 GGAD 扰动的重命名。

**验证方式：** R1 与 R2/R3 对照，并检查 Spearman/top-k overlap/score decomposition。

## H3 — `D_psi` 第一版应作为 C-LEG3 PCA 残差正控制

`D_psi` 不作为新模块命名贡献。它是 C-LEG3 / old-exact 中已经存在的 PCA 正常子空间残差，作用是提供：

1. 正控制：normal-manifold deviation 是否与异常排序有关；
2. 伪异常方向：生成 residual-guided pseudo anomaly；
3. 解释边界：若 R4 已足够强，则无需训练头。

## H4 — 无权重分数优先

第一阶段最终分数不加入 `W` 或 learned reliability gate。若无权重 relation score 无法被两条约束校准，加入 `W` 只会扩大自由度并增加过拟合/换皮风险。

## H5 — 可学习空间必须通过“非同序”证明

若新分数与 margin/mat_mean Spearman 接近 1，且 top-k overlap 接近 1，即使 AUC 微升也不应视为新信号。
