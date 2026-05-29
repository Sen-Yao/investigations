# Hypothesis — Loss Portfolio Selection + MTL Optimizer

## H1: A1 should dominate A0 on no-leakage stability

如果 A0 的持续压低 unlabeled compactness 与持续 anti-collapse loss 干扰异常排序，那么 A1 的 hinge/dead-zone + barrier 形态应当改善以下现象：

- known-normal energy plateau 更稳定；
- trimmed subset Jaccard 不再剧烈震荡；
- covariance/effective-rank 不塌缩；
- 最终诊断 AUC/AP 至少不显著低于 A0。

## H2: A2 only passes if consistency improves perturbation stability without freezing wrong ranking

reference-view consistency 应优先改善 reference dropout ranking stability；若 AUC/AP 下降且 score-tail 变窄，则说明它可能只是稳定了错误排序。

## H3: A3 only passes if pair reliability helps noisy reference-pair suppression

pair reliability weighting 的价值不在于新增一个大模型，而在于降低低质量 reference-pair 对 response matrix 的扰动。如果它只让训练更平滑但没有改善 view stability / tail mass / final diagnostic metrics，应降级为 monitor 或 ablation，不进入主 portfolio。

## H4: multi-objective optimizer must be downstream of portfolio selection

PCGrad/FAMO/CAGrad 的第一轮比较只应在 A0→A3 中通过 gate 的 3–4 项上执行。若所有新增项都不改善 allowed monitors，则优化器不应被用来“拯救”错误目标。
