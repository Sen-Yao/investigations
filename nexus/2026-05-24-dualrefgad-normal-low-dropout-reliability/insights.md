# Insights

## 2026-05-24 initial insight

当前最重要的结论不是增加更多 loss，而是先验证一个最小机制：

> 先证明 `normal-low + reference-dropout consistency` 是否能让 shallow reliability gate 学到 reference reliability，而不是退化为 margin proxy。

这一方向保留了 DualRefGAD 的 reference-relation 核心，但避免早期用 reference ranking、entropy、anti-hub、residual-guided hard negatives 等目标把机制判断混在一起。

## Working interpretation

- `L_normal-low` 提供 anomaly detection 的 normality anchor：已知正常点不应被判异常。
- `L_ref-drop` 提供 reliability anchor：可靠判断不应依赖偶然 reference subset。
- 两者组合应先回答“可靠 reference relation 能否稳定存在”，而不是直接追求完整 semi-supervised anomaly ranking。

## What would be meaningful evidence

有意义的证据不是单一 AUC 改善，而是机制层面的同时满足：

1. known normal nodes 的异常分数下降；
2. reference subset perturbation 下输出更稳定；
3. reliability gate 不只是 raw margin 的单调替身；
4. 不需要异常标签、伪异常或 hard negatives 即可产生可解释的 early validation signal。

## Risk to watch

如果 gate 与 margin proxy 高度一致，则所谓 reliability 可能只是旧分数的再参数化；此时应停止扩大 loss 组合，优先做诊断与消融。
