# Hypothesis: normal-low + reference-dropout reliability

## Core hypothesis

DualRefGAD 的早期可靠性学习可以用两个最小、可解释、第一性原理一致的训练约束启动：

1. `L_normal-low` 让已知正常节点保持低异常分数，从 normality 出发定义 anomaly detector 的基础方向。
2. `L_ref-drop` 让同一节点在合理 reference 子集扰动下保持异常分数 / reliability 输出一致，从而学习 reference relation 的稳定性。

若该假设成立，模型应在不使用异常标签、不生成伪异常、不引入 hard negative 的情况下，表现出比纯 margin proxy 更稳定的 shallow reliability gate。

## Loss 1: `L_normal-low`

### Definition intent

对已知 normal set `L_n` 中节点施加低异常分数约束：

- normal nodes should receive low anomaly scores;
- training supervision uses only known normal labels;
- no anomaly labels are used for optimization;
- no pseudo-anomalies are generated.

### First-principles rationale

异常检测的第一性原理不是先学习“异常长什么样”，而是先学习 normality：已知正常点不应被判为异常。该约束最直接、可解释，也最不容易引入早期伪信号。

## Loss 2: `L_ref-drop`

### Definition intent

对 reference set 做合理 dropout / subsampling，要求同一 target node 的输出在不同 reference 子集下保持一致：

- anomaly score consistency under reference subset perturbations;
- reliability gate consistency under reference subset perturbations;
- sensitivity to accidental single-reference dependence should be penalized.

### First-principles rationale

一个节点是否异常不应强依赖某个偶然 reference 点。可靠的 reference relation 应对合理 reference 子集扰动稳定。因此该项学习的是 **reference reliability**，不是 anomaly label。

## Why only these two losses now

用户已明确指出：上一版五项损失函数对早期验证会灾难。早期最多两个损失函数，并应选择最本质、最符合异常检测第一性原理、可解释性强的目标。

本 investigation 因此不把以下项放入第一步训练 loss：

- reference ranking
- entropy regularization
- anti-hub regularization
- residual-guided hard negatives

这些项只允许作为 diagnostics / report metrics / stop gates，以避免早期目标过多、机制混叠、以及通过伪相关提升指标。

## Main mechanistic question

> normal-low + reference-dropout consistency 能否让 shallow reliability gate 学到 reference reliability，而不是退化成 margin proxy？

需要比较至少以下现象：

- reliability gate 与 raw margin / scalar anomaly score 的相关性是否过高；
- reference dropout 后输出方差是否降低；
- normal nodes 的 anomaly score 是否稳定降低；
- gate 是否避免集中押注少数 hub/reference；
- 不使用异常标签时，是否仍有合理的 validation signal。

## Stop rules

第一步 probe 出现以下任一情况时应停止扩大训练目标，回到诊断：

1. `L_normal-low` 明显把所有节点分数压低，导致 anomaly separability 崩塌。
2. `L_ref-drop` 仅让输出变平滑，但没有降低 reference perturbation sensitivity。
3. reliability gate 与 margin proxy 高度同质，无法提供独立机制价值。
4. early validation 出现灾难性退化，并且 diagnostics 指向 loss 目标冲突。
5. 需要引入第三个训练 loss 才能维持基本稳定性。

## Continue rules

只有在以下条件基本满足时，才考虑进入下一阶段：

1. normal nodes 的分数确实稳定降低；
2. reference dropout sensitivity 降低；
3. reliability gate 不完全等价于 margin proxy；
4. 未使用异常标签与伪异常；
5. diagnostics 支持该机制值得扩展，而不是仅靠调参或偶然 seed。
