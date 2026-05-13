# Hypotheses — DualRefGAD Reference Geometry Anatomy

## H1: Margin-only 的有效性主要来自一侧 reference signal，而不是两侧均衡贡献

如果 normal-reference distance 或 anomaly-reference distance 单独已经接近 margin 表现，说明 margin 的主要信号来源偏向单侧。后续应围绕该侧重构 scoring，而不是继续保持模糊的双侧差值叙事。

**诊断证据：**
- normal-side score AUC/AP；
- anomaly-side score AUC/AP；
- margin 与两侧 score 的 Pearson/Spearman；
- per-seed / per-split 稳定性。

## H2: Margin top-k false positives 具有可解释的结构模式

如果 top-k FP 集中在高 degree、特殊 hop descriptor、或某类 reference response pattern，那么 margin 的错误不是随机噪声，而是暴露了可以修正的结构偏置。

**诊断证据：**
- top-k FP/FN 的 degree / local statistics；
- descriptor/hop contribution；
- reference response vector signature；
- 与 true positives 的差异。

## H3: Scalar margin 丢失了 reference response distribution 的信息

即使 scalar margin 已经很强，节点到多个 references 的 response vector 可能包含额外结构：例如 response variance、entropy、top-reference concentration 或 normal/anomaly response shape mismatch。

**诊断证据：**
- response vector statistics 的 AUC/AP；
- 在控制 margin 后，distributional feature 是否仍能区分 anomaly/normal 或 FP/TP；
- response distribution 是否能解释 top-k failure case。

## H4: 如果 response distribution 没有独立信号，则 DualRefGAD 应收束为简洁 reference-margin 方法

若所有 response-vector 派生指标都只是 margin 的单调变换，且无法解释 FP/FN，那么不应继续堆复杂模块。方法叙事应简化为 reference-margin geometry，并把复杂 learned head 全部删除。

**关闭条件：**
- distributional features 与 margin 高度相关；
- residualized features AUC/AP 接近随机或无实际收益；
- top-k FP/FN 没有可利用的稳定模式。
