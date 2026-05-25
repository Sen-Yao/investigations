# Hypotheses — Reliability / Heterogeneous Proxy Map

## H1 — 先做 oracle-to-proxy 映射，而不是直接训练

当前信息足以设计一个短 pure probe，但不足以直接选择一个 trainable learning signal。原因是 `mat_mean` 的收益来自 boundary reordering：它既移除 false positives，也牺牲 heterogeneous true anomalies。若不先把 oracle categories 映射到无标签 proxy，训练目标容易学成 `mat_mean` 蒸馏或 margin patch。

**验证方式：** 定义四类 top-K discordant categories，并检查候选 proxy 在类别之间的分布差异、效应方向和 seed 稳定性。

## H2 — Row / Column Reliability 是最优先的候选 proxy

Removed false positives 的矩阵通常表现为高 margin 但低多 reference 共识、高 row/column dispersion。若 row/column reliability 能无标签地区分这类节点，它可以解释 `mat_mean` 的 false-positive cleanup 机制。

**验证方式：** 计算 row mean / column mean 的 dispersion、range、entropy/effective-count、top-row/top-column dominance、trimmed/quantile consensus，并与 oracle categories 对齐。

## H3 — Heterogeneous-support Handling 不能等同于“惩罚异质性”

Lost anomalies 往往也有高 matrix heterogeneity。若简单惩罚异质性，会继续丢掉这些 true anomalies。可行 proxy 必须区分“孤立 spike / 低质量异质性”和“局部一致 / 高质量异质支持”。

**验证方式：** 比较 mixture-style support、top-row/top-column robust pooling、consensus-minus-fragmentation 在 lost anomalies 与 removed false positives 上的方向差异。

## H4 — 候选 readout 必须通过 anti-shortcut 审计

如果候选 proxy 主要由 degree、rejection、reference density 或 margin 单调解释，则它不是一个值得训练的学习信号。

**验证方式：** 保存 Spearman/Pearson-like correlation、top-K overlap、oracle category effect sizes，并报告 degree/rejection/residual_norm proxy 的相关性。

## H5 — 进入 shallow reliability gate 的条件是“机制清楚”，不只是 AUC 微升

短 probe 的目标是筛选学习信号，不是刷 leaderboard。即使某个 readout AUC 只接近 `mat_mean`，只要它稳定减少 lost anomalies 或提供互补 top-K，也可能值得训练；反过来，AUC 微升但与 `mat_mean` 同序则不值得推进。
