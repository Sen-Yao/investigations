# Literature Notes

## 关于“现有 GAD 很少有 Transformer 工作”这一说法

当前判断：

> 这个说法在方向上成立，但如果直接写成绝对判断，还不够严谨。

更稳的版本应当是：

> 与大量 GNN-based、contrastive、reconstruction-based GAD 方法相比，基于 Transformer 的 GAD 工作仍相对较少，尤其是显式围绕 tokenization / token family 设计全图 reference interaction 的工作更少。

## 当前调研到的支持点

### 1. 过去内部调研记忆
- 2026-04-01 研究调研中已记录：NAGphormer 是 tokenized graph transformer，但主要应用在 node classification，不是 GAD。
- 这说明“Graph Transformer 很多，但直接做 GAD 的不多”这个论点是有内部证据支撑的。

### 2. Web 调研（初步）
- 搜到 dynamic graph anomaly detection via transformer 相关工作，说明 Transformer 并非完全空白；
- 也搜到 Graph Transformer survey，但更多是在 broader graph learning 语境；
- 未看到大量静态 attributed GAD 中围绕 tokenization 做 retrieval-style reference token 的方法。

## 当前应避免的表述

### 不够严谨
- “现有 GAD 几乎没有 Transformer 工作”
- “Transformer 的最大优势就是抓远程节点”

### 更严谨
- “相比主流 GNN-based GAD 方法，Transformer-based GAD 工作仍相对较少”
- “Transformer 的一个潜在优势在于，它更适合建模来自不同信息来源的非局部 token 交互”
- “目前尚缺少显式围绕全图 reference retrieval tokenization 的 GAD 方法”

## 当前结论

如果写进新探究文档，建议使用下面这版表述：

> 现有图异常检测研究中，基于 Transformer 的方法相较于 GNN-based 方法仍相对较少。尤其在 tokenization 设计层面，现有方法大多仍围绕局部邻域或局部传播展开，较少显式利用整图范围内的 reference retrieval 来构造 anomaly-aware token family。
