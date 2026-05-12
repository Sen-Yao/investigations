# 课题组 GT 研究调研报告

## 基本信息

| 项目 | 内容 |
|------|------|
| **调研日期** | 2026-04-03 13:18-13:24 |
| **调研成员** | Jinsong Chen, Gaichao Li, Chenyang Li, Kaiyuan Gao |
| **目的** | 评估方案C（Hop-物理意义注入注意力）是否与课题组已有研究重复 |

---

## 课题组 GT 研究成果

### 核心论文

| 论文 | 作者 | 会议/期刊 | 核心技术 |
|------|------|----------|---------|
| **NAGphormer** | Jinsong Chen, Kaiyuan Gao, Gaichao Li | ICLR 2023 | Hop2Token + Readout后加权 |
| **SignGT** | Jinsong Chen, Gaichao Li | arXiv 2023 | Signed Attention（有符号注意力）|
| **NAGphormer+** | Jinsong Chen, Chang Liu, Kaiyuan Gao, Gaichao Li | IEEE TBD 2025 | Hop2Token + Neighborhood Augmentation |
| **GTPool** | Gaichao Li, Jinsong Chen | IEEE TBD 2025 | Graph Transformer Pooling |
| **Rethinking Tokenized GT** | Jinsong Chen, Chenyang Li, Gaichao Li | arXiv 2025 | Tokenized GT 理论分析 |

---

## SignGT 详细分析

### 核心思想

标准注意力的局限性：
- 只产生正值 → 只捕获低频信息（平滑操作）
- 异质图需要高频信息

SignGT 的创新：
- **Signed Self-Attention (SignSA)**：根据语义相关性产生有符号注意力
- **Structure-aware FFN (SFFN)**：引入邻域偏置保留局部拓扑

### 技术特点

| 特点 | 说明 |
|------|------|
| **有符号注意力** | 正值捕获低频，负值捕获高频 |
| **频率响应** | 适应异质图的频率需求 |
| **应用场景** | Node Classification |

---

## 方案C 与 SignGT 对比

### 核心差异

| 维度 | SignGT | 方案C (Hop-物理意义注入) |
|------|--------|------------------------|
| **目标问题** | 高/低频信息捕获 | Hop 因果结构感知 |
| **技术手段** | 有符号注意力 | hop_bias 注入 |
| **注入位置** | 注意力值本身 | 注意力分数偏置 |
| **应用场景** | Node Classification (异质图) | **GAD (异常检测)** |
| **物理意义** | 频率响应 | Hop 传播因果 |

### 不重复的原因

| 问题 | SignGT | 方案C | 结论 |
|------|--------|-------|------|
| **核心机制不同** | 有符号注意力 | hop_bias 注入 | ✅ 不重复 |
| **物理意义不同** | 高低频信号 | Hop 因果结构 | ✅ 不重复 |
| **应用场景不同** | Node Classification | **GAD** | ✅ 不重复 |

---

## 新颖性评估

| 维度 | 评估 |
|------|------|
| **技术新颖性** | ⭐⭐⭐ 中高（hop_bias 是新设计）|
| **应用新颖性** | ⭐⭐⭐⭐ 高（GAD 场景未覆盖）|
| **理论新颖性** | ⭐⭐⭐ 中高（Hop 因果结构是新的角度）|

---

## 结论

### 方案C可行，但需差异化

| 建议方向 | 说明 |
|---------|------|
| **明确与 SignGT 区别** | SignGT 解决高低频，方案C解决 Hop 因果 |
| **聚焦 GAD 特有问题** | 异常检测需要的注意力模式不同于分类 |
| **结合 VecGAD** | 与现有工作融合，增强故事性 |

### 差异化策略

- SignGT：解决异质图（高/低频信息），应用 Node Classification，手段有符号注意力
- 方案C：解决 GT 学不到 Delta（Hop 因果结构），应用 GAD，手段 hop_bias + 因果注意力

---

## 参考资料

1. NAGphormer: https://arxiv.org/abs/2206.04910
2. SignGT: https://arxiv.org/abs/2310.11025
3. Gaichao Li DBLP: https://dblp.org/pid/322/4040

---
_调研完成时间: 2026-04-03 13:24_
