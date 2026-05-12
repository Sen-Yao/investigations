# 探究总结：课题组 Token 信息利用率研究

## 基本信息

| 项目 | 内容 |
|------|------|
| **主题** | 验证课题组是否有 Token 信息利用率相关研究 |
| **时间** | 2026-04-01 18:23-18:28 |
| **耗时** | ~5 分钟 |
| **结论** | ✅ 有相关研究，但应用场景不同 |

---

## 核心发现

### NAGphormer（课题组已有）

| 组件 | 说明 |
|------|------|
| **Hop2Token** | 将 multi-hop neighborhood 转换为 token 序列 |
| **Token Hop Attention** | Transformer 学习不同 hop 的权重 |
| **Attention-based Readout** | 自适应学习 hop 重要性 |

**作者**：Jinsong Chen, Chenyang Li

### 应用场景差异

| 方法 | 应用场景 | Token 策略 |
|------|---------|-----------|
| **NAGphormer** | Node Classification | Hop2Token |
| **VecGAD** | Graph Anomaly Detection | Hop Token + RDV |
| **方向 1** | GAD Token 利用率 | **技术方案已有** |

---

## 关键结论

### 方向 1 的评估

| 维度 | 评估 |
|------|------|
| **技术新颖性** | ⭐ 低（NAGphormer 已有）|
| **应用新颖性** | ⭐⭐ 中（GAD vs Node Classification）|
| **研究价值** | ⭐⭐ 中等 |

### 建议

| 建议 | 说明 |
|------|------|
| **不重复 NAGphormer** | 技术方案已有 |
| **聚焦 GAD 特有问题** | 异常节点的 Token 模式 |
| **结合 VecGAD** | Token + RDV 的深度融合（方向 2）|

---

## 下一步建议

| 优先级 | 方向 | 原因 |
|--------|------|------|
| ⭐⭐⭐ | **方向 2：Token + RDV 深度融合** | 最具新颖性，符合 VecGAD |
| ⭐⭐ | **方向 3：Hop-Attention 机制** | GAD 特有的注意力设计 |
| ⭐ | **方向 1：信息利用率** | NAGphormer 已有类似研究 |

---

_探究完成时间: 2026-04-01 18:28_