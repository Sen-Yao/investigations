# Offset 信息论与物理意义探究

**创建时间**：2026-03-31
**状态**：规划阶段
**类型**：理论探究（非 trick 性创新）

---

## 🎯 核心问题

### 研究动机

**当前问题**：Concat 等 trick 性方法缺乏理论支撑，无法解释：
- 为什么 Offset 有时有效有时无效？
- Offset 到底提供了什么信息？
- GT 如何感知和利用这些信息？

**研究目标**：建立 Offset 的理论基础，指导高质量 GT 创新设计。

---

## 📋 研究问题

### Q1: 信息量问题

> Offset 向量到底含有多少「信息」？

- Offset 的信息熵是多少？
- 与原始 Hop 特征相比，信息增益是多少？
- 信息分布在哪些维度？

### Q2: 区分能力问题

> 这些信息可用于区分正常点和异常点吗？

- 哪些维度的信息最具区分力？
- 区分能力与数据集特性的关系？
- Fisher 判别分析？

### Q3: 物理意义问题

> Offset 有什么物理意义？

- Offset 代表什么？（变化向量？趋势？关系？）
- 正/异常节点的 Offset 有什么物理差异？
- 如何从图论/网络科学角度解释？

### Q4: GT 利用问题

> 应该如何被下游 GT 更好地使用？

- 当前 GT 如何处理 Offset？（注意力分析）
- GT 的哪些层/头在关注 Offset？
- 如何设计机制让 GT 更好地感知 Offset？

---

## 🧪 初步假设

### 假设 1: 信息增益假设

Offset 不是原始信息的简单变换，而是提供了**增量信息**。

**验证方法**：
- 计算互信息 I(Offset; Label) vs I(Original; Label)
- 信息增益 = I(Offset; Label) - I(Original; Label)

### 假设 2: 物理意义假设

Offset 代表节点的**局部演化趋势**。

- 正常节点：演化趋势稳定（方向一致）
- 异常节点：演化趋势异常（方向偏离）

### 假设 3: GT 感知假设

当前 GT 可能**没有充分利用** Offset 信息。

**原因**：
- Offset 被当作普通 token，缺乏专门处理
- 注意力机制可能被原始 token 主导
- 缺乏显式的"变化感知"机制

---

## 📅 研究计划

### 阶段 1: 信息论分析（1-2 天）

1. **信息熵计算**
   - 计算各数据集上 Offset 的信息熵
   - 对比 Original, Delta, Offset 的信息量

2. **互信息分析**
   - I(Offset; Label)
   - I(Original; Label)
   - I(Offset; Original) — 冗余度

3. **信息增益分解**
   - 哪些维度贡献最多信息？
   - PCA 分解后分析

### 阶段 2: 物理意义探索（2-3 天）

1. **可视化分析**
   - Offset 在低维空间的分布
   - 正/异常节点的 Offset 差异

2. **统计特征分析**
   - Offset 范数分布
   - Offset 方向分布
   - Hop-wise 特征

3. **网络科学解释**
   - Offset 与节点度/中心性的关系
   - Offset 与社区结构的关系

### 阶段 3: GT 注意力分析（2-3 天）

1. **注意力权重分析**
   - 哪些 token 获得最多注意力？
   - 不同层/头的注意力分布

2. **消融实验**
   - 移除 Offset token，观察注意力变化
   - 只保留 Offset token，观察性能

3. **信息流分析**
   - Offset 信息在 GT 中的传播路径
   - 最终表示中 Offset 的贡献

### 阶段 4: 创新机制设计（3-5 天）

基于以上分析，设计 **GT 原生创新**：

1. **变化感知注意力**
   - 显式建模 Offset 的注意力机制
   - 让 GT "知道" 这是变化信息

2. **双流架构**
   - 原始 token 流 + 变化 token 流
   - 独立编码，交叉注意力

3. **物理先验注入**
   - 将物理意义编码为约束
   - 如"正常变化方向一致"

---

## 🔬 实验设计

### 实验 1: 信息熵对比

```python
def compute_entropy(features):
    """计算特征的信息熵"""
    # 连续变量：使用 k-NN 估计
    pass

def compute_mutual_information(features, labels):
    """计算特征与标签的互信息"""
    pass

# 对比
H(original) vs H(offset)
I(original; label) vs I(offset; label)
I(offset; original)  # 冗余度
```

### 实验 2: GT 注意力分析

```python
def analyze_attention_weights(model, data):
    """分析 GT 对不同 token 的注意力"""
    # 提取注意力权重
    # 分析 offset token 的注意力占比
    pass
```

### 实验 3: 信息增益分解

```python
def information_gain_decomposition(offsets, labels):
    """分析哪些维度贡献最多信息增益"""
    # 逐维度计算 I(offset_dim; label)
    # 排序，找出最重要维度
    pass
```

---

## 📊 预期产出

| 产出 | 内容 |
|------|------|
| **理论发现** | Offset 的信息量、物理意义 |
| **机制设计** | GT 原生创新（非 trick） |
| **论文素材** | 理论分析 + 创新设计 |

---

## 🔗 相关探究

- `2026-03-31-offset-semantic-enhancement` — H1 方向语义分析（已完成）

---

_从 trick 走向理论，从排列组合走向机制创新。_