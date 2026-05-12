# 邻域 Delta 一致性研究报告

**日期**: 2026-04-16  
**作者**: Nexus  
**主题**: 探索 hop2token 中 Delta 序列的异常检测特性

---

## 一、Idea 来源

### 1.1 背景

在回顾 SmoothGNN 论文时，我们关注其提出的两个核心概念：

- **ISP (Individual Smoothing Pattern)**: 异常节点的表示更难被平滑
- **NSP (Neighborhood Smoothing Pattern)**: 邻域平滑程度可作为异常检测的系数

SmoothGNN 的核心假设是：异常节点偏离邻居，因此难以平滑。

### 1.2 问题提出

我们将 ISP/NSP 的概念映射到 VoxG 的 hop2token 架构中：

- **Hop features**: 多 hop 传播后的特征序列 $[X^{(0)}, X^{(1)}, ..., X^{(K)}]$
- **Delta vectors**: 相邻 hop 之间的变化量 $[\delta_1, \delta_2, ..., \delta_K]$

**核心问题**: Delta 序列是否能反映节点异常特性？是否存在类似 ISP/NSP 的可利用指标？

---

## 二、假设

### 2.1 原始假设（来自 SmoothGNN）

> **H0**: 异常节点偏离邻居 → Delta 与邻居不相似 → 邻域 Delta 一致性低

### 2.2 转换后的假设

> **H1**: 正常节点的 Delta 序列与邻居的 Delta 序列高度一致（高相关性）  
> **H2**: 异常节点的 Delta 序列与邻居的 Delta 序列不一致（低相关性）

**预期**: Normal > Anomaly（正常节点相关性更高）

---

## 三、形式化公式

### 3.1 Hop Features 定义

设节点 $v$ 的 hop features 为：

$$
H_v = [h_v^{(0)}, h_v^{(1)}, ..., h_v^{(K)}] \in \mathbb{R}^{(K+1) \times D}
$$

其中 $h_v^{(k)}$ 是第 $k$ hop 传播后的特征。

### 3.2 Delta Vector 定义

节点 $v$ 的 Delta vectors 为：

$$
\Delta_v = [\delta_v^{(1)}, \delta_v^{(2)}, ..., \delta_v^{(K)}] \in \mathbb{R}^{K \times D}
$$

其中：

$$
\delta_v^{(k)} = h_v^{(k)} - h_v^{(k-1)}
$$

### 3.3 邻域 Delta 一致性定义

对于节点 $v$，其邻域 Delta 一致性定义为：

$$
\text{NDC}(v) = \text{corr}(\Delta_v, \bar{\Delta}_{\mathcal{N}(v)})
$$

其中：
- $\mathcal{N}(v)$ 是节点 $v$ 的邻居集合
- $\bar{\Delta}_{\mathcal{N}(v)} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \Delta_u$ 是邻居 Delta 的平均
- $\text{corr}$ 是 Pearson 相关系数

**取值范围**: $[-1, 1]$

| NDC 值 | 含义 |
|--------|------|
| $\approx 1$ | Delta 序列与邻居高度相似 |
| $\approx 0$ | 无相关性 |
| $\approx -1$ | Delta 序列与邻居相反 |

---

## 四、伪代码逻辑

```python
def compute_neighborhood_delta_consistency(features, adj, K=6):
    """
    计算每个节点的邻域 Delta 一致性
    
    Args:
        features: 节点特征 [N, D]
        adj: 邻接矩阵 [N, N]
        K: hop 数量
    
    Returns:
        ndc: 邻域 Delta 一致性 [N]
    """
    # Step 1: 计算多 hop 特征
    hop_features = compute_hop_features(features, adj, K)  # [N, K+1, D]
    
    # Step 2: 计算 Delta vectors
    delta_vectors = hop_features[:, 1:] - hop_features[:, :-1]  # [N, K, D]
    
    # Step 3: 对每个节点计算邻域 Delta 一致性
    ndc = zeros(N)
    
    for i in range(N):
        # 找邻居
        neighbors = where(adj[i] > 0)
        
        if len(neighbors) == 0:
            ndc[i] = 0
            continue
        
        # 计算邻居平均 Delta
        neighbor_delta_mean = mean(delta_vectors[neighbors], axis=0)  # [K, D]
        
        # Flatten 并计算相关性
        node_delta_flat = delta_vectors[i].flatten()  # [K*D]
        neighbor_delta_flat = neighbor_delta_mean.flatten()  # [K*D]
        
        # Pearson 相关系数
        ndc[i] = correlation(node_delta_flat, neighbor_delta_flat)
    
    return ndc
```

---

## 五、详细实验结果

### 5.1 数据集统计

| 数据集 | 节点数 N | 异常数 | 异常率 |
|--------|---------|--------|--------|
| Photo | 7535 | 698 | 9.26% |
| Amazon | 11944 | 821 | 6.87% |
| Tolokers | 11758 | 2566 | 21.82% |
| Elliptic | 46564 | 4545 | 9.76% |
| Reddit | 10984 | 366 | 3.33% |
| t_finance | 39357 | 1803 | 4.58% |

### 5.2 邻域 Delta 一致性统计

| 数据集 | Normal Mean | Normal Std | Normal Var | Anomaly Mean | Anomaly Std | Anomaly Var |
|--------|-------------|------------|------------|--------------|-------------|-------------|
| Photo | -0.353 | 0.246 | 0.0605 | **-0.282** | 0.212 | **0.0449** |
| Amazon | -0.190 | 0.850 | 0.7221 | **+0.189** | 0.805 | **0.6481** |
| Tolokers | 0.546 | 0.346 | 0.1195 | **0.612** | 0.341 | **0.1162** |
| Elliptic | -0.692 | 0.375 | 0.1405 | **-0.494** | 0.458 | **0.2098** |
| Reddit | -0.9445 | 0.0989 | 0.0098 | **-0.9433** | 0.0850 | **0.0072** |
| t_finance | 0.1175 | 0.4618 | 0.2132 | **0.2432** | 0.4252 | **0.1808** |

### 5.3 方向一致性分析

| 数据集 | Mean 差异 | 方向 | KS p-value | 统计显著 |
|--------|----------|------|-----------|---------|
| Photo | +0.071 | Anomaly > Normal | 3.35e-14 | ✅ |
| Amazon | +0.379 | Anomaly > Normal | 3.97e-40 | ✅ |
| Tolokers | +0.066 | Anomaly > Normal | 1.88e-16 | ✅ |
| Elliptic | +0.198 | Anomaly > Normal | 1.40e-219 | ✅ |
| Reddit | +0.0013 | Anomaly > Normal | 4.15e-04 | ✅ |
| t_finance | +0.1257 | Anomaly > Normal | 3.10e-27 | ✅ |

**关键发现**: 所有 6 个数据集方向一致（Anomaly > Normal）

### 5.4 方差对比

| 数据集 | Normal Var | Anomaly Var | Ratio (A/N) | 异常更集中 |
|--------|-----------|-------------|-------------|----------|
| Photo | 0.0605 | 0.0449 | 0.74 | ✅ |
| Amazon | 0.7221 | 0.6481 | 0.90 | ✅ |
| Tolokers | 0.1195 | 0.1162 | 0.97 | ✅ |
| Elliptic | 0.1405 | 0.2098 | 1.49 | ❌ |
| Reddit | 0.0098 | 0.0072 | 0.73 | ✅ |
| t_finance | 0.2132 | 0.1808 | 0.85 | ✅ |

---

## 六、最终结论

### 6.1 核心发现

> **发现**: 异常节点的邻域 Delta 一致性显著高于正常节点（Anomaly > Normal），这与原始假设相反。

**统计证据**:
- ✅ 方向一致性: **6/6** 数据集
- ✅ 统计显著性: **6/6** 数据集 p < 1e-3
- ⚠️ 方差一致性: **5/6** 数据集异常方差更低（更集中）

### 6.2 与原始假设的对比

| SmoothGNN 假设 | 实验发现 |
|---------------|---------|
| 异常节点偏离邻居（ISP 大） | ❌ 相反：异常节点与邻居更相似 |
| 邻域平滑程度可区分异常 | ⚠️ 方向相反，但仍有区分能力 |

### 6.3 解释

**为什么异常节点与邻居更相似？**

可能原因：
1. **社区结构**: 异常节点形成紧密社区，邻居往往也是异常
2. **传播一致性**: 在异常社区内，Delta 模式高度一致
3. **"偏离邻居"假设不适用**: 对于社区型异常，节点与邻居传播模式相似

**方差更低意味着什么？**

- 异常节点的 NDC 分布更集中（方差低）
- 说明异常节点传播模式更**一致化**
- 正常节点传播模式更**多样化**

### 6.4 实际区分能力分析

| 数据集 | Mean 差异 | 区分能力等级 |
|--------|----------|-------------|
| Amazon | 0.38 | ⭐⭐⭐ 较强 |
| Elliptic | 0.20 | ⭐⭐ 中等 |
| t_finance | 0.13 | ⭐⭐ 中等 |
| Photo/Tolokers | 0.07 | ⭐ 弱 |
| Reddit | 0.0013 | ❌ 极弱 |

### 6.5 结论可信度评估

| 结论 | 可信度 | 证据 |
|------|--------|------|
| 方向一致（Anomaly > Normal） | ⭐⭐⭐⭐⭐ 高 | 6/6 数据集一致 |
| 统计显著 | ⭐⭐⭐⭐⭐ 高 | 6/6 p < 1e-3 |
| 实际区分能力 | ⭐⭐ 中 | Mean差异小，但方向一致 |
| 方差一致性 | ⭐⭐⭐ 中高 | 5/6 数据集一致 |

### 6.6 学术价值

**这个发现的价值**：

1. **挑战传统假设**: 证明"异常节点偏离邻居"假设在社区型异常场景下不成立
2. **提供新视角**: 异常检测应考虑社区级别的传播模式
3. **支持 VecGAD 设计**: Delta 相似性可能在社区级别有意义

**局限性**：

1. **区分能力有限**: Mean差异较小，实际分类能力弱
2. **半监督应用受限**: 无法直接用作异常分数（方向与预期相反）
3. **需要更多验证**: 理论解释仍需进一步研究

---

## 七、后续建议

### 7.1 进一步研究方向

1. **验证社区结构**: 分析异常节点的社区连接模式
2. **扩展到更多数据集**: 测试不同类型的异常（结构异常 vs 特征异常）
3. **理论分析**: 研究为什么社区型异常会表现这种特性

### 7.2 与 Transformer 结合的潜力

当前发现表明：
- Delta 序列模式与邻居的关系可能对注意力机制有参考价值
- 但区分能力有限，不适合直接作为注意力权重

---

**报告完成于**: 2026-04-16  
**位置**: `investigations/2026-04-16-isp-hop2token/experiments/outputs/`