# Prototype Token 与 Community-aware Token 候选定义表

## 一、目标

本文件的目标，是把前一版 `anomaly_aware_multi_token_draft.md` 中较抽象的 token family 设计，进一步落到可操作的定义层面。重点聚焦两类新 token：

1. **Neighborhood Prototype Token**
2. **Community-aware Token**

这两类 token 对应的是我们最近最关键的两条现象：

- NDC：异常节点与邻域 Delta 更一致
- ANR：异常节点往往处在异常浓度更高的局部社区中

因此，它们不只是“新增 token”，而是把已观察到的现象显式纳入表示层。

---

## 二、Neighborhood Prototype Token：候选定义

### 2.1 设计目标

Prototype token 的核心任务，是回答：

> 目标节点所处邻域的“平均传播模式”是什么？

这样做的意义是把 NDC 对应的关系显式化。NDC 本质上衡量的是：

- node delta
n- 与 neighborhood mean delta 的相关性

如果我们把 neighborhood mean delta 直接变成 token，那么模型就不需要只依赖事后统计分析，而可以在表示层就接触到“邻域原型”。

---

### 2.2 候选定义 A：Hop Prototype Token

**定义**：对节点邻域的 hop 表示按 hop 逐层求均值，得到一个 prototype sequence。

形式上：
- 对每个 hop k
- 取邻居在 hop k 的表示均值
- 拼成 prototype hop token sequence

**语义**：
- 邻域在不同传播深度下的平均状态

**优点**：
- 稳定、直观
- 容易和现有 hop token 对齐
- 方差较低

**缺点**：
- 只能表达“平均状态”
- 对动态变化的敏感度不如 delta-based prototype

**适合场景**：
- 作为最基础的 prototype 版本
- 用来先验证“邻域平均传播状态”是否有额外区分力

---

### 2.3 候选定义 B：Delta Prototype Token

**定义**：对节点邻域的 delta 表示逐层求均值，得到一个 prototype delta sequence。

形式上：
- 对每个 delta_k
- 取邻居在该 delta 上的均值
- 拼成 prototype delta token sequence

**语义**：
- 邻域平均传播变化模式

**优点**：
- 与 NDC 的定义最直接对齐
- 更能表达“邻域如何演化”而不只是“邻域是什么状态”
- 和当前 Delta token 路线高度兼容

**缺点**：
- 可能对噪声更敏感
- 若邻域本身异质性较强，均值可能被冲淡

**适合场景**：
- 作为 prototype token 的首选候选
- 用于直接承接 NDC 现象

---

### 2.4 候选定义 C：Weighted Prototype Token

**定义**：不是简单均值，而是按结构或表示相似度做加权均值。

可选权重来源：
- 邻接权重 / 度归一化权重
- 表示空间相似度
- attention-style weight
- 局部一致性分数

**语义**：
- 邻域中的“主导传播模式”

**优点**：
- 比简单平均更灵活
- 能减弱异质邻居的干扰

**缺点**：
- 设计更复杂
- 更容易引入超参数敏感性
- 暂时不适合第一步验证

**适合场景**：
- 当 simple mean prototype 证明有效后，再进一步细化

---

## 三、Prototype Token 的推荐推进顺序

| 候选 | 推荐级别 | 原因 |
|------|---------|------|
| Delta Prototype Token | ⭐⭐⭐⭐⭐ | 最贴近 NDC，最值得先做 |
| Hop Prototype Token | ⭐⭐⭐⭐ | 稳定、直观，适合作为对照 |
| Weighted Prototype Token | ⭐⭐ | 适合后续扩展，不适合第一步 |

**当前建议**：
先从 **Delta Prototype Token** 开始，把它作为 NDC 对应的显式 token；同时保留 Hop Prototype 作为一个更平稳的 baseline。

---

## 四、Community-aware Token：候选定义

### 4.1 设计目标

Community-aware token 的目标，是回答：

> 这个节点是否处在一个局部一致、可能异常集中的社区中？

它要承接的是 ANR 现象，但不能直接使用标签，因此必须寻找无监督 proxy。

---

### 4.2 候选定义 A：Local Consistency Token

**定义**：基于节点与邻域表示的一致性统计量构造 token。

可选统计量：
- node delta 与 neighbor delta mean 的相关性
- node delta 与邻域 prototype 的距离
- 邻域内部 delta 的方差

**语义**：
- 节点是否处在一个局部一致的传播环境中

**优点**：
- 与 NDC 直接相关
- 不依赖标签
- 容易实现

**缺点**：
- 更像 consistency token，而不是完整 community token
- 只能捕捉一致性，未必能区分“正常同质社区”和“异常社区”

**适合场景**：
- 作为 community-aware token 的第一步 proxy

---

### 4.3 候选定义 B：Local Density / Cohesion Token

**定义**：基于局部连通密度、聚类系数、邻域内部连接比例等结构统计构造 token。

**语义**：
- 节点所在局部子图是否形成紧密小团体

**优点**：
- 更接近“社区”这个概念
- 易于计算

**缺点**：
- 只看结构，不看传播模式
- 容易把普通高同质正常社区也编码成高值

**适合场景**：
- 作为补充结构 proxy，而不宜单独使用

---

### 4.4 候选定义 C：Representation-cluster Token

**定义**：在表示空间中，衡量节点与其局部邻域是否形成高密度小簇，再把该局部簇统计编码成 token。

可选方式：
- 邻域表示的局部聚类中心
- 邻域中 top-k 相似节点的 prototype
- 局部簇内方差 / 紧致度

**语义**：
- 节点在表示空间中是否处于一个高一致簇中

**优点**：
- 比纯结构指标更贴近异常检测表示语义
- 更可能承接“异常社区内部一致”这个现象

**缺点**：
- 设计复杂度更高
- 需要更谨慎验证

**适合场景**：
- 第二阶段探索对象

---

## 五、Community-aware Token 的推荐推进顺序

| 候选 | 推荐级别 | 原因 |
|------|---------|------|
| Local Consistency Token | ⭐⭐⭐⭐⭐ | 最容易落地，最贴近 NDC/ANR 中间层解释 |
| Local Density / Cohesion Token | ⭐⭐⭐ | 可作为结构补充，但单独不够 |
| Representation-cluster Token | ⭐⭐⭐⭐ | 潜力大，但应放在第二步 |

**当前建议**：
先把 **Local Consistency Token** 当作 community-aware token 的初始 proxy。虽然它还不是完整社区表示，但足够承接“局部一致传播环境”这一关键语义。

---

## 六、如何把两类 token 接到现有现象上

| 现象 | 对应 token | 解释 |
|------|-----------|------|
| NDC 高 | Delta Prototype Token | 异常节点与邻域平均 Delta 模式更一致 |
| NDC 高 + 邻域内部低方差 | Local Consistency Token | 节点处于局部一致传播环境 |
| ANR 高 | Community-aware Token | 异常往往出现在局部一致、异常浓度更高的社区 |

这张表说明，新 token 的设计不是凭空添加，而是直接对应已有观测现象。

---

## 七、建议的最小验证组合

### 组合 1：Hop + Delta + Delta Prototype
**目的**：验证 prototype 是否提供超出原始 delta 的额外信息

### 组合 2：Delta + Local Consistency
**目的**：验证 consistency proxy 是否能补足 node-wise delta 的不足

### 组合 3：Hop + Delta + Delta Prototype + Local Consistency
**目的**：作为当前最小 anomaly-aware multi-token 草案

---

## 八、当前结论

如果从可操作性和与现有现象的贴合度来看，最值得优先推进的定义是：

1. **Delta Prototype Token**
2. **Local Consistency Token**

它们分别对应：
- NDC 的显式 prototype 化
- ANR 背后“局部一致异常环境”的无监督 proxy 化

这两步如果能走通，我们就能把当前的统计发现，真正转成 VoxG 的表示设计语言。
