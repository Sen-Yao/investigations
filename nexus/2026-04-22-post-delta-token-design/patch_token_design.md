# Patch Token 设计草案

## 一、设计目标

Patch Token 的目标不是再对单节点传播轨迹做变换，而是把局部结构单元本身作为输入对象。它要解决的问题是：

> 如果异常往往表现为局部一致、局部聚集、局部社区化，那么为什么我们还要坚持只用 node-wise token 来建模？

因此，Patch Token 的核心价值在于：
- 把建模粒度从节点提升到局部块
- 显式承接 ANR 所揭示的社区型异常
- 为 Transformer 提供真正不同的信息单元，而不是 hop/delta 的重写

---

## 二、Patch Token 的三层语义

### 1. 局部环境语义
Patch 表示目标节点所在最直接局部环境，而不仅是它自身状态。

### 2. 局部一致性语义
Patch 可表达该局部区域内部是否形成一致模式，这比单节点更接近异常社区现象。

### 3. 内外对比语义
Patch 还能自然表达“内部一致 / 外部偏离”，这正是很多异常子结构的核心特征。

---

## 三、当前最值得优先的两个候选

### P-C1: Local Ego-Patch Token

**定义**：
以目标节点为中心，提取 1-hop 或 2-hop 局部子图，并对该局部子图形成一个 patch 表示。

**最小实现形式**：
- patch 内节点特征均值
- patch 内边密度
- patch 内表示一致性
- target node 与 patch summary 的关系

**优点**：
- 构造简单
- 是最自然的 baseline
- 与 ANR 对接直接

**缺点**：
- patch 边界粗糙
- 拓扑邻域不一定等于异常社区

---

### P-C2: Representation Patch Token

**定义**：
不按纯拓扑邻接切 patch，而是基于表示相似性或局部聚类，把目标节点最相近的一组节点当作 patch。

**最小实现形式**：
- top-k 相似节点集合
- patch 表示均值 / 方差
- patch 内 pairwise similarity
- target-to-patch center relation

**优点**：
- 更贴近模式一致性
- 更可能接近真实异常团块

**缺点**：
- 依赖底层表示质量
- 与 prototype 路线有交叉

---

## 四、Patch Token 如何进入下游模型

我认为 Patch Token 不应简单替代节点 token，而应作为第二类 token family 并行输入。

### 方案 A：Node Token + Patch Token
- node token 负责节点自身状态
- patch token 负责局部结构单元
- Transformer 学 node-to-patch 的交互

### 方案 B：Node Token + Patch Summary Relation
- 不直接把完整 patch 当 token
- 而把 target 与 patch 的关系作为 token

### 当前偏好
我更倾向先从 **方案 A** 开始思考，因为它更像真正的 multi-token family。

---

## 五、为什么 Patch Token 比 hop/delta 更像新增坐标系

因为 hop/delta 的问题在于：
- 基本对象始终还是“同一个节点”
- 只是从不同传播层次重写它

而 patch token 的基本对象已经变成：
- 一个局部结构单元

这意味着它的新增量不是代数变换带来的，而是**建模粒度变化**带来的。

这正是我认为它最有希望的地方。

---

## 六、最小验证路线（Patch 专项）

### 验证 1：Patch 一致性分析
比较 normal patch 与 anomaly patch：
- 内部一致性
- 内部密度
- 对外部邻域的偏离程度

### 验证 2：Patch 摘要特征轻量 probe
比较：
- node-only
- patch-only
- node + patch

### 验证 3：Patch 是否解释 ANR
检查 patch 级指标是否比 node 级指标更自然地区分异常聚集现象。

---

## 七、当前判断

Patch Token 最大的吸引力在于：

> **它不是对节点表示做另一种展开，而是把异常可能真正存在的局部结构单元直接送进模型。**

如果 post-delta 时代要选一个最值得押注的方向，我目前仍然认为 Patch Token 是第一优先级。
