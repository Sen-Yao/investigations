# Relation Token 设计草案

## 一、设计目标

Relation Token 的目标，是把“节点与上下文之间的关系”本身当作输入单元，而不是只把节点状态输入模型，再让模型自己隐式推断这些关系。

这条线的核心问题是：

> 对图异常检测来说，异常往往不是绝对状态异常，而是关系异常。那么为什么不把关系本身 token 化？

Relation Token 最直接承接 NDC，也能够桥接到社区型异常分析。

---

## 二、Relation Token 的核心语义

### 1. 协调 / 不协调语义
节点与其邻域、patch、community 是否一致。

### 2. 归属 / 边缘语义
节点是否真正属于当前局部上下文，还是处在边缘或偏离位置。

### 3. 对比 / 张力语义
节点与局部环境之间是否存在结构性张力，而这种张力是否与异常有关。

---

## 三、当前最值得优先的两个候选

### R-C1: Node-to-Neighborhood Relation Token

**定义**：
直接编码节点与其邻域 summary 的关系。

**候选组成**：
- cosine / l2 / correlation
- target 与邻域均值的偏差
- target 与邻域一致性的适配程度

**优点**：
- 与 NDC 对接最直接
- 构造成本低
- 是最自然的 relation baseline

**风险**：
- 若只停留在简单统计，容易再次变成弱辅助特征

---

### R-C2: Node-to-Community / Patch Relation Token

**定义**：
编码目标节点与所在 patch 或局部 community 的关系。

**候选组成**：
- target-to-patch center distance
- target-to-patch consistency gap
- target 在 patch 内 / patch 外的对比位置

**优点**：
- 能桥接 NDC 与 ANR
- 既保留 relation 语义，又接入 patch 语义

**风险**：
- 需要先有 patch / community 定义
- 与 patch token 家族耦合较强

---

## 四、Relation Token 如何进入下游模型

我认为 Relation Token 不一定需要作为“实体 token”，它可以作为：

### 方案 A：独立 token family
- node token
- relation token
- model 学 node-relation 交互

### 方案 B：结构化 side token
- relation token 作为附属上下文 token
- 强调 node 相对上下文的位置

### 当前偏好
如果目标是保持 Tokenphormer 风格的 multi-token family，我更倾向 **方案 A**。

---

## 五、为什么 Relation Token 比 delta 更自然

Delta 的局限在于：
- 它试图通过“变化量”间接表达异常
- 但很多异常本质上不是变化异常，而是关系异常

Relation Token 则直接把问题改写成：

- 节点与上下文是否一致？
- 节点是否属于该局部区域？
- 节点是否与周围结构形成张力？

这比继续围绕传播差分做文章，更接近异常检测本体。

---

## 六、最小验证路线（Relation 专项）

### 验证 1：relation feature 统计可分性
比较不同 relation feature 在 normal / anomaly 上的分布差异。

### 验证 2：relation feature 对 residual 的解释力
先用 node state baseline，再看 relation token 是否能解释剩余错误。

### 验证 3：relation vs node-only probe
比较：
- node-only
- relation-only
- node + relation

---

## 七、当前判断

Relation Token 最大的价值在于：

> **它让 NDC 从“分析现象”变成“表示设计原则”。**

如果我们不再执着于 delta，那么 Relation Token 很可能是保留 NDC 研究价值的最自然方向。
