# Patch + Relation First 验证计划

## 一、为什么第一轮先不做完整 unified framework

虽然 post-delta multi-token framework 已经初步成型，但当前还不适合直接进入完整统一建模。原因有三：

1. **证据链还不够硬**
   我们目前有方向性判断，但还缺少足够多的低成本实证来证明 Patch / Relation 真的形成了新增量。

2. **完整框架耦合度太高**
   如果一开始就把 node + patch + relation + prototype 全部并行输入，实验结果很难解释。我们无法知道收益究竟来自哪条线，也无法判断哪类 token 真正有效。

3. **当前最强现象锚点主要落在 Patch 与 Relation 上**
   ANR 最直接指向 Patch，NDC 最直接指向 Relation。因此从现象到设计的对应关系看，第一轮优先验证这两条线是最自然的。

因此，第一轮的策略应当是：

> **先做 Patch + Relation first，等这两条线站稳，再决定 Prototype 在 unified framework 中的角色。**

---

## 二、为什么是 Patch + Relation

### Patch
Patch 是 ANR 的最直接表示层翻译。它的意义在于：
- 把局部异常聚集从统计现象变成建模对象
- 改变建模粒度，从 node-wise 变为局部结构单元

### Relation
Relation 是 NDC 的最直接表示层翻译。它的意义在于：
- 把“节点与上下文关系异常”从分析指标变成 token family
- 保留 NDC 的研究价值，而不继续执着于 delta token

### 二者组合的逻辑
Patch 与 Relation 的组合刚好覆盖：
- **局部结构单元**
- **节点—上下文关系**

这已经足以形成一版真正不同于 hop/delta 的新设计基础。

---

## 三、第一轮验证目标

第一轮验证不追求完整模型性能，而要回答三个更基础的问题：

1. **Patch token 是否真的捕捉到了 node-wise 表示之外的结构信号？**
2. **Relation token 是否比 delta 更自然地承接 NDC？**
3. **Patch + Relation 是否至少在轻量验证层面表现出比 hop/delta 更像“新增坐标系”？**

---

## 四、建议的最低成本实验

### 实验 A：Patch 统计分析

#### 目标
检查 anomaly patch 与 normal patch 是否在以下指标上显著不同：
- patch 内部一致性
- patch 内部密度
- patch 对外部邻域的偏离程度

#### 意义
若这些量显著，则说明 Patch 确实对应局部结构差异，而非节点状态重写。

---

### 实验 B：Patch 轻量 Probe

#### 输入对比
- node-only
- patch-only
- node + patch

#### 目标
观察 patch 摘要是否提供独立于 node state 的补充能力。

#### 当前优先数据集
- Photo
- Amazon

---

### 实验 C：Relation 统计分析

#### 目标
构造多种 relation feature，如：
- node-to-neighborhood
- node-to-patch
- node-to-community（若 patch 定义可用）

检查这些特征在 normal / anomaly 上是否显著不同。

#### 意义
若成立，则说明 NDC 可从统计现象过渡为关系型表示设计。

---

### 实验 D：Relation 残差解释力

#### 目标
先用 node state baseline 预测，再检查 relation feature 是否能解释 residual。

#### 意义
这一步比单纯看 relation-only AUC 更重要，因为它直接回答 relation 是否真有新增量。

---

### 实验 E：Patch + Relation 组合 Probe

#### 输入对比
- node-only
- node + patch
- node + relation
- node + patch + relation

#### 目标
判断 Patch 与 Relation 是否同时提供互补信息。

#### 当前要求
先用轻量线性 / logistic probe，不上完整 Transformer。

---

## 五、判断标准

第一轮验证不是看绝对 SOTA，而看三种信号是否出现：

### 信号 1：Patch 级指标稳定显著
若 anomaly patch 与 normal patch 在结构或一致性上稳定分离，说明 Patch token 有存在价值。

### 信号 2：Relation 能解释 residual
若 relation feature 能解释 node baseline 未覆盖的误差，说明它不只是 node state 的重写。

### 信号 3：Patch + Relation 比单独任一路线更稳
若两者组合比各自单独更有价值，则说明它们可能形成真正的 multi-token 基础。

---

## 六、为什么暂时不先做 Prototype First

不是因为 Prototype 不重要，而是因为：

1. 它目前更像“解释层 / 参考层”
2. 它依赖 patch / relation 提供更稳的局部上下文
3. 若 Patch / Relation 都还没站稳，就很难判断 prototype 的参考系统该怎么定义

所以更合理的顺序是：

> **先 Patch / Relation，再 Prototype。**

---

## 七、当前结论

我会把第一轮实验化路线总结成一句话：

> **post-delta 时代的第一步，不是立刻训练一个更复杂的 multi-token 模型，而是先证明 Patch 与 Relation 这两个现象锚点，真的能提供超出 hop/delta 的新增坐标系。**

如果这一步站住了，Prototype 才有条件作为第三层进入 unified framework。
