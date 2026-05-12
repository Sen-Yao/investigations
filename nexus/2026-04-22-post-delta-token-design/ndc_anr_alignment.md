# 三类新 Token 与 NDC / ANR 的对应关系

## 一、目标

本文件的目标，是明确三类新 token 来源如何分别承接我们已经观察到的两个关键现象：

- **NDC**：节点与邻域传播模式关系异常
- **ANR**：异常节点周围异常比例显著更高，呈现社区型异常倾向

这一步的意义在于：

> 不让新 token 设计停留在抽象灵感层，而是让每一类 token 都对应已有证据链中的具体现象。

---

## 二、Community / Patch Token 对应关系

### 与 ANR 的关系：强对应

Community / Patch Token 最直接承接 ANR。

原因是：
- ANR 说明异常不是均匀散布，而是局部聚集
- patch / community token 直接把这种局部聚集结构对象化

如果 ANR 是 post-delta 时代最重要的现象锚点，那么 patch token 就是最自然的表示层对应物。

### 与 NDC 的关系：中等对应

NDC 说明节点与邻域传播模式关系特殊，而 patch token 可以进一步把“邻域一致传播模式”上升为 patch 级别模式。

因此，patch token 不是直接对应 NDC 指标本身，但可以把 NDC 所揭示的“局部一致性”提升为结构单元语义。

---

## 三、Prototype Assignment / Contrast Token 对应关系

### 与 NDC 的关系：中强对应

NDC 本质上说明节点与邻域传播模式相似。Prototype 路线若升级为 assignment / contrast，则可把这个现象解释为：

- 节点属于某个局部模式 prototype
- 节点偏离 normal prototype 或全局 prototype
- 节点在多个 prototype 之间的归属不稳定

也就是说，NDC 提供的是“相似性现象”，而 prototype assignment 提供的是“模式归属解释”。

### 与 ANR 的关系：中等对应

如果异常社区内部节点共享某种原型，那么 prototype assignment 也可间接承接 ANR。但它的对应不如 patch token 那么直接。

因此 prototype 路线更像是：
- 用模式归属解释局部聚集现象
- 但不是局部聚集的最原始表达层

---

## 四、Relation Token 对应关系

### 与 NDC 的关系：最强对应

Relation Token 与 NDC 的联系最直接。

因为 NDC 本质上已经是“node-to-neighborhood relation”的统计量。Relation token 只是把这个现象从分析指标，升级成输入表示单元。

因此：
- 如果想保留 NDC 的研究价值
- 又不想继续走 delta token 这条线
- 那么最自然的转向就是 relation token

### 与 ANR 的关系：中强对应

ANR 描述的是节点与局部环境之间的标签关系结构。虽然 relation token 不能像 patch token 那样直接对象化社区，但它能表达：

- 节点与社区是否协调
- 节点与局部中心是否一致
- 节点是否处在“内部一致 / 外部冲突”的位置

所以 relation token 也是连接 NDC 与 ANR 的桥梁型设计。

---

## 五、对应强度总结表

| Token 类别 | 对 NDC 的承接强度 | 对 ANR 的承接强度 | 核心理由 |
|-----------|------------------|------------------|----------|
| Community / Patch | 中 | 强 | ANR 本质上就是局部聚集现象 |
| Prototype Assignment / Contrast | 中强 | 中 | 用模式归属解释局部一致与偏离 |
| Relation Token | 强 | 中强 | NDC 本质就是关系现象 |

---

## 六、当前判断

如果按“哪个 token 最自然承接哪个现象”来分工，我当前会建议：

- **Patch Token 主接 ANR**
- **Relation Token 主接 NDC**
- **Prototype Assignment / Contrast Token 作为模式归属解释层**

这种分工比让一种 token 同时承担所有语义更自然，也更符合 Tokenphormer 的 multi-token family 精神。
