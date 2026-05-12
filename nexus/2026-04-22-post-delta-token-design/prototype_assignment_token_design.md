# Prototype Assignment / Contrast Token 设计草案

## 一、设计目标

Prototype 路线在 post-delta 时代仍值得保留，但前提是它必须从“均值 prototype”升级到“模式归属 / 模式对比”层。其核心问题不再是：

- 节点和邻域均值有多接近？

而是：

> 节点属于哪个局部模式？它偏离哪个参考原型？它是否处在模式边界？

因此，Prototype Assignment / Contrast Token 的价值，在于把异常检测从单纯状态比较，推进到模式归属与参考偏离的层面。

---

## 二、为什么均值 prototype 不够

上一轮轻量验证已经说明：
- mean-delta prototype 有统计信号
- 但它太容易与 delta 主干重合
- 更像局部平滑版辅助特征，而不是新坐标系

所以真正值得保留的，不是“prototype 均值本身”，而是：
- assignment
- contrast
- uncertainty / margin

也就是：

> **让 prototype 不再只是一个向量，而是一个模式参考系统。**

---

## 三、当前最值得优先的两个候选

### P-A1: Local Prototype Assignment Token

**定义**：
在节点的局部环境或局部表示空间中构造若干 prototype，token 表达该节点对这些 prototype 的 assignment 权重。

**候选组成**：
- 对各 prototype 的 soft assignment
- 最近 prototype id / weight
- assignment entropy

**语义**：
- 节点属于哪个局部模式
- 节点是否落在模糊、边缘或少数模式中

**优点**：
- 比 mean prototype 更接近模式归属
- 可自然扩展到 mixture 形式

**风险**：
- prototype 数量敏感
- assignment 质量依赖局部聚类

---

### P-A2: Prototype Contrast / Margin Token

**定义**：
不只看最近 prototype，而看节点相对于多个 prototype 的对比关系，例如最近与次近 prototype 的 margin、对 normal prototype 的偏离等。

**候选组成**：
- closest vs second-closest margin
- distance to normal prototype
- local prototype contrast vector

**语义**：
- 节点是否处于模式边界
- 节点是否明显偏离正常参考

**优点**：
- 更贴近异常检测中的“偏离参考”语义
- 不会退化成单一均值距离

**风险**：
- 需要先定义比较稳定的 prototype 系统
- 容易和 relation token 家族有交叉

---

## 四、Prototype Token 如何进入下游模型

我认为 prototype token 更适合作为：

### 方案 A：独立 token family
- node token
- prototype assignment / contrast token
- Transformer 学 node-to-prototype 交互

### 方案 B：作为 relation token 的参考支撑层
- prototype 不单独做主 token
- 而是作为 relation token 的 reference bank

### 当前偏好
我更倾向于：

> **prototype 路线在现阶段先不要做第一主 token，而更适合作为 Patch / Relation 之外的“模式参考层”。**

也就是它重要，但未必应该作为第一主线单独压上去。

---

## 五、为什么 Prototype 路线仍有保留价值

因为它回答的是 patch 与 relation 都没有完全回答的问题：

- patch 更强调结构单元
- relation 更强调上下文关系
- prototype 则强调模式归属与偏离参考

这三者不是完全同类问题。

Prototype 路线真正的价值在于：

> **把异常检测从“你和邻域像不像”推进到“你属于哪个模式、你偏离哪个参考”。**

如果设计得当，它依然可能成为 multi-token family 中的重要组成部分。

---

## 六、最小验证路线（Prototype 专项）

### 验证 1：assignment 可分性
检查 anomaly 节点是否在：
- assignment entropy
- assignment margin
- normal-prototype distance
上表现出稳定差异。

### 验证 2：assignment / contrast vs mean prototype
比较：
- mean prototype baseline
- assignment token
- contrast token

### 验证 3：prototype 是否解释 relation / patch 之外的残差信号
若 patch 与 relation 已经很强，prototype 是否还有独立增量？

---

## 七、当前判断

Prototype 路线不应被直接放弃，但必须完成一次升级：

> **从均值统计，升级为模式归属与参考对比。**

在当前三类 token 中，我仍然认为它的优先级低于 Patch 与 Relation，但它很可能是最终 unified framework 中不可缺少的第三层。
