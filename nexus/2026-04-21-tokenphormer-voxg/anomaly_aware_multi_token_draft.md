# Anomaly-aware Multi-token 设计草案

## 一、问题收敛

当前我们已经有两条关键现象证据：

1. **NDC 高**：异常节点与其邻域在 Delta 表达上更相似
2. **ANR 高**：异常节点的邻居中异常比例显著更高

这两条证据共同指向一个重要判断：

> 图异常并不总是表现为“单点偏离”，而可能表现为“局部社区内部的一致异常模式”。

因此，若仍然只使用单一节点级 token 表示，模型可能无法完整表达这种“社区内部一致 + 社区外部有差异”的异常结构。

---

## 二、设计目标

本草案的目标不是立刻给出最终模型，而是明确一套面向 VoxG 的 token family 设计框架，使其能够同时编码：

- 多 hop 传播状态
- hop 间变化过程
- 节点与邻域平均传播模式的关系
- 局部社区的一致异常线索

---

## 三、候选 token family

### 3.1 Hop Token

**作用**：编码多 hop 传播后的状态表示。

**对应语义**：
- 节点在不同传播深度下的表征快照
- 提供最基础的传播层级视角

**价值**：
- 保留多跳结构影响
- 为后续 Delta / prototype 提供基底

---

### 3.2 Delta Token

**作用**：编码相邻 hop 之间的变化量。

**对应语义**：
- 节点表示如何随传播逐步演化
- 捕捉传播过程中的动态变化

**与现象关系**：
- NDC 高说明异常节点在 Delta 层面与邻域更一致
- Delta token 是当前最直接承接 NDC 发现的表示对象

---

### 3.3 Neighborhood Prototype Token

**作用**：编码邻域平均传播模式，作为节点局部上下文的 prototype。

**候选定义**：
- 邻居 hop 表示的平均
- 邻居 delta 表示的平均
- 加权邻域 prototype（按相似度或结构权重）

**对应语义**：
- 节点“所处局部环境”的平均传播原型
- 节点是否贴近该原型，可能携带异常信息

**与现象关系**：
- NDC 本质上就是“节点 Delta 与邻域 Delta prototype 的一致性”
- 因此，prototype token 是把 NDC 现象显式 token 化的自然方向

---

### 3.4 Community-aware Token

**作用**：编码局部社区一致性或异常浓度的代理信号。

**候选来源**：
- 邻域一致性统计量
- 局部聚类 / 连通密度 proxy
- 基于表示空间的局部异常簇特征
- 无监督近似的社区一致性分数

**对应语义**：
- 节点是否处于一个高一致性的局部子群体中
- 该子群体是否更像异常社区而非普通同质社区

**与现象关系**：
- ANR 高说明异常经常形成社区
- 但 ANR 本身依赖标签，不能直接用于半监督
- 因此我们需要一个无监督 proxy，把 ANR 对应的“社区异常性”转译成 token

---

## 四、四类 token 的分工

| Token | 主要回答的问题 | 对应现象 |
|------|----------------|---------|
| Hop Token | 节点在不同 hop 上是什么状态？ | 多跳传播 |
| Delta Token | 节点如何随 hop 演化？ | NDC |
| Neighborhood Prototype Token | 节点所处邻域的平均传播模式是什么？ | NDC 的 prototype 解释 |
| Community-aware Token | 节点是否位于一致异常社区中？ | ANR 的无监督近似 |

这个分工很重要，因为它避免“多 token 只是更多输入”的问题。每一类 token 都对应一个明确的问题，而不是功能重叠。

---

## 五、最小方法框架（草图）

一个可能的最小版本可以写成：

1. 构造多 hop 表示 → 形成 Hop Token
2. 计算 hop 间差分 → 形成 Delta Token
3. 对邻域 hop / delta 做聚合 → 形成 Neighborhood Prototype Token
4. 基于局部一致性 proxy 构造 Community-aware Token
5. 将上述 token 作为一个 token set 送入共享编码器（Transformer 或轻量聚合器）
6. 输出节点异常分数

在这个框架里，最关键的不是 backbone，而是 token family 的任务分工和互补性。

---

## 六、为什么这条线适合 VoxG

这套设计比直接照搬 Tokenphormer 更适合 VoxG，原因在于：

### 6.1 VoxG 的优势在传播过程

Tokenphormer 偏结构片段，而 VoxG 当前最强的线索来自传播过程本身。Hop / Delta token 延续了这一优势。

### 6.2 NDC / ANR 给了我们异常语义锚点

我们不是凭空设计 token，而是有现象支撑：
- NDC → prototype / delta consistency
- ANR → community-aware consistency

### 6.3 这条线兼顾 node-wise 与 community-wise 异常

传统异常检测很多时候只看单节点，而这套设计同时考虑：
- 节点本身如何变化
- 节点与邻域的关系
- 节点所在社区的一致性

---

## 七、最小验证路线

当前不建议直接上大规模训练，而更适合先做轻量验证：

### 7.1 验证 A：Prototype 是否携带额外区分信息
- 比较 node delta 与 neighborhood prototype delta 的关系
- 检查 prototype-based 统计量是否能区分 normal / anomaly

### 7.2 验证 B：Community proxy 是否与异常显著相关
- 设计无监督社区一致性代理
- 检查该 proxy 与标签、NDC 的关联性

### 7.3 验证 C：token family 是否互补
- 比较 hop-only, delta-only, hop+delta, delta+prototype 等组合
- 先用轻量 probe，而不是完整训练

---

## 八、当前建议

如果下一步继续推进，我建议优先顺序如下：

1. **先定义 prototype token 的精确形式**
2. **再设计 community-aware proxy**
3. **最后决定是否需要完整训练验证**

因为当前最稳的一步，是把 NDC 从“统计现象”进一步变成“表示构件”；而 ANR 则更适合先转译成无监督 proxy，再考虑进入模型。

---

## 九、当前结论

> 最值得顺着 Tokenphormer 继续展开的，并不是 walk-token 路线，而是“多 token family 的异常检测化改写”。

对 VoxG 来说，下一步最有希望的方向是：

**Hop Token + Delta Token + Neighborhood Prototype Token + Community-aware Token**

这套结构既继承了 VoxG 对传播过程的优势，也自然吸收了 NDC / ANR 所揭示的社区型异常现象。
