# Post-Delta Token Design：三类新 token 来源的初步分析

## 一、问题转向

本探究的出发点很明确：

在上一轮 Tokenphormer × VoxG × NDC/ANR 探究中，我们逐步发现，一阶差分并没有形成足够强的新坐标系。它更多像传播主干的一种重写，而不是异常检测所需的全新信息来源。因此，如果还要继续从 Tokenphormer 那里学习，重点就不该放在“怎么设计另一个差分 token”，而应放在它更本质的方法论上：

> **为任务寻找真正不同的信息来源，并把这些来源组织成 token family。**

对 VoxG 来说，这意味着新的问题不是“还要不要 delta”，而是：

> **异常检测里，什么样的 token 才代表真正新增的异常语义？**

基于前序探究，我认为当前最值得系统展开的三个来源是：

1. Community / Patch Token
2. Prototype Assignment / Contrast Token
3. Relation Token

---

## 二、Community / Patch Token

### 2.1 为什么值得单独成类

如果 ANR 的结论是可信的，那么它已经告诉我们：异常并不总是单点偏离，而经常表现为局部社区中的一致异常。换句话说，异常的基本单元可能不是节点，而是一个小型局部块（patch）。

如果仍然只把“单节点 + 它的 hop 表示”当作 token，模型会天然偏向 node-wise 学习，而难以显式看见：

- 哪些节点共同构成一个异常簇
- 一个局部子图内部是否高度一致
- 一个 patch 是否与周围区域存在明显不协调

所以 Community / Patch Token 的价值在于：

> **把异常检测的建模粒度，从单节点提升到局部结构单元。**

### 2.2 候选定义方向

可以考虑几种 patch 构造方式：

- 基于局部邻域的 patch：以目标节点为中心，提取 k-hop 局部子图
- 基于表示相似性的 patch：取 top-k 表示最接近的邻居组成 patch
- 基于社区发现的 patch：取局部社区或小簇作为 token 单元
- 基于异常一致性的 patch：优先抽取内部高度一致、外部偏离的局部块

### 2.3 它与 NDC / ANR 的关系

这类 token 与 ANR 的联系最直接：

- ANR 告诉我们异常邻居聚集
- patch token 则把这种“聚集结构”直接对象化

与 NDC 的关系则在于：
- 若一个 patch 内节点的传播模式高度一致
- 那么 patch token 可能比单节点 delta 更自然地表达这种一致性

### 2.4 最大潜力

Community / Patch Token 是当前三类中最像“真正新模态”的一类，因为它不再是对单节点传播轨迹做变换，而是改变了基本建模单位。

---

## 三、Prototype Assignment / Contrast Token

### 3.1 为什么 mean prototype 不够

上一轮探究已经表明，mean-delta prototype 有一些信息，但它太像 Delta 的邻域平滑版，容易与 Delta 主干重合。因此更值得探索的不是“求一个均值 prototype”，而是：

> **节点如何与若干模式原型发生关系。**

异常检测很多时候不是问“你的状态是什么”，而是问：

- 你更像哪一类模式？
- 你偏离哪个参考原型？
- 你是否落在一个局部小众模式簇里？

这已经不再是均值统计，而是“模式归属与偏离”的问题。

### 3.2 候选定义方向

可能的设计包括：

- local prototype assignment token：节点属于哪个局部模式原型
- normal-reference contrast token：节点与正常原型的偏离
- cluster contrast token：节点与最近 prototype、第二近 prototype 的差异关系
- prototype mixture token：节点对多个原型的 soft assignment

### 3.3 它与 NDC / ANR 的关系

与 NDC 的关系：
- NDC 提示节点与邻域平均传播模式相近
- 但更进一步的问题是：这种“相近”是否表示它属于某个局部异常模式簇？

与 ANR 的关系：
- 若异常在局部聚集，则这些节点可能共享某种 prototype assignment 结构
- prototype token 能把这种“模式归属”显式化

### 3.4 最大潜力

Prototype Assignment / Contrast Token 的最大价值，在于它把问题从“状态表示”提升到“模式归属表示”。这是比均值 prototype 更贴近异常检测本质的方向。

---

## 四、Relation Token

### 4.1 为什么关系应成为 token

异常检测里，很多异常不是绝对值异常，而是关系异常。例如：

- 节点本身不特殊，但与邻域的关系很反常
- 节点与社区中心关系不协调
- 节点处在一个内部一致但外部冲突的位置

这说明单节点 token 很可能不够，因为关键不在节点自身，而在“节点—上下文”的关系结构。

Relation Token 的核心思想是：

> **把关系本身编码为 token，而不是只把关系当作模型内部要自己学出来的隐含量。**

### 4.2 候选定义方向

可探索的 relation token 包括：

- node-to-neighborhood relation token
- node-to-community relation token
- node-to-prototype relation token
- intra-patch / extra-patch contrast token

这些 token 不是描述节点“是什么”，而是描述节点“相对于谁、以何种方式存在”。

### 4.3 它与 NDC / ANR 的关系

与 NDC 的关系：
- NDC 本质就是 node delta 与 neighborhood delta 的关系指标
- relation token 可以把这种关系直接 token 化，而不是停留在统计分析层面

与 ANR 的关系：
- ANR 提示节点所处局部环境与自身标签关系紧密
- relation token 可以进一步表达“节点与环境”的不协调或一致程度

### 4.4 最大潜力

Relation Token 的强项在于：

- 它最接近异常检测的“关系异常”语义
- 它不是对节点状态的另一种编码，而是显式引入二元或高阶关系

这可能是最容易真正形成“新增坐标系”的方向之一。

---

## 五、三类 token 的比较

| 类别 | 基本对象 | 最贴近的现象 | 最大价值 | 主要风险 |
|------|----------|--------------|----------|----------|
| Community / Patch | 局部子图 / 小簇 | ANR | 改变建模粒度，引入局部结构单元 | patch 构造方式不稳 |
| Prototype Assignment / Contrast | 模式原型与归属 | NDC + ANR | 从状态表示转向模式归属 | prototype 学得不好会退化 |
| Relation Token | 节点与上下文的关系 | NDC | 显式表达关系异常 | token 定义可能过多、过碎 |

从“与现有 hop/delta 路线的距离”看：

- Community / Patch Token 最远，最可能是新模态
- Relation Token 次之，最贴近异常检测语义
- Prototype Assignment / Contrast Token 介于二者之间，既继承 prototype 思路，又有机会摆脱 mean-prototype 的弱点

---

## 六、当前阶段判断

### 6.1 最值得优先推进的方向

如果只选一个优先级最高的方向，我目前会选：

> **Community / Patch Token**

因为它最直接承接 ANR，也最有可能真正跳出 hop/delta 那种“同一传播轨迹不同写法”的局限。

### 6.2 最值得作为第二优先级的方向

我会选：

> **Relation Token**

因为 NDC 本质上已经是一个关系型现象，而不是纯状态现象。Relation token 可能比 delta 更自然地承接这条发现。

### 6.3 Prototype 路线是否还值得保留

值得，但必须升级：
- 不再做均值 prototype
- 改做 assignment / contrast / mixture

否则它仍然容易退化成弱辅助特征。

---

## 七、下一步建议

下一步最自然的推进不是立刻做实验，而是分别为三类 token 进一步补一份：

1. **候选定义表**：每类 token 的 2~4 个具体候选形式
2. **与 NDC / ANR 的对应关系表**
3. **最小验证路线**：如何低成本判断它是否值得继续

这样才能把新探究从方向讨论推进到真正可验证的方法草图。

---

## 八、当前结论

如果把这次 post-delta 转向总结成一句话，我会写成：

> **放弃一阶差分之后，Tokenphormer 对 VoxG 仍然最有价值的启发，是按任务语义去组织多来源 token family，而不是围绕同一传播轨迹反复做重参数化。**

对 VoxG 而言，当前最值得深挖的三类新来源是：

- Community / Patch Token
- Prototype Assignment / Contrast Token
- Relation Token

它们分别对应：
- 局部结构单元
- 模式归属与偏离
- 节点—上下文关系

这三者都比一阶差分更接近真正的 anomaly-aware token design。
