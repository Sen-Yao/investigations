# 三类新 Token 的候选定义表

## 一、目标

本文件的目标，是把 `insights.md` 中较高层的 post-delta 方向，进一步压到可设计、可比较、可验证的 token 定义层面。我们聚焦三类新来源：

1. Community / Patch Token
2. Prototype Assignment / Contrast Token
3. Relation Token

每类 token 都给出若干候选形式，并明确其对应语义与验证价值。

---

## 二、Community / Patch Token

### C1. Local Ego-Patch Token

**定义**：以目标节点为中心，取其 1-hop 或 2-hop 局部子图，将该局部子图的结构与表示摘要为一个 patch token。

**语义**：
- 节点所在最直接局部环境
- patch 作为基本建模单元，而不是单节点

**与现象关系**：
- 若 ANR 高，异常节点周围往往聚集异常邻居
- ego-patch 可直接承接这种局部异常聚集

**优点**：
- 构造自然
- 不依赖额外聚类算法
- 适合作为 community/patch 的 baseline

**风险**：
- patch 边界粗糙
- 容易把正常稠密邻域与异常社区混淆

---

### C2. Representation Patch Token

**定义**：不按图拓扑直接截取邻域，而是按表示空间相似性，为每个节点选取 top-k 最相近节点组成 patch。

**语义**：
- 节点所属的局部模式簇
- 更强调表示一致性，而非单纯结构邻接

**与现象关系**：
- 若异常社区在表示空间中形成局部簇，则这类 token 更容易捕捉“异常模式团块”

**优点**：
- 更贴近模式一致性
- 有机会比拓扑 patch 更接近异常检测目标

**风险**：
- 依赖底层表示质量
- 可能与 prototype assignment 路线部分重合

---

### C3. Community-Core Patch Token

**定义**：在局部范围内先做轻量社区发现或局部聚类，再抽取与目标节点最相关的社区核心子块作为 token。

**语义**：
- 节点所属的局部结构核心
- 更接近“社区型异常”本体

**与现象关系**：
- 这是最直接承接 ANR 的候选形式

**优点**：
- 异常社区语义最强
- 最可能与 ANR 形成闭环

**风险**：
- 构造复杂
- 社区边界与算法选择敏感

---

## 三、Prototype Assignment / Contrast Token

### P1. Local Prototype Assignment Token

**定义**：为每个节点分配一个或多个局部 prototype，token 表达该节点对各 prototype 的 assignment 权重。

**语义**：
- 节点属于哪个局部模式原型
- 节点是否处于边缘、模糊或少数模式

**与现象关系**：
- 若异常形成局部小簇，则它们可能共享类似 assignment 结构

**优点**：
- 比 mean prototype 更接近“模式归属”
- 可自然扩展到 soft assignment

**风险**：
- prototype 学不好会失真
- assignment 数量需要控制

---

### P2. Normal-Reference Contrast Token

**定义**：引入一个或一组 normal reference prototype，token 表达目标节点与这些正常参考的偏离程度。

**语义**：
- 节点距离“正常模式”有多远
- 异常作为偏离正常参考来定义

**与现象关系**：
- 如果异常不是绝对孤立，而是局部一致偏离，那么 contrast to normal prototype 可能比 delta 更自然

**优点**：
- 直接贴近异常检测语义
- 容易解释

**风险**：
- normal reference 如何构造是关键难点
- 容易受噪声污染

---

### P3. Prototype Margin Token

**定义**：不是只看最近 prototype，而看最近 prototype 与第二近 prototype 之间的 margin 或归属不确定性。

**语义**：
- 节点是否处于模式边界
- 节点是否属于模糊/不稳定区域

**与现象关系**：
- 可帮助识别“局部上看一致，但全局模式归属不稳”的异常点

**优点**：
- 不是简单均值
- 有机会表达模式边界异常

**风险**：
- 需要多 prototype 框架
- 可能对聚类质量敏感

---

## 四、Relation Token

### R1. Node-to-Neighborhood Relation Token

**定义**：直接编码节点与邻域表示之间的关系，如相似度、偏差、对齐程度、局部 contrast。

**语义**：
- 节点与直接上下文是否协调

**与现象关系**：
- NDC 本质上就属于这类关系的统计表达

**优点**：
- 与现有 NDC 发现衔接最自然
- 构造相对低成本

**风险**：
- 若只用简单统计，可能再次退化成弱辅助特征

---

### R2. Node-to-Community Relation Token

**定义**：编码节点相对于所在局部社区中心或 patch 的关系，如对社区中心的偏差、对社区内部一致性的适配程度。

**语义**：
- 节点是否“属于”该社区
- 节点在社区中是否是边缘成员或异常成员

**与现象关系**：
- 可同时承接 ANR 与社区型异常解释

**优点**：
- 把 node-wise 与 community-wise 视角桥接起来

**风险**：
- 需要先有稳定的 community / patch 定义

---

### R3. Node-to-Prototype Relation Token

**定义**：直接编码节点与若干 prototype 之间的关系，而不把 prototype assignment 本身当 token。

**语义**：
- 节点相对各类模式参考的位置关系

**与现象关系**：
- 是 relation 与 prototype 两条线的交叉形式

**优点**：
- 表达灵活
- 更适合做多关系输入

**风险**：
- 容易与 prototype token 家族重叠

---

## 五、三类 token 的当前优先级判断

| 类别 | 候选 | 当前优先级 | 原因 |
|------|------|-----------|------|
| Community / Patch | C1 | 高 | 最自然、最低门槛 baseline |
| Community / Patch | C2 | 高 | 更贴近模式一致性 |
| Community / Patch | C3 | 中高 | 语义最强但复杂 |
| Prototype | P1 | 高 | 最有希望摆脱 mean prototype 弱点 |
| Prototype | P2 | 中高 | 贴近异常检测但 normal reference 难构造 |
| Prototype | P3 | 中 | 有趣但依赖 prototype 体系成熟 |
| Relation | R1 | 高 | 最直接承接 NDC |
| Relation | R2 | 高 | 最可能接住社区型异常 |
| Relation | R3 | 中高 | 灵活但与 prototype 家族交叠 |

---

## 六、当前建议

若进入下一步方法收敛，我建议先聚焦以下 4 个候选：

1. **C1 Local Ego-Patch Token**
2. **C2 Representation Patch Token**
3. **P1 Local Prototype Assignment Token**
4. **R1 / R2 Relation Token**

这四个候选覆盖：
- 局部结构单元
- 局部模式簇
- 模式归属
- 节点与上下文关系

已经足够形成一个有辨识度的 post-delta token design 草案。
