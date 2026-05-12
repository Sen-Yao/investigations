# 冗余分析与特征互补性分析（第一版）

## 一、问题重新定义

在 Photo 与 Amazon 两个数据集的轻量验证之后，问题已经从：

> prototype / consistency 是否有信息？

转向：

> prototype / consistency 与 Delta 的信息重叠度到底有多高？为什么它们统计显著，却不总能带来稳定增益？

与此同时，还出现了一个更根本的问题：

> **原始 Hop Token 与 Delta Token 本身，对下游 Transformer 来说，是否已经高度信息重合？**

这个问题非常关键。因为如果 hop 和 delta 本来就高度相关，那么在其上继续叠加 prototype / consistency，很可能只是对同一传播信息的重复编码，而不是引入真正的新视角。

---

## 二、从结果倒推：当前到底发生了什么

### 2.1 Photo 的模式

Photo 上的结果是：
- prototype / consistency 统计显著
- prototype-only / consistency-only 单独效果一般
- 但与 Delta 组合后有稳定小幅增益

这说明在 Photo 上：
- 新特征不是主信息源
- 但它们确实提供了少量 Delta 未完全覆盖的信息

### 2.2 Amazon 的模式

Amazon 上的结果是：
- prototype / consistency 统计显著，且单独区分力比 Photo 更强
- 但加入 Delta 后没有带来增益

这说明在 Amazon 上：
- 新特征本身有异常相关信号
- 但这些信号很可能已被 Delta 主特征充分吸收
- 或者 probe 层无法从重叠特征中再挖出新增量

### 2.3 两组结果合起来的含义

这两组结果放在一起，给出一个很清晰的判断：

> **统计显著 ≠ 表示互补。**

一个特征可以非常显著地区分 normal / anomaly，但如果它与现有强特征高度相关，那么加进去也不一定提升下游效果。

因此，我们下一步必须显式分析：

1. 新特征与 Delta 的重叠度
2. Hop 与 Delta 之间的重叠度
3. 哪类 token 才真正带来“新增视角”

---

## 三、Prototype / Consistency 为什么容易和 Delta 重叠

### 3.1 Delta Prototype 本质上是 Delta 的邻域聚合

Delta Prototype Token 的定义，本质上是：

- 先有 node delta
- 再对邻域 delta 做均值或聚合
- 得到 neighborhood delta prototype

所以它天然不是一个“完全独立的新模态”，而更像是：

> **Delta 在邻域层面的重写。**

这意味着如果 Delta flatten 本身已经很强，prototype 很可能只是在补充局部平滑版本的同类信息。

### 3.2 Local Consistency 是 Delta 关系统计量

当前的 consistency proxy 基本建立在：

- node delta 与 prototype 的关系
- 邻域 delta 的方差
- 邻域 delta 的 pairwise 相似度

因此它也不是全新信息源，而是：

> **Delta 空间中的二阶统计。**

这种特征有解释价值，但它未必能给下游模型带来大量新增判别能力，尤其当 Delta 已经能隐式表达这些关系时。

---

## 四、最值得追问的问题：Hop 与 Delta 是否本来就高度重合

这是当前更深的一层问题。

### 4.1 从定义上看，Delta 是 Hop 的线性差分

若定义为：

- hop_0, hop_1, ..., hop_K
- delta_k = hop_k - hop_{k-1}

那么 Delta 不是独立于 Hop 的新信息，而是 Hop 的线性变换。

反过来，给定 hop_0 和所有 delta，理论上也可递推出所有 hop：

- hop_1 = hop_0 + delta_1
- hop_2 = hop_1 + delta_2
- ...

因此从**信息论层面**看：

> **Hop 序列与 Delta 序列在理想条件下近似等价，只是基底不同。**

这意味着：
- 如果下游模型容量足够强，单看表达能力，Hop 与 Delta 可能高度可互相恢复
- 二者差异主要不在“是否包含信息”，而在“信息以什么坐标系被呈现给模型”

### 4.2 为什么 Delta 仍可能优于 Hop

即便信息等价，表示方式仍然会影响下游模型学习难度。

Delta 相比 Hop 的优势可能在于：

1. **显式强调变化量**
   - Transformer 不必自己从相邻 hop 之间学差分

2. **弱化绝对状态，强化演化结构**
   - 更适合捕捉传播过程中异常发生的位置与强度

3. **更接近异常检测语义**
   - GAD 往往更关心“哪里不协调/变化异常”，而不是“最终状态长什么样”

所以：

> Hop 与 Delta 可能在原始信息上高度重合，但在学习偏置上并不等价。

### 4.3 对下游 Transformer 来说，Hop + Delta 可能会怎样

如果同时把 Hop Token 和 Delta Token 都送入 Transformer，会出现两种可能：

#### 情况 A：互补
- Hop 提供绝对传播状态
- Delta 提供局部变化线索
- Transformer 能利用二者差异进行融合

#### 情况 B：冗余
- Transformer 需要处理大量近线性依赖 token
- 注意力预算被消耗在重复信息上
- 训练更难，收益反而有限

当前从理论上看，我更倾向于：

> **Hop + Delta 不是天然互补，而是“可能轻度互补，但存在高冗余风险”。**

尤其当：
- K 不大
- hop 表示已经很平滑
- Delta 由 Hop 直接线性构造

这种冗余风险会更高。

---

## 五、对下游 Transformer 的真正挑战：不是 token 多，而是 token 是否“新增坐标系”

当前这条线给出的一个重要认识是：

> 多 token family 不等于把同一信息换几个形式重复输入。

对于下游 Transformer，真正有价值的 token family 应该满足至少一条：

1. **提供难以从现有 token 线性恢复的信息**
2. **提供不同粒度的归纳偏置**
3. **显式暴露模型不易自行学出的关系结构**

据此看：

- **Hop 与 Delta**：可能是“同一传播轨迹的两种坐标表达”
- **Delta Prototype**：是 Delta 的邻域聚合版本，新增信息有限但可能提供更平滑的局部 context
- **Local Consistency**：是 Delta 空间的关系统计，新增信息主要体现在“二阶关系摘要”

因此它们都不是完全独立的新模态，而是围绕同一传播主轴的不同投影。

这解释了为什么：
- 统计上能显著
- 但增益不一定稳定

---

## 六、一个更严格的判断标准：什么才配进入最终 multi-token 框架

如果我们要让某类 token 真正进入 VoxG 的 anomaly-aware multi-token 设计，我认为至少要满足以下之一：

### 标准 1：它提供稳定的新增量
即在 Delta 主干之上，跨数据集带来稳定收益。

### 标准 2：它虽不提供稳定增益，但显著提升解释性，且代价极低
例如它能明确刻画异常社区一致性，而几乎不增加训练负担。

### 标准 3：它引入的是新的结构视角，而不只是 Delta 的重参数化
例如真正的 community-level token、subgraph token、prototype assignment token。

按这个标准看，当前：

- Delta：肯定保留
- Prototype：暂时处于“候选辅助 token”
- Consistency：暂时处于“解释性辅助 token”
- Hop：需要重新评估其是否值得和 Delta 同时保留

---

## 七、关于 Hop vs Delta 的当前判断

### 7.1 从信息重合度看
大概率较高。

### 7.2 从学习偏置看
Delta 更贴近异常检测语义，因此更值得优先保留。

### 7.3 从 Transformer 负担看
若 Hop 与 Delta 同时输入，可能增加冗余 token，稀释注意力预算。

### 7.4 当前更合理的策略
我倾向于：

> **默认以 Delta 为主表示，Hop 不作为并行主 token，而是作为可选辅助或派生背景信息。**

这比直接把 Hop + Delta 都当主干 token 更稳。

---

## 八、我建议的下一步分析

若继续推进，最值得做的不是先加更多数据集，而是加两类分析：

### 分析 A：显式相关性 / 冗余分析
对以下组合计算相关性、CCA、或线性 probe 可恢复性：
- Hop vs Delta
- Delta vs Prototype
- Delta vs Consistency

目标是回答：
- 哪些特征可由 Delta 线性恢复？
- 哪些特征仍有独立残差信息？

### 分析 B：残差增益分析
先用 Delta 预测标签，再看 Prototype / Consistency 是否能解释 Delta 未解释掉的剩余错误。

这比只看最终 AUC 更能回答“互补性”问题。

---

## 九、当前阶段结论

当前我会把结论收敛成下面三句话：

1. **Prototype / Consistency 已经证明有异常相关信号，但其与 Delta 的重叠度可能较高。**
2. **Hop 与 Delta 在原始信息上很可能高度可互相恢复，区别主要在表示偏置，而非信息量本身。**
3. **对下游 Transformer 来说，真正的问题不是 token 数量不够，而是 token 是否真的提供新增坐标系；否则 multi-token 很容易退化为重复编码。**

这意味着 VoxG 的下一步重点，不应是继续增加 token 数，而应是：

> **找出哪些 token 真正提供不可替代的异常检测视角。**
