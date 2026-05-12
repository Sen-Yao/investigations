# Implementation Plan V0

## 一、实现目标

当前 V0 的目标不是一次性完成完整论文模型，而是先实现一个最小可运行版本，能够验证以下问题：

1. `hybrid score` 能否被稳定计算；
2. 基于该 score 的整图 retrieval 是否能生成有意义的 normal / anomaly-like reference sequence；
3. GT interaction 是否比“只用 score 排名”更强。

因此，V0 的实现原则是：
- **最小闭环**
- **先 probe，后完整模型**
- **优先保证可解释性**

---

## 二、V0 模块拆解

### Module A: 预表征构造
需要先得到一个轻量、稳定的节点表示 \(z_v\)。

#### 当前建议
第一版不要直接用最终 GT embedding，而使用一个轻量前置表示：
- 可以是当前 VecGAD tokenization 后的浅层表示；
- 也可以是更简单的 feature + local aggregation 表示；
- 目标是给 hybrid score 提供一个稳定、低成本的输入空间。

#### 原则
- 不引入过重模型；
- 不依赖异常标签；
- 与 VecGAD 半监督设定兼容。

---

### Module B: Normal center / normal basis

利用训练集中可见的正常节点，构造：
- `normal center`
- 或轻量 `normal basis / prototype set`

#### V0 建议
第一步先做：
- 单 normal center

第二步可扩展为：
- 多 prototype / basis set

#### 输出
- `c_normal`
- 若有需要，再加 `B_normal`

---

### Module C: Residual estimation

为每个节点定义一个 normal-aware residual。

#### V0 最小方案
优先实现：
- 用正常节点 basis / prototype 对当前节点表示做轻量近似；
- residual = 近似误差。

#### 备选简化
如果 basis 版一开始实现复杂，可先上：
- projection residual
- 或 center-based residual 的轻量近似版

---

### Module D: Hybrid score computation

根据：
- center alignment
- residual

构造耦合的 normality continuum score：

\[
s(v)=0.5 \cdot a(v) - 0.5 \cdot r(v)
\]

#### 输出
对整图每个节点得到一个 score。

---

### Module E: Global reference bank

基于整图所有节点的 score，构造两个 reference bank：
- `normal bank`（高分）
- `anomaly-like bank`（低分）

#### V0 建议
先采用：
- **全图统一 bank**
- 每个目标节点先使用统一 top-k reference 序列

这样实现最简单，也最适合第一轮 probe。

---

### Module F: Tokenization for GT

为每个目标节点构造输入序列：
1. self token
2. normal-reference tokens
3. anomaly-like reference tokens

#### V0 注意
第一版先不做复杂 reference-specific positional encoding，
重点只是验证：
- 这些 token 是否有用
- 是否被利用

---

### Module G: GT + readout + MLP

沿用当前已接受的主框架：
- Graph Transformer
- readout
- embedding
- MLP
- anomaly score

#### V0 原则
尽量不要在这一阶段引入额外复杂模块，以免把问题混在 architecture 本身。

---

## 三、Probe 实现优先级

### Probe 1: score distribution sanity check
**优先级：P0**

实现内容：
- 统计 score 在 normal / anomaly 上的分布
- KS / AUC / rank separation

目的：
- 判断 score 是否有基础信息量

---

### Probe 2: top-k reference purity
**优先级：P0**

实现内容：
- normal bank purity
- anomaly-like bank purity
- 与随机 reference 的对比

目的：
- 判断 retrieval 是否比随机选择更合理

---

### Probe 3: non-locality check
**优先级：P1**

实现内容：
- reference 与目标节点的最短路径距离分布
- 超出局部邻域的比例

目的：
- 判断方法是否真的利用了非局部 reference

---

### Probe 4: interaction necessity
**优先级：P1**

实现内容：
- 只用 score 排名的 baseline
- score + GT interaction 的模型
- 比较两者差异

目的：
- 判断 GT interaction 是否真正提供新增量

---

## 四、建议的执行顺序

### Phase 1
先实现：
- Module A/B/C/D
- Probe 1 + Probe 2

这一步只验证 score 与 retrieval 本身。

### Phase 2
再实现：
- Module E/F
- Probe 3

这一步验证 reference tokenization 是否真的非局部。

### Phase 3
最后实现：
- Module G
- Probe 4

这一步才进入完整的 tokenization-GT-readout-MLP 流水线。

---

## 五、当前最小实现策略

如果要把 V0 压缩成一个工程上最合理的最小实现，我建议是：

1. 先基于轻量预表征构造 `normal center`；
2. 再定义一个最简单可行的 residual；
3. 先在整图上计算 score，并做 retrieval probe；
4. probe 通过后，再把 reference token 接到 GT；
5. 最后再比较：
   - local-only
   - retrieval-only
   - retrieval + GT interaction

---

## 六、当前结论

V0 的实现不应以“马上追最终性能”为目标，而应以“先回答 retrieval tokenization 是否真的成立”为目标。

因此，当前最合理的工程推进方式是：

> **先把 hybrid score 和整图 retrieval 这两个最核心的前置模块做稳，再把它们接入 GT。**
