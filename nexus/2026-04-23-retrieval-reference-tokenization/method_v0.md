# Method V0

## 一、任务设定

本方法严格采用与 VecGAD 对齐的半监督设定：
- 只能访问训练集的一部分节点；
- 只有训练集中的正常标签可见；
- 所有异常标签不可见；
- 测试集标签不可见。

因此，本方法中的 retrieval score 不能依赖真实异常标签，而只能建立在 normal-only 先验与弱异常相关指标之上。

---

## 二、核心思想

当前方法不再把 tokenization 限定在局部 hop / neighborhood 展开，而是尝试从整张图中为每个目标节点检索一组更有参考价值的远程节点，并把它们构造成 reference token family，供 Graph Transformer 交互建模。

第一版输入由三部分组成：
1. `self token`
2. `normal-reference sequence`
3. `anomaly-like reference sequence`

其中 normal/anomaly-like reference 并不是基于几何近邻，而是基于整图范围内的 weak retrieval prior 排名得到。

---

## 三、第一版 score 决策

### 当前已定
第一版采用：

> **Normal-center cosine alignment + residual hybrid score**

定义一个耦合的 normality continuum score：

\[
s(v)=\lambda \cdot a(v) - (1-\lambda) \cdot r(v)
\]

其中：
- \(a(v)\)：节点与正常模式中心的对齐程度；
- \(r(v)\)：节点对正常解释机制的 residual；
- score 高：更 normal-like；
- score 低：更 anomaly-like。

当前第一版默认：
- \(\lambda = 0.5\)

---

## 四、score 的两部分

### 1. Normal-center alignment
对训练集中可见的正常节点构造 normal center：

\[
c_{normal} = \frac{1}{|N_{train}|}\sum_{u \in N_{train}} z_u
\]

然后定义：

\[
a(v)=\cos(z_v, c_{normal})
\]

第一版选择 cosine 而不是 L2，原因是：
- 更稳；
- 对尺度不敏感；
- 更适合直接做 ranking。

### 2. Residual
Residual 表示节点多大程度上无法被正常模式解释。

第一版不使用复杂大模型，而采用轻量 normal-aware residual 机制，例如：
- 基于训练正常节点构造轻量 normal basis / prototype set；
- 用这些正常模式近似当前节点表示；
- residual 定义为重构误差或投影误差。

第一版目标不是得到最强 residual，而是得到一个：
> **不依赖异常标签、但能表达“偏离正常机制”的弱指标。**

---

## 五、整图 retrieval

对于每个目标节点，不在局部邻域中取 token，而是在整图范围内基于 \(s(v)\) 排名：

- `top-k highest score` → `normal-reference sequence`
- `top-k lowest score` → `anomaly-like reference sequence`

当前第一版采用：
> **整图统一 retrieval**

也就是 reference bank 在整图中维护，再为每个节点取序列。

当前不采用复杂的 per-node pairwise retrieval，以避免第一版复杂度过高。

---

## 六、模型流水线

第一版整体流程为：

`hybrid score -> global retrieval tokenization -> Graph Transformer -> readout -> embedding -> MLP -> anomaly score`

更细化地说：

### Step 1
构造 normal center 与 residual 机制。

### Step 2
对整图所有节点计算耦合 score \(s(v)\)。

### Step 3
根据 score 形成：
- `normal-reference sequence`
- `anomaly-like reference sequence`

### Step 4
将三部分 token 输入 Graph Transformer：
- self token
- normal-reference tokens
- anomaly-like reference tokens

### Step 5
经过 readout 得到节点 embedding，最后用 MLP 输出 anomaly score。

---

## 七、bootstrap 风险的处理

本方法的关键风险在于：
- 如果 retrieval score 本身已经近似完成异常检测，那么 tokenization 只是在重复检测；
- 这样会削弱方法的合理性和创新性。

因此，第一版必须坚持：

> **hybrid score 不是最终异常检测器，而是 weak retrieval prior。**

它只负责构造参考序列，不负责完成最终分类。最终异常判断必须依赖：
- self token
- reference tokens
- GT interaction
- downstream readout / MLP

---

## 八、第一版目标

第一版不直接追 SOTA，而先回答：
1. hybrid score 是否真有信息量；
2. top-k reference 是否比随机参考更合理；
3. reference token 是否真提供了局部邻域之外的新增量；
4. GT interaction 是否确实比“只用 score 排名”更强。

这四个问题回答清楚后，才适合继续推进完整实验与结构优化。
