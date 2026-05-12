# Score Candidates

## 当前已选第一版 score

### V0 主选择
> **Normal-center cosine alignment + residual hybrid score**

\[
s(v)=0.5 \cdot a(v) - 0.5 \cdot r(v)
\]

其中：
- \(a(v) = \cos(z_v, c_{normal})\)
- \(r(v)\) = normal-aware residual

当前选择这一方案的原因：
1. 最符合 normal-only 半监督设定；
2. 直接继承过去一个月中最稳定的两类信号：中心/原型 与 residual；
3. 比 consistency-only 更稳；
4. 比“直接 anomaly score”更不容易陷入 bootstrap 循环。

---

## 第一部分候选：center alignment

### 候选 A1（当前采用）
- `cosine(z_v, c_normal)`
- 优点：稳、尺度无关、适合排序

### 候选 A2
- `-||z_v - c_normal||_2`
- 优点：直观
- 缺点：更受尺度影响

### 候选 A3
- `max cosine(z_v, prototype_i)`
- 多 prototype 版本
- 适合作为第二阶段增强，而不是第一版必选项

当前决定：
> 第一版先用 **A1**。

---

## 第二部分候选：residual

### 候选 B1（当前推荐）
- 基于训练正常节点构造轻量 normal basis / prototype set
- 用这些 basis 对 \(z_v\) 做近似重构
- residual = 重构误差

### 候选 B2
- 投影到 normal center / normal subspace 后的 projection error

### 候选 B3
- 局部上下文 consistency residual
- 如 node-to-neighborhood / patch 的不协调程度

### 候选 B4
- reconstruction residual from a lightweight normal-only autoencoder

当前建议：
> 第一版优先从 **B1 / B2** 中选一个最轻量实现。

理由：
- 更符合“normal explanation failure”的叙事；
- 不直接依赖复杂局部关系估计；
- 更容易作为 weak retrieval prior。

---

## 为什么暂不优先选 consistency-only

虽然 consistency / coordination 在过去探究里非常重要，但当前不建议第一版直接选：
- pure node-context consistency score
- pure relation tension score

原因：
1. 更容易受数据集局部噪声影响；
2. 容易和 Patch / Relation 线过早耦合；
3. 第一版应先追求最稳的 weak retrieval prior。

因此，consistency 更适合作为：
> 第二阶段增强项

---

## 为什么不直接用 anomaly probability

不建议第一版直接训练一个 anomaly ranker 再取 top-k，原因是：
1. 太像已经完成检测；
2. bootstrap 风险过大；
3. 会让 tokenization 失去独立的研究意义。

因此，第一版应坚持：
> score 是 retrieval prior，不是最终异常分类器。

---

## 当前结论

第一版 score 选择已经确定为：

> **A1 + (B1/B2) 的 hybrid normality score**

其中：
- alignment 采用 cosine-to-normal-center；
- residual 采用轻量 normal-aware residual；
- 后续若 probe 表明 consistency 有额外信息，再作为第二阶段增强。
