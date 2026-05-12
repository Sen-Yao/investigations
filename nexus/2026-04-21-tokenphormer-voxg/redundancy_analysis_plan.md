# 冗余分析计划

## 一、目标

本计划的目标，是把当前关于特征冗余与互补性的讨论落到可检验的问题上。核心要回答三件事：

1. **Hop 与 Delta 是否对下游 Transformer 过于信息重合？**
2. **Prototype / Consistency 与 Delta 的重叠程度有多高？**
3. **哪些 token 真的提供“新增坐标系”，而不是同一传播信息的重复编码？**

---

## 二、分析主线

### 主线 A：Hop vs Delta 冗余分析
目标是判断：
- Hop 是否大部分可由 Delta 线性恢复
- Delta 是否大部分可由 Hop 线性恢复
- 二者的差异是否主要停留在“学习偏置”层，而非信息层

### 主线 B：Delta vs Prototype / Consistency 互补性分析
目标是判断：
- Prototype / Consistency 是否只是 Delta 的局部平滑版本或关系摘要
- 它们是否还能解释 Delta 未覆盖的剩余信号

---

## 三、建议的分析模块

### 模块 1：基础相关性分析

#### 目的
先做最便宜的第一层判断，检查各组特征之间是否存在强相关。

#### 对象
- Hop flatten vs Delta flatten
- Delta flatten vs Prototype features
- Delta flatten vs Consistency features

#### 指标
- Pearson correlation（逐维平均 / 绝对值均值）
- Cosine similarity（样本级平均）
- 距离分布比较

#### 作用
如果已经出现极强相关，则基本说明二者高度冗余。

---

### 模块 2：线性可恢复性分析

#### 目的
判断一组特征能否被另一组特征线性重建。

#### 对象
- 用 Hop 预测 Delta
- 用 Delta 预测 Hop
- 用 Delta 预测 Prototype
- 用 Delta 预测 Consistency

#### 方法
- Ridge regression / linear regression
- 评估 R²、MSE

#### 解释
- 若 `Delta -> Hop` 和 `Hop -> Delta` 的 R² 都很高，则二者高度可互相恢复
- 若 `Delta -> Prototype` 的 R² 很高，则 Prototype 更像 Delta 的重参数化
- 若 `Delta -> Consistency` 的 R² 较低，则 Consistency 可能包含额外关系信息

---

### 模块 3：CCA / 子空间重叠分析

#### 目的
判断两组表示在低维主子空间上是否高度重叠。

#### 对象
- Hop vs Delta
- Delta vs Prototype

#### 方法
- Canonical Correlation Analysis (CCA)
- PCA 后比较主子空间夹角（可选）

#### 作用
这一步能避免只看原始维度相关，而是观察“整体表示空间”是否本质相近。

---

### 模块 4：残差增量分析

#### 目的
直接回答“Prototype / Consistency 是否能解释 Delta 没解释掉的部分”。

#### 方法
1. 先用 Delta flatten 做轻量 probe
2. 记录预测误差 / logit / residual
3. 再检查 Prototype / Consistency 是否与 residual 显著相关
4. 或在 Delta 基础上加入新特征，看 residual 是否被系统性减少

#### 解释
这一步比只看最终 AUC 更能回答“新增量”问题。

---

### 模块 5：token 对 Transformer 的理论负担分析

#### 目的
从建模视角讨论：如果 Hop + Delta 同时输入，是否会造成注意力冗余。

#### 关注点
- token 数增长
- token 间线性依赖
- 是否需要 Transformer 自己学“差分”或“积分”
- 是否存在更优输入方式（如只保留 Delta，把 Hop 作为 side info）

#### 结论输出形式
不一定是数值实验，也可以形成方法设计建议。

---

## 四、建议的执行顺序

### 第一步：Hop vs Delta 线性可恢复性
这是当前最重要的问题，因为它决定 Hop 是否值得作为主 token 保留。

### 第二步：Delta vs Prototype / Consistency 的线性可恢复性
这一步判断新 token 的冗余度。

### 第三步：残差增量分析
这一步判断“即使相关性高，是否仍有剩余信息”。

### 第四步：如有必要，再做 CCA
CCA 更重，但适合作为更强佐证。

---

## 五、优先数据集

### 第一优先级：Photo
原因：
- 已观察到小幅实际增益
- 最适合检查“互补性为何成立”

### 第二优先级：Amazon
原因：
- 统计显著但无增益
- 最适合检查“为何有信号却无新增量”

这两个数据集一正一反，足够支撑第一轮冗余分析。

---

## 六、预期输出

本轮分析结束后，希望得到下面几类判断：

1. Hop 与 Delta 是否可高度线性互相恢复
2. Prototype 是否基本可由 Delta 预测
3. Consistency 是否比 Prototype 保留更多独立残差信息
4. 对最终 multi-token 框架，应保留哪些 token，舍弃哪些 token

---

## 七、当前建议

我建议先实现一个小型分析脚本，至少覆盖：
- Hop vs Delta 线性恢复 R²
- Delta vs Prototype 线性恢复 R²
- Delta vs Consistency 线性恢复 R²
- Delta 基线 residual 与新特征的相关性

只要这四项结果出来，我们就能更清楚地判断：

> **当前的 multi-token 设计是在增加视角，还是在重复编码。**
