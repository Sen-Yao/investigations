# Post-Delta Multi-Token Framework（第一版）

## 一、核心目标

本框架尝试把 post-delta 时代的三类新 token 统一到一个更清晰的 VoxG 设计蓝图中。目标不是继续扩展 hop/delta 家族，而是围绕真正不同的信息来源，构造 anomaly-aware multi-token family。

三类核心 token：

1. **Patch Token**：局部结构单元
2. **Relation Token**：节点—上下文关系单元
3. **Prototype Assignment / Contrast Token**：模式归属与参考偏离单元

---

## 二、三类 token 的分工

### 1. Patch Token
回答：
- 节点所在的局部结构单元是什么？
- 是否存在局部异常团块或社区型异常？

### 2. Relation Token
回答：
- 节点与当前上下文是否协调？
- 它与邻域 / patch / community 的关系是否异常？

### 3. Prototype Token
回答：
- 节点属于哪个模式？
- 它偏离哪个参考原型？
- 它是否处在模式边界或模糊区？

---

## 三、为什么这比 hop/delta 更合理

hop/delta 路线的问题在于：
- 基本对象始终是同一个节点
- 新 token 往往只是同一传播轨迹的重参数化
- 很难保证真正新增坐标系

而 post-delta 框架中：
- Patch 改变建模粒度
- Relation 改变建模对象
- Prototype 改变解释层次

因此它们的新增量不是来自代数变换，而来自：

> **信息来源、建模对象、语义层次的变化。**

---

## 四、当前主次结构建议

### 主 token family
1. **Node Token**（保留基础节点表示）
2. **Patch Token**
3. **Relation Token**

### 辅助 token / 参考层
4. **Prototype Assignment / Contrast Token**

当前我不建议一上来让 prototype 也作为与 patch 同等级的主 token，而更建议先把它作为模式参考层，用于增强 relation / patch 的解释与对比。

---

## 五、与 NDC / ANR 的闭环

### ANR → Patch Token
ANR 说明异常具有局部聚集倾向，因此 patch token 是最直接的表示层回应。

### NDC → Relation Token
NDC 说明异常与邻域关系特殊，因此 relation token 是最直接的设计转化。

### NDC / ANR → Prototype Token
二者共同提示节点可能落在某些局部模式中，而 prototype token 提供了模式归属与参考偏离的解释层。

这意味着：

> NDC 和 ANR 不应再被只当作分析现象，而应被上升为 token family 的设计原则。

---

## 六、当前最自然的实现顺序

### 第一阶段
- 先构造 Patch Token 与 Relation Token 的最小实现
- 检查它们是否能超出 hop/delta 的重参数化范围

### 第二阶段
- 再引入 Prototype Assignment / Contrast 作为模式参考层
- 检查它是否解释 Patch / Relation 未解释的残差信号

### 第三阶段
- 统一到真正的 multi-token Transformer 输入框架中

---

## 七、当前结论

如果要用一句话概括这个 post-delta 框架，我会写成：

> **VoxG 的下一代 token 设计，不应继续围绕传播轨迹重写，而应围绕局部结构单元、上下文关系、模式归属三种不同信息来源来组织 anomaly-aware multi-token family。**

这也意味着，Tokenphormer 对我们的真正启发已经被重新翻译为：

- 不是“怎么再造一个 delta token”
- 而是“怎么为异常检测找到真正不同的信息来源，并把它们做成 token family”
