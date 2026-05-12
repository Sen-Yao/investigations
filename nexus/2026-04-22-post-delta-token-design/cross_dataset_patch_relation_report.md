# Cross-Dataset Patch / Relation Report

## 一、背景

今天的 post-delta 探究已经从纯方法设计推进到第一轮轻量验证。目标不是直接训练完整 multi-token Transformer，而是先用低成本 probe 检查：

1. Patch 是否提供了 node-wise 之外的结构信息；
2. Relation 是否比 delta 更自然地承接 NDC；
3. Patch + Relation 是否表现出真正的新增坐标系迹象。

本轮实验使用 `lightweight_patch_relation_validation.py`，在现有节点特征基础上构造 patch / relation 特征，并比较：

- `node_only`
- `patch_only`
- `relation_only`
- `node_plus_patch`
- `node_plus_relation`
- `node_plus_patch_plus_relation`

本轮结果的意义是**方向判断**，不是 SOTA 对比，也不是最终模型性能。

---

## 二、跨数据集结果总表

| Dataset | node_only AUC/AP | patch_only AUC/AP | relation_only AUC/AP | node+patch AUC/AP | node+relation AUC/AP | node+patch+relation AUC/AP |
|---|---:|---:|---:|---:|---:|---:|
| Amazon | 0.9757 / 0.8559 | 0.7936 / 0.2424 | 0.7604 / 0.2871 | **0.9777 / 0.8586** | 0.9754 / 0.8558 | 0.9776 / 0.8585 |
| Elliptic | 0.9351 / 0.6003 | 0.8623 / 0.4908 | 0.7637 / 0.2287 | 0.9405 / 0.7201 | 0.9373 / 0.6388 | **0.9436 / 0.7622** |
| Photo | 0.9490 / 0.7485 | 0.6898 / 0.2554 | 0.6867 / 0.1756 | 0.9523 / 0.7614 | 0.9508 / 0.7558 | **0.9529 / 0.7617** |
| Reddit | 0.6938 / 0.0673 | 0.6061 / 0.0475 | 0.5803 / 0.0450 | **0.7078 / 0.0737** | 0.7052 / 0.0733 | 0.7074 / 0.0730 |
| Tolokers | 0.7219 / 0.3807 | 0.6014 / 0.2835 | 0.5907 / 0.2737 | 0.7216 / 0.3690 | **0.7239 / 0.3758** | 0.7206 / 0.3663 |

---

## 三、数据集级别解读

### 1. Amazon
Amazon 上 baseline 已经非常强，因此新增量空间很小。从数值上看：
- Patch-only / Relation-only 都有可分性，说明不是纯噪声；
- 但组合增益非常有限；
- `node_plus_patch` 略优于 `node_only`，`node_plus_relation` 基本持平。

当前判断：
> Amazon 不是这条路线最有说服力的数据集，但也不能据此否定 Patch / Relation，因为 baseline 已经接近饱和。

### 2. Elliptic
Elliptic 是本轮最重要的正反馈数据集：
- Patch-only 已经很强；
- `node_plus_patch` 相比 baseline 有明显提升；
- `node_plus_patch_plus_relation` 达到本轮最佳。

当前判断：
> Elliptic 强烈支持 Patch 作为 post-delta 第一主线，也说明 Relation 与 Patch 之间可能存在互补信息。

### 3. Photo
Photo 上结果也比较稳定：
- Patch / Relation 单独都能做出一定区分；
- 与 node 组合后都有正增益；
- `node_plus_patch_plus_relation` 略优于各单分支。

当前判断：
> Photo 提供了 Patch + Relation 路线的第二个稳定正例，支持继续沿这条线收敛。

### 4. Reddit
Reddit 本身是难数据集，baseline 较弱，但：
- Patch / Relation 组合后仍然有小幅提升；
- 说明这两类特征至少不是无效噪声。

当前判断：
> Reddit 不足以证明这条线很强，但可以说明其方向性是对的。

### 5. Tolokers
Tolokers 是当前的边界例子：
- Patch 几乎没有正增益；
- Relation 有轻微 AUC 改善，但 AP 没有同步强化；
- 组合效果不稳定。

当前判断：
> Tolokers 提醒我们：Patch / Relation 不是普适增益，它们的价值具有数据集依赖性。

---

## 四、跨数据集总判断

### 1. Patch 比 Relation 更稳
从 5 个数据集的整体表现看，Patch 的收益更稳定，尤其在：
- Elliptic
- Photo
- Reddit

因此，当前优先级判断可以进一步收紧为：
> **Patch Token 是 post-delta 路线的第一优先级主方向。**

### 2. Relation 更像补充视角
Relation 单独的稳定性不如 Patch，但它在多个数据集上能与 Patch 形成一定互补，尤其是在：
- Elliptic
- Photo

因此，当前更合适的定位是：
> **Relation Token 是重要的第二视角，而不是当前第一主抓手。**

### 3. Patch + Relation 在部分数据集上形成明显互补
这最明显地体现在：
- Elliptic
- Photo

说明 Relation 并不是纯重复信息，而可能在某些图结构和异常模式中补足 Patch 未覆盖的上下文关系信息。

---

## 五、与 NDC / ANR 的对应关系

本轮结果与此前的现象判断基本一致：

- **ANR → Patch**
  - Patch 更能承接局部异常聚集、局部结构单元异常；
  - Elliptic / Photo 的结果尤其支持这一点。

- **NDC → Relation**
  - Relation 更适合表达“节点与上下文之间的协调/不协调”；
  - 虽然它单独不如 Patch 稳，但在组合中具有解释价值。

因此，本轮实验最大的价值，不只是数值提升，而是：
> **它让 NDC / ANR 从分析现象变成了 token 设计原则。**

---

## 六、当前结论

如果要把今天这轮实验压缩成一句阶段性结论，可以写成：

> **post-delta 路线已经获得跨数据集初步支持，其中 Patch Token 是当前最有前景的主方向，Relation Token 作为补充关系视角在多个数据集上提供了额外信息，尤其在 Elliptic 与 Photo 上最值得继续深挖。**

---

## 七、下一步建议

### Next Action 1
从“轻量 probe 验证”推进到“Patch-first 方法设计”，优先回答：
- Patch 用哪种定义（`P-C1` 还是 `P-C2`）？
- Patch token 如何构造？
- `Node Token + Patch Token` 如何接入下游？

### Next Action 2
为 Relation 做角色收紧，而不是继续发散：
- 它是独立 token family？
- 还是 patch 的辅助关系分支？
- 还是 residual explainer？

### Next Action 3
Prototype Assignment / Contrast 先作为统一框架中的参考层保留，暂不进入第一轮主方法实现。

---

## 八、当前推荐的执行顺序

1. Patch-first 方法草案；
2. Relation 角色定位；
3. 再决定是否进入 unified multi-token Transformer。
