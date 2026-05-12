# Insights

## 1. G_a 的职责边界已经明确

`G_a` 不应该回答“候选节点 u 是否与当前节点 v 相关”。它只回答：

> 节点 u 本身是否适合作为 anomaly-reference candidate？

因此 `G_a` 是全局一次性计算的长度 N 数组。当前节点相关性由 `L_a(u|v)` 负责。

## 2. 旧 q*c 不是稳定的 final anomaly-reference prior

`q*c` 在某些数据集上能提供尖端 top-k 候选，但与 `L_a` 组合后并不稳定。Photo 上 baseline `q*c` 的 reference purity 很低，说明单纯结构 anomaly support 不足以作为最终 anomaly-reference eligibility。

## 3. 5% normal labels 可以合法辅助 G_a

VecGAD setting 允许访问少量正常标签。因此使用 `V_train^n` 校准 normal manifold 不构成数据泄漏。当前 `G_a` 使用正常节点估计 center / density / attribute deviation，不使用任何异常标签。

## 4. Normal-calibrated deviation 是当前最强 G_a family

Photo 上最强信号来自 `normal_cal_attr_dev`；Elliptic 上最强信号来自 `normal_cal_max_dev`。二者共同结构都是 normal-calibrated deviation。

这说明跨数据集主方法应写成 multi-view family，而不是固定某个单一 view。

## 5. 加法被采纳为主组合方式

已比较 multiply / add / rank_add。

- Photo：strong G_a 下 add 与 multiply 几乎持平；
- Elliptic：add 的 anomaly-node reference purity 更高且更稳；
- rank_add 在 strong G_a 下明显较弱。

因此主公式采用：

```text
S_a(u|v) = G_a(u) + L_a(u|v)
```

乘法保留为 ablation。

## 6. 当前阶段不应继续无限探索 G_a

Reference purity probe 已经足够支持 normal-calibrated deviation 方向。继续在 Photo/Elliptic 上微调 `G_a` 容易过拟合 probe。下一步应进入完整模型最小训练验证。

## 7. Full model validation 的关键观察项

接入完整模型后需要同时观察：

- AUC / AP；
- normal-reference purity；
- anomaly-reference purity；
- attention 是否关注 anomaly-reference tokens；
- token sequence 是否导致训练不稳定；
- seed-level mean ± std。
