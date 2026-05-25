# DualRefGAD D-MiT Matrix Token Readout

> 创建时间：2026-05-22  
> 状态：🟡 活跃 / 新探究  
> 上游探究：`2026-05-21-dualrefgad-constraint-calibrated-reference-relation`  
> 已批准代号：**D-MiT** = Dual-reference Matrix Token readout  
> 第一阶段服务器：HCCS-25  
> 第一阶段执行原则：通过 `experiment-runner` 的 runner-registered probe pattern 登记、启动、监控；不手动绕过 runner。

## 1. 中心问题

上游 C-LEG3 / old-exact 固定参考构造已经证明：完整 response matrix 的简单均值 `mat_mean` 是很强的标量基线。Step-1 decomposition gate 中，`mat_mean` 在 5 seed 上稳定获胜；Step-2 autopsy 进一步说明，`mat_mean` 的优势来自边界重排：它能救回一批 margin 漏掉的异常，也能移除一批高 margin 正常假阳性。

因此，本探究不问“再加一个大模型能不能刷分”，而问一个更窄的问题：

> 在固定 C-LEG3 / old-exact response matrix 后，把每个 normal-reference / anomaly-reference pair 的响应值视为 token，是否存在一个 learnable token reader 能读出比 `mat_mean` 更互补、更可解释、且不依赖错误位置先验的异常信号？

## 2. 命名与分档

- **D-MiT**：Dual-reference Matrix Token readout，矩阵 token 读出这一类设计的总称。
- **C-MiT**：C-LEG3 Matrix Token readout，第一阶段固定 C-LEG3 / old-exact 参考构造的 D-MiT probe。
- **RePair-T**：Reference-Pair Transformer，解释性别名，用于强调每个 token 是一个 normal-reference / anomaly-reference pair。

第一轮执行 V0/V1/V2：

1. **V0 / Set-D-MiT**：无位置编码；64 个 entry 只作为集合/分布读入。它只能学习 robust pooling、分位数、稀疏强响应、分布形状等 set-level 信号，不能声称理解矩阵几何结构。
2. **V1 / ID-D-MiT**：加入 row-id / col-id / reference identity，但不加入 2D 邻域假设。它检验特定 reference slot 是否有稳定身份含义。
3. **V2 / Grid-D-MiT**：加入 2D 位置或相对位置偏置。只有当 reference 排序稳定且 permutation/order controls 通过时，才允许讨论“矩阵结构”或“grid-like structure”。

## 3. 位置语义边界

本探究明确不把 response matrix 当作天然图像。行列顺序来自 reference selection 排序，而不是物理空间、图像邻域或稳定二维几何。若 V2 使用 2D 位置编码，它必须面对顺序控制：如果打乱 reference 顺序后性能崩溃，只能说明模型依赖该输入顺序；不能直接证明它学到了科学上稳定的矩阵结构。

所以第一步先运行 V0：如果没有位置信息时仍能形成有用信号，说明 full-matrix distribution 本身有可学习空间；如果 V0 只复刻 `mat_mean` 或明显低于标量基线，则后续 V1/V2 的价值需要谨慎解释。

## 4. 两条训练约束

第一阶段不引入复杂组合损失，回到两条文字约束：

1. 对已知正常节点，最终异常分数应低；
2. 若引入伪异常，伪异常分数应高于其源正常节点。

V0 的 learnable probe 只使用已知正常节点和由其生成的伪异常进行训练。真实 anomaly labels 只用于最终 AUC/AP、top-k、相关性和 autopsy 诊断，不参与训练、早停或 checkpoint 选择。

## 5. 成功/停止规则

V0 不是最终方法，只是门禁。

继续推进的软条件：

- V0 的 AUC/AP 接近 `mat_mean`，且与 `mat_mean` 的 Spearman / top-k overlap 不过高，说明它提供互补排序；或
- V0 在某些 seed 的 top-k autopsy 中能减少 `mat_mean` 的假阳性，同时不明显损失真异常；或
- V0 虽不赢 `mat_mean`，但清楚显示 set-level 分布形状能够解释 Step-2 中“异质真异常被惩罚”的失败模式。

降级或停止条件：

- V0 稳定低于 `mat_mean` 且高度同序，说明它只是学了均值或分位数的软版本；
- V0 只在 pseudo constraints 上表现好，真实 anomaly AUC/AP 不动，说明训练约束没有迁移到真实异常排序；
- V0 的收益主要来自训练伪异常生成方式，而非 response matrix 读出本身。

## 6. 首个实验

首个实验为 `v0_set_dmit_probe`：

- 固定 C-LEG3 / old_exact reference regime；
- response matrix 形状为 `normal_k × anom_k = 4 × 16`，共 64 entry tokens；
- Set-D-MiT 不加入 row/column/2D 位置编码；
- 使用 5 seeds：0,1,2,3,4；
- 通过 HCCS-25 的空闲 GPU 并行执行；
- 结果拉回到本 investigation 的 `experiments/outputs/`、`experiments/logs/`；
- 完成后发布详细 HTML 报告。

---
*Created: 2026-05-22 | Hermes / research-investigation skill*