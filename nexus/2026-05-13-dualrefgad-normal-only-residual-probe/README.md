# 2026-05-13-dualrefgad-normal-only-residual-probe

## 研究问题

DualRefGAD 当前已经有一个很强但仍不够稳定的几何基线：`margin-only` 在既有记录中达到 AUC `0.7952±0.0071`，而带 learned head 的 `dual_margin_two_score` 反而下降到 `0.7455±0.0188`。这说明问题不在于“双参考几何完全无效”，而在于训练目标、学习头或额外参数可能破坏了原本有用的排序结构。

本探究关注一个更窄、更干净的问题：在 normal-only semi-supervised GAD 协议下，固定强基线之后，是否还存在可学习的残差信号？如果有，这个残差信号应该被解释为 normal manifold deviation / reference inconsistency，而不是简单在 margin 上叠一个表面 correction head。

## 假设

1. **H1：冻结 margin 排序后，仍可能存在残差异常信号。** 这个残差不应由真实 anomaly label 选择或训练，只能用已知 normal 节点的 normal-only 约束来学习；anomaly label 仅用于最终诊断评估。

2. **H2：如果 residual probe 只能学到全局 suppression / calibration，而不能稳定改变 top-k 排序，那么这条路线不应被包装成最终方法。** 这种结果只说明 normal-only 训练信号可以改变分数尺度，不说明发现了新的异常排序几何。

3. **H3：如果 residual probe 在 5-seed 下稳定提升 margin-only，并且 Spearman、top-k membership、correction variance 都显示出非平凡排序改变，那么下一步应把 residual 机制重构为统一 scoring principle。** 也就是说，成功信号不是“保留 additive patch”，而是反推更可解释的 reference / embedding 机制。

## 关键发现

本探究刚创建，尚未运行 HCCS-88 5-seed 诊断。已有前置证据来自 `2026-05-09-semisupervised-negative-signal-for-dualrefgad`：

| 前置观察 | 数值或结论 | 对本探究的含义 |
|---|---:|---|
| margin-only 强于 learned head | `0.7952±0.0071` vs `0.7455±0.0188` | 学习头可能破坏几何排序，不能默认“多学一点就更好”。 |
| N1/N2/N3/N4 proxy 不足以作为主证据 | proxy AUC 只证明规则与 hand-crafted score 兼容 | 本探究必须避免把人工 proxy 当作真实异常检测证据。 |
| VecGAD / GGAD / RHO 的共同启发 | negative/contrastive 信号需要理论来源 | residual 需要 grounded in normal-manifold deviation 或 reference inconsistency。 |

**Insight：** 当前最重要的不是追求一个好看的新分数，而是判断“margin 之外是否有稳定、协议干净、可解释的剩余信号”。如果答案是否定的，这也是有效科研结论：停止在 margin 上堆 head，转向 reference 机制本身。

## 诊断协议

本探究把 additive residual 视为 **diagnostic probe**，不是最终方法叙事。

- 冻结 baseline / margin ranking backbone。
- 训练只使用 labeled normal nodes；known normal split 为 train / validation。
- anomaly labels 只用于最终诊断，不用于训练、早停、checkpoint selection。
- correction head 必须低容量、bounded magnitude、带 no-op / near-zero 初始化。
- checkpoint 由 normal-only validation loss 选择，不由 AUC/AP 选择。
- 报告必须同时给出 baseline 与 probe 的 delta，而不是只报绝对 AUC/AP。

## 必须记录的指标

| 指标 | 目的 |
|---|---|
| `delta_auc`, `delta_ap` per seed and mean±std | 判断是否稳定超过 margin-only。 |
| `corr_mean`, `corr_std`, `corr_abs_mean` | 区分没训练、全局 suppression、真实 residual。 |
| Spearman(final score, baseline score) | 判断排序是否几乎只是 monotonic calibration。 |
| top-k overlap / ratio changes | 判断异常候选集合是否发生实质变化。 |
| selected epoch + validation loss components | 确认选择规则协议干净。 |

## 实验记录

| 日期 | 实验 | 结果 | 备注 |
|------|------|------|------|
| 2026-05-13 | 创建 investigation | 待跑 HCCS-88 5-seed | 当前只完成归档与诊断设计。 |

## 下一步

等待用户确认后，在 HCCS-88 上运行 5-seed normal-only residual diagnostic probe。若结果无稳定提升，直接关闭这条 additive residual 路线，并记录为 negative finding；若结果稳定提升，则分析 residual 学到的结构，再将其重写为统一的 reference-inconsistency / normal-manifold scoring 机制。

---
*Created: 2026-05-13 | Nexus Agent*
