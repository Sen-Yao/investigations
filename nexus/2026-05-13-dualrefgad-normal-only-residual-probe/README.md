# 2026-05-13-dualrefgad-normal-only-residual-probe

## 研究问题

DualRefGAD 当前已经有一个很强但仍不够稳定的几何基线：`margin-only` 在既有记录中达到 AUC `0.7952±0.0071`，而带 learned head 的 `dual_margin_two_score` 反而下降到 `0.7455±0.0188`。这说明问题不在于“双参考几何完全无效”，而在于训练目标、学习头或额外参数可能破坏了原本有用的排序结构。

本探究关注一个更窄、更干净的问题：在 normal-only semi-supervised GAD 协议下，固定强基线之后，是否还存在可学习的残差信号？如果有，这个残差信号应该被解释为 normal manifold deviation / reference inconsistency，而不是简单在 margin 上叠一个表面 correction head。

## 假设

1. **H1：冻结 margin 排序后，仍可能存在残差异常信号。** 这个残差不应由真实 anomaly label 选择或训练，只能用已知 normal 节点的 normal-only 约束来学习；anomaly label 仅用于最终诊断评估。

2. **H2：如果 residual probe 只能学到全局 suppression / calibration，而不能稳定改变 top-k 排序，那么这条路线不应被包装成最终方法。** 这种结果只说明 normal-only 训练信号可以改变分数尺度，不说明发现了新的异常排序几何。

3. **H3：如果 residual probe 在 5-seed 下稳定提升 margin-only，并且 Spearman、top-k membership、correction variance 都显示出非平凡排序改变，那么下一步应把 residual 机制重构为统一 scoring principle。** 也就是说，成功信号不是“保留 additive patch”，而是反推更可解释的 reference / embedding 机制。

## 关键发现

HCCS-88 5-seed 诊断已完成，结论是 **negative finding / route closed**：normal-only residual correction 可以被训练出来，但没有发现稳定、baseline-independent 的异常排序信号。

| 观察 | 5-seed 数值或结论 | 对本探究的含义 |
|---|---:|---|
| margin-only 与 residual score 几乎相同 | AUC `0.7952±0.0071` → `0.7953±0.0071`; ΔAUC `6.29e-05±2.72e-04` | 没有实质提升。 |
| AP 不稳定且平均下降 | AP `0.5165±0.0220` → `0.5161±0.0221`; ΔAP `-4.46e-04±2.50e-03` | residual 不具备稳定收益。 |
| final score 与 margin 几乎同序 | `spearman(score, margin)=0.9985±0.0002`; `top5_jaccard=0.9996±0.0005` | correction 没有改变候选集合。 |
| correction 主要是 margin compression | `corr ≈ a - 0.212 * margin + residual`; `score ≈ a + 0.788 * margin + small_residual` | 学到的是 calibration / compression，不是新异常几何。 |
| ABCD flags | A: global shift, B: margin-linked, C: too weak all raised; D 不是主失败项 | additive residual patch 应关闭。 |

**Insight：** 这个结果支持之前的怀疑：在 margin 上叠 learnable correction/head 多半只是表面修饰。DualRefGAD 的有效信号仍然来自 reference geometry / margin 本身；继续堆 head 不值得。后续若推进，应从 reference 构造、normal-manifold deviation 或 multi-reference distributional inconsistency 重构异常信号，而不是保留 `margin + correction`。

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
| 2026-05-13 | 创建 investigation | 完成归档与诊断设计 | additive residual 仅作为 probe，不作为最终方法。 |
| 2026-05-13 | Stage-3 residual ABCD diagnostics, 5 seeds, elliptic | **Route closed** | Sweep `vtkl5ykv`; 详见 `experiments/outputs/stage3_residual_abcd_analysis.md`。 |

## 下一步

本路线已关闭。不要继续调参 `margin + correction` 作为主方法。下一步应回到更根本的机制选择：

1. reference construction：是否能让 normal/anomaly reference 分布本身更有解释力；
2. normal-manifold deviation：是否能定义协议干净的 normal-only 偏离量；
3. multi-reference distributional inconsistency：是否能用多个 reference 视角的不一致性替代 additive head。

---
*Created: 2026-05-13 | Nexus Agent*
