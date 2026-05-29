# DualRefGAD Loss Portfolio Selection + Multi-objective Optimizer

> Created: 2026-05-29 19:32 CST  
> Status: active / runner-managed probe preparation  
> Source design report: https://report.senyao.org/reports/2026/05/29/dualrefgad-loss-selection-mtl-optimizer-2026-05-29.html

## 主体

本探究承接报告《从 8 个候选项里挑 3–4 个核心项，再用多目标优化处理冲突》的核心建议：不要把 ROCC 现有三项与额外候选 loss 一次性堆叠，而是先做 **loss portfolio selection**，用 A0→A1→A2→A3 的增量链条判断每一项是否有独立价值；再在留下的少数目标上比较 weighted sum / PCGrad / FAMO / CAGrad 等多目标优化器。

## 研究问题

在 DualRefGAD / RIFT-GT 当前 response-manifold 训练语境下：

1. A0 当前 ROCC-MC（`LN + LU + Lvar`）作为基线，实际 allowed monitors 与最终诊断 AUC/AP 的表现如何？
2. A1 将 `LU` 改为 hinge/dead-zone，并把 anti-collapse 改为 rank barrier 后，是否减少目标错配和训练塌缩风险？
3. A2 加入 reference-view consistency 后，是提升 reference dropout / view stability，还是稳定了错误排序？
4. A3 加入 pair reliability weighting 后，是否能筛除低质量 reference-pair 干扰，而不引入额外 heuristic 偏置？
5. 若 A0→A3 中只有部分项有效，后续 optimizer 比较应在哪个 portfolio 上进行？

## 实验边界

- 本探究先跑 A0→A1→A2→A3 的增量链；不直接做 8 loss 全排列。
- 首轮以 runner-registered probe 形式执行，产出 terminal aggregate JSON 后再解释。
- allowed monitors 用于候选筛选；测试 AUC/AP 只作为事后诊断读数，不参与训练/选择。
- 若 runner CLI 无原生 probe launcher，按 `experiment-runner/references/runner-registered-probe.md` 的 bundled multi-variant probe pattern 执行，并用 watchdog 管理进度。

## A0→A3 定义

- **A0**：当前 ROCC-MC：known-normal multi-center energy + trimmed-unlabeled compactness + anti-collapse variance。
- **A1**：A0 的目标形态修正：known-normal multi-center 主项 + trimmed-unlabeled hinge/dead-zone + rank barrier（anti-collapse 保险丝）。
- **A2**：A1 + reference-view consistency（dead-zone consistency）。
- **A3**：A2 + reference-pair reliability weighting。

## 预期交付

1. 代码库 git 状态整理与本探究目录提交。
2. runner-registered A0→A3 probe job id、远端日志、进度 JSON、最终 aggregate JSON。
3. 看门狗监控进度；实验完成后发布 HTML 报告。
4. 报告在既有 URL 所代表的设计报告基础上补充实验结果、诊断和下一步尝试建议。
