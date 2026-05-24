# DualRefGAD: normal-low + reference-dropout reliability

> Created: 2026-05-24  
> Project: DualRefGAD  
> Status: investigation created; first-step probe specified, not yet run.

## Research direction

本 investigation 记录 DualRefGAD 的新早期探究方向：**Reference Reliability 的最小验证**。

核心转向：早期训练不再堆叠五项损失函数，而是只验证最本质、最符合异常检测第一性原理、可解释性强的两个 loss：

1. **`L_normal-low`**：已知正常节点低异常分数约束。  
   - 异常检测首先学习 normality。
   - 训练标签只使用 `L_n` 中的 normal label。
   - 不使用异常标签，不生成伪异常。

2. **`L_ref-drop`**：reference dropout consistency / reference 子采样一致性。  
   - 一个节点是否异常不应强依赖某个偶然 reference 点。
   - 可靠的 reference relation 应对合理 reference 子集扰动稳定。
   - 它学习的是 reliability，而不是 anomaly label。

## Scope of the first step

第一步实验目标不是最大化 benchmark AUC，而是回答一个更早期的问题：

> 在只使用 normal-low 与 reference-dropout consistency 的情况下，shallow reliability gate 是否能学习到稳定的 reference reliability，而不是退化成 margin proxy？

## Explicit non-goals for the first step

以下信号暂不进入训练 loss，只作为 diagnostics / report metrics / stop gates：

- reference ranking
- entropy regularization
- anti-hub regularization
- residual-guided hard negatives

原因：早期目标过多会掩盖机制判断，并可能造成灾难性验证。第一步最多两个 loss。

## Directory structure

- `README.md` — investigation overview and direction
- `hypothesis.md` — hypotheses, loss rationale, non-goals, stop rules
- `insights.md` — initial conclusion and expected mechanistic question
- `PROGRESS.md` — timeline and next actions
- `experiments/configs/first_step_probe.yaml` — first-step probe placeholder config; not a completed run
- `experiments/scripts/` — reserved for runner-compatible scripts
- `experiments/outputs/` — reserved for results
- `experiments/plots/` — reserved for plots
- `references/` — reserved for references and external notes
