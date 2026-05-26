# DualRefGAD Learning Signal Recovery

> 创建时间：2026-05-26 15:00 CST  
> 状态：🟡 活跃 / 新探究  
> 上游探究：`2026-05-25-dualrefgad-reliability-heterogeneous-proxy-map`  
> 上游报告：`https://report.senyao.org/reports/2026/05/26/dualrefgad-learning-signal-recovery-discussion-2026-05-26.html`

## 1. 中心问题

本探究承接 Reliability / Heterogeneous Proxy Map 的收尾结论：当前问题不是“response matrix 是否有信号”，也不是“再加一个 learnable head 是否能刷分”，而是：

> 在只有正常标签可用于训练、异常标签只能用于诊断的边界下，能否从 reference relation 与 response matrix 中恢复一个可靠、可审计、可训练的 label-free 学习目标 `T(v)`？

第一原则：**先验收目标，再训练目标。**

若固定公式层或低容量目标都不能稳定改善 AP、减少 false-positive reintroduction，或不能解释 recovered anomaly / reintroduced FP 的差异，则不进入更高容量训练模块。

## 2. 研究边界

### 2.1 允许

- 复用上游 C-LEG3 / `old_exact_080_regime` 的 response matrix、split discipline 与 5 seed 结果。
- 编写 runner-registered pure probe，输出固定公式候选、autopsy、fragmentation 分解、reference relation reliability 诊断。
- anomaly labels 只用于 probe 后的 AUC/AP、top-K category/autopsy 与 diagnostic-only 解释。
- 使用所有可用 GPU 执行独立 pure-probe tasks，但 strict reproduction / RNG-sensitive 环节优先保持确定性。

### 2.2 不允许 / 非目标

- 暂不做大 head。
- 暂不做 GT reader / Matrix image-like reader。
- 暂不用异常标签训练、调参、早停或构造公式。
- 不把 AUC 微升直接升级为方法贡献；必须解释 top-ranked AP、FP reintroduction、shortcut 与 seed 稳定性。

## 3. ABCD 实验阶段

### Phase A — Top-K failure autopsy 扩展

重新划分并解释四类节点：

1. `mat_mean rescued anomaly`
2. `mat_mean introduced FP`
3. `Layer 1 recovered anomaly`
4. `Layer 1 reintroduced FP`

对每类统计 row/column dispersion、reference hubness、dropout stability、degree/time/regime descriptor、fragmentation 来源分量。目标是回答：当前 proxy 为什么救回 anomaly 的同时引回 FP。

### Phase B — Fragmentation decomposition probe

把单一 fragmentation 拆成 normal-side、deviation-side、pair-interaction、hub-dominance 与 regime-conditional components，构造若干固定公式候选 `T(v)`。不训练；异常标签只在输出后做 AUC/AP 与 oracle category 诊断。

### Phase C — Reference relation reliability probe

检查可靠 reference relation 的必要条件：reference dropout 后响应是否稳定、强 pair 是否由少数 reference 主导、normal/deviation references 是否在 descriptor 上跨 regime、pair contribution 是否能区分 true anomaly 与 false positive。

### Phase D — Trainable target readiness

只有候选 `T(v)` 通过固定公式 continuation gate，才设计浅层 label-free objective。Phase D 的目标不是立即训练大模型，而是判断：哪些固定目标具备转化为 shallow target-learning objective 的资格；如果没有候选通过，则回退 reference constructor。

## 4. Continuation gates

至少满足以下之一才允许继续到 trainable target：

1. **AP gate**：5 seed 上稳定改善 AP，尤其改善 top-ranked anomaly density。
2. **FP gate**：相对 `mat_mean` 减少 false-positive reintroduction，且不明显牺牲 anomaly retention。
3. **Shortcut gate**：不是 `margin` / `mat_mean` / degree / rejection 的同序改写。
4. **Seed gate**：5 seed 方向一致；若 seed 间分歧，必须由 split fingerprint 或 mechanism 解释。

## 5. 预期产物

- `experiments/scripts/dualrefgad_learning_signal_abcd_probe.py`
- `experiments/configs/dualrefgad_learning_signal_abcd_probe.yaml`
- `experiments/outputs/dualrefgad_learning_signal_abcd_probe.json`
- `experiments/outputs/dualrefgad_learning_signal_abcd_probe.progress.json`
- `experiments/logs/dualrefgad_learning_signal_abcd_probe.log`
- D 完成后补充上游 HTML 报告或发布 sibling result report。
- `PROGRESS.md` / `insights.md` 记录实验事实、解释和后续方向。

---
*Created: 2026-05-26 | Hermes / research-investigation skill*

## 6. ABCD 后续定位（2026-05-26 terminal）

ABCD probe 已完成。结论不是“学习方向失败”，而是：**ABCD 已经把问题推进到机器学习方法形态，但当前 label-free target 尚未强到可以直接晋升为正式训练目标。**

- `mat_mean` 仍是最强稳定读数：AUC/AP = 0.8104 ± 0.0068 / 0.5593 ± 0.0279。
- 固定公式层存在弱但真实的 continuation hints：best-AUC fixed formula ΔAUC `+0.0007`；best-AP fixed formula ΔAP `+0.0031`。
- 浅层 label-free gate 是本探究的关键跨越：它不使用异常标签训练，而是用 response-matrix 派生的锚点和单调约束学习一个低容量 gate。当前 best Layer-1 相对 `mat_mean` ΔAUC `-0.0026`、ΔAP `-0.0022`，因此只保留 diagnostic / target-shaping 角色。
- 下一步不应堆大 head；应先改进 reference relation 与 fragmentation 分解，让固定目标先通过 AP/FP continuation gate。

补充报告：`https://report.senyao.org/reports/2026/05/26/dualrefgad-learning-signal-abcd-result-supplement-2026-05-26.html`

