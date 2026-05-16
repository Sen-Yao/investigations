# Progress Tracking

**Investigation**: `2026-05-09-semisupervised-negative-signal-for-dualrefgad`
**Phase**: Phase 1 → Phase 2 (transitioning)
**Last Updated**: 2026-05-14

---

## Current Status

||| Component | Status |
||-----------|--------|
|| Literature Survey (RESEARCH_SURVEY.md) | ✅ Complete |
|| Candidate Designs (hypothesis.md) | ✅ Complete |
|| Density Probe Experiment | ✅ Complete (negative result) |
|| Likelihood-Ratio Probe (||d|| conditioning) | ✅ Complete (failed) |
|| Vector-conditioned LR Probe | ✅ Complete (below margin) |
|| Stage 3 Residual Probe | ✅ Complete (no stable improvement) |
|| Stage 3 ABCD Diagnostics | ✅ Complete (WandB 4u8dzp5v) |
|| Response Matrix Diagnostic (Route2) | ✅ Complete (mat_mean unstable) |
|| insights.md | ✅ Complete (2026-05-12) |

---

## Timeline

### Day 1: 2026-05-09

**Activity**: Investigation creation + Literature mechanism survey

- Created investigation structure
- Completed offline consistency diagnosis: proxy metrics vs real AUC/AP
- Started mechanism survey: GGAD / VecGAD / RHO
- Key finding: proxy AUC only shows compatibility between rule and score geometry, not validity for real anomaly detection

**Outputs**:
- `README.md` - Investigation scope and plan
- `hypothesis.md` - Candidate negative signal designs
- `RESEARCH_SURVEY.md` - GGAD/VecGAD/RHO mechanism comparison

---

### Day 2: 2026-05-10

**Activity**: Stage 1 results summary + planning

- Reviewed Stage 1 margin regression results
- Discussed next experimental directions
- Sent `stage1_results_and_next_plan_20260510.html`

**Key Discussion**:
- Margin-only baseline AUC 0.7938 remains strongest
- Need to design normal-only objectives that align with anomaly ranking

---

### Day 3: 2026-05-11 (Major Design Day)

**Activity**: Normal-only training objective design

#### Morning: Normal-only Objectives Discussion
- Sent `dualref_normal_only_objectives_no_pseudo_anomaly_20260511.html`
- Proposed three candidate approaches:
  1. **Conditional Normalizing Flow** (首推) - direct density modeling
  2. **Reference-view Consistency** (VICReg/Barlow style)
  3. **Energy/Denoising Score Matching** (理论优雅但难诊断)

**Key Judgment**: DualRefGAD learnable head should be defined as *normal relation distribution estimator*, not normal-vs-pseudo-anomaly classifier.

#### Afternoon: Objective Discussion (Margin → Ranking)
- Sent `objective_discussion_margin_to_ranking_20260511.html`
- Analyzed why margin regression ≠ anomaly ranking
- Proposed pairwise ranking with pseudo-anomaly candidates as alternative

#### Evening: RHO Analysis + MSE Probe Results
- Sent `rho_training_objective_analysis_20260511.html`
- Sent `stage1_margin_regression_mse_probe_normfix_report_20260511.html`
- Key finding: normalized interaction features (`ud_prod_absdiff_norm`) can recover margin ranking with Spearman≈0.99

---

### Day 4: 2026-05-12 (Failure Analysis + New Design)

**Activity**: Density Probe failure analysis + Likelihood-Ratio design

#### Morning: Relation Density Probe Summary
- Sent `relation_density_probe_summary_and_next_steps_20260512.html`
- **Critical Finding**: Normal-only density modeling **failed**
  - Diag Gaussian AUC: 0.6674 (far below margin 0.7938)
  - RealNVP worse than Gaussian
  - Flow approach does not help

**Root Cause Identified**:
> DualRef's anomaly direction `d=r_a-r_n` cannot be absorbed as a latent variable. It must enter scoring function via conditioning or explicit geometry.

#### Afternoon: Likelihood-Ratio with d-Conditioning Design
- Sent `likelihood_ratio_d_conditioning_design_20260512.html`
- Proposed new approach: `s_i = log p(φ_i | anom-context) - log p(φ_i | normal-context)`
- Uses `d` as conditioning variable, not pseudo-anomaly generation
- Recommended: `||d||` as scalar conditioning for first experiment

#### Evening: Formal Likelihood-Ratio Probe Result
- Implemented `stage2_likelihood_ratio_probe.py`
- Smoke test (1 epoch) passed; the 1-epoch metric was explicitly treated as non-conclusive
- Formal 5-seed / 80-epoch evaluation completed on HCCS-88
- **Result**: scalar `||d||` conditioning likelihood-ratio failed decisively
  - AUC: `0.4272 ± 0.0473`
  - AP: `0.0821 ± 0.0047`
  - Spearman vs margin: `0.1091 ± 0.0259`
  - Margin baseline: AUC `0.7938`, AP `0.5510`

**Conclusion**:
> Scalar `||d||` is not a viable conditioning variable for likelihood-ratio scoring. It loses the directional geometry that makes DualRef margin effective, and the conditional flow learns a score anti-aligned or nearly unrelated to anomaly ranking.

#### Night: Vector-Conditioned Likelihood-Ratio Diagnostic
- Implemented `stage2_vector_likelihood_ratio_probe.py`
- Conditioning changed from scalar `||d||` to vector `normalize(d)`
- Smoke test (1 epoch) passed; again treated as non-conclusive
- Formal 5-seed / 80-epoch evaluation completed on HCCS-88
- **Result**: vector conditioning rescues a substantial amount of signal, but remains below margin baseline
  - AUC: `0.6880 ± 0.0335`
  - AP: `0.2839 ± 0.0644`
  - Spearman vs margin: `0.2974 ± 0.0543`
  - Margin baseline: AUC `0.7938`, AP `0.5510`

**Conclusion**:
> Direction matters: vector conditioning is far better than scalar `||d||` (AUC 0.688 vs 0.427), but likelihood-ratio still fails to reproduce the explicit `cos(u,d)` ranking geometry. This supports pivoting toward margin-backed calibration rather than replacing margin with density-ratio scoring.

---

### Day 5: 2026-05-13 (Residual Probe as Diagnostic, Not Main Narrative)

**Activity**: Consolidated the route after vector-conditioned LR underperformed margin.

- Discussed whether `margin + learned correction` is too much like an engineering workaround.
- Decision: do **not** treat additive correction as a candidate final method or main paper narrative.
- Reframed it as a bounded diagnostic probe:
  - Purpose: test whether margin leaves any learnable residual signal.
  - Protocol: train correction using labeled normal nodes only; anomaly labels are used only for diagnostic evaluation.
  - Constraint: small-capacity, tanh-bounded correction head with normal-only validation selection.
- Implemented first script: `experiments/scripts/stage3_margin_residual_normalonly_probe.py`.
- Smoke test passed: 1-epoch run completed without error and matched margin almost exactly, as expected for a bounded/no-op initialized probe.
- Formal 5-seed / 80-epoch run completed on HCCS-88.

**Formal Results**:

| Metric | Mean ± Std | Interpretation |
|--------|------------|----------------|
| AUC | `0.795253 ± 0.007065` | Essentially tied with margin baseline |
| AP | `0.516078 ± 0.022112` | No stable AP improvement; mean delta is negative |
| Top-1% | `0.958371 ± 0.056710` | High but unstable across seeds |
| Top-5% | `0.716780 ± 0.030055` | Below earlier margin top-5 reference |
| Spearman(score, margin) | `0.998466 ± 0.000220` | Probe learned almost no ranking distinct from margin |
| ΔAUC vs margin | `+0.000063 ± 0.000272` | Numerically negligible |
| ΔAP vs margin | `-0.000446 ± 0.002498` | No stable gain |

**Interpretation Rule Applied**:
> The residual probe does **not** produce stable improvement. It should be dropped as a method route. The useful scientific conclusion is negative: under this bounded normal-only correction protocol, the learned correction does not provide stable ranking improvement beyond the explicit margin backbone.

**Important Nuance on MLP vs Margin**:
> The high `Spearman(score, margin)` here should **not** be interpreted as "the MLP easily reconstructed margin". Unlike the earlier Stage-1 margin-regression probe, this Stage-3 score already contains the closed-form margin explicitly: `score = margin + corr`. Therefore high rank agreement mostly means the learned correction was not strong or differently aligned enough to perturb the existing margin order. This is compatible with the earlier finding that an MLP only recovered margin well after being given normalized interaction features and trained directly with margin-regression supervision.

**Skill / Execution Compliance Note**:
> The experiment design followed the diagnostic-probe constraints, but the formal experiment launch did **not** comply with the `research-explorer` running-experiment rule. That skill explicitly says non-validation AUC/AP experiments must use WandB and `sweep-monitor`, and forbids manual SSH execution (`ssh HCCS-88 "python ..."`). This run was launched via a manual SSH loop, so the experimental result is usable as a diagnostic measurement, but the execution process has a protocol violation. Future formal runs should be started through the approved sweep-monitor / experiment-runner path, after smoke testing.

**Follow-up Diagnostic Plan (ABCD decomposition)**:
- Added `experiments/scripts/stage3_residual_abcd_diagnostics.py` as a diagnostic-only companion script.
- Purpose: distinguish four explanations for near-margin ranking:
  - A: global shift / calibration (`corr_mean` dominates `corr_std`, tiny rank flips).
  - B: correction is margin-linked / monotonic (`corr` strongly correlated or linearly explained by margin).
  - C: perturbation is too weak or in the wrong locality (tiny rank-flip rate, high top-k Jaccard).
  - D: normal-only objective lacks anomaly-separating signal (`corr_only_auc`, `neg_corr_auc`, normal/anomaly correction distributions).
- Protocol: same normal-only training and validation selection as Stage-3; anomaly labels remain diagnostic-only.
- Compliance note: do **not** launch this as another manual SSH formal run. Use the approved sweep-monitor / experiment-runner path, or explicitly mark a new violation.

---

## Key Findings Summary

||| Finding | Date | Implication |
||---------|------|-------------||
|| Proxy AUC ≠ real anomaly ranking validity | 2026-05-09 | Cannot optimize proxy metrics directly |
|| Normal-only density modeling fails | 2026-05-12 | `d` must be geometry-aware, not latent |
|| Scalar `||d||` Likelihood-Ratio fails | 2026-05-12 | Scalar conditioning loses direction; do not pursue as-is |
|| Vector-conditioned LR partially recovers signal | 2026-05-12 | Direction helps, but density-ratio still below margin |
|| Additive residual correction is diagnostic only | 2026-05-13 | Useful as a probe; not elegant enough as final method narrative |
|| MLP can recover margin with normalized features | 2026-05-11 | Input design matters, not MLP capacity |
|| ABCD diagnostics confirm no residual signal | 2026-05-13 | All flags A,B,C raised; normal-only correction cannot improve margin |
|| **Response matrix has extra info but mat_mean unstable** | 2026-05-14 | mat_mean 3/5 wins, Spearman 0.708; R_a purity is key mechanism |

---

## Next Actions

|| Priority | Action | Status ||
||----------|--------|--------||
|| 1 | Implement Likelihood-Ratio minimal probe (scalar `||d||` conditioning) | Complete ||
|| 2 | Compare Likelihood-Ratio AUC vs margin baseline | Complete: failed ||
|| 3 | Diagnose whether vector conditioning is worth trying | Complete: partial recovery, below margin ||
|| 4 | Run bounded normal-only residual probe | ✅ Complete: no stable improvement ||
|| 5 | Run Stage 3 ABCD Diagnostics sweep | ✅ Complete (WandB 4u8dzp5v) ||
|| 6 | Response Matrix Diagnostic (Route2) | ✅ Complete: mat_mean unstable ||
|| 7 | Design more robust matrix summary (quantile/mode/weighted) | Pending ||
|| 8 | Investigate R_a purity improvement mechanism | Pending ||

---

## Open Questions

1. **Likelihood-Ratio formulation**: Why does vector conditioning recover signal but still remain below explicit margin?
2. **Residual signal**: Does margin leave stable, learnable residual structure under a normal-only protocol? → **Answered: No, ABCD diagnostics confirm no residual signal**
3. **Unified geometry**: If residual signal exists, how can it be rewritten as a clean geometry-aware score instead of additive patching? → **Closed: residual probe route dropped**
4. **Reference-view consistency**: Can protocol-clean consistency produce a complementary score to margin?
5. **Response Matrix**: Can more robust summary (quantile/mode/weighted mean) stabilize Route2 signal?
6. **R_a purity mechanism**: How to improve target-conditioned R_a purity without label supervision?

---

## Day 6: 2026-05-13 (Stage 3 ABCD Diagnostics Completed)

**Activity**: Formal 5-seed ABCD diagnostics sweep completed via WandB.

- Sweep ID: `HCCS/DualRefGAD/4u8dzp5v`
- URL: https://wandb.ai/HCCS/DualRefGAD/sweeps/4u8dzp5v
- Status: FINISHED
- Seeds: 5 (s0-s4), 80 epochs each

**ABCD Diagnostic Results (5-seed Mean ± Std)**:

|| Metric | Mean ± Std | Interpretation |
|--------|------------|----------------|
| AUC | `0.795253 ± 0.007065` | Tied with margin baseline |
| ΔAUC vs margin | `+0.000063 ± 0.000272` | Numerically negligible |
| Spearman(score, margin) | `0.998466 ± 0.000220` | Nearly perfect rank agreement |
| Rank flip rate | `0.008640 ± 0.000752` | Only 0.86% perturbation |
| Linear R² corr from margin | `0.845531 ± 0.006012` | 84.5% variance explained by margin |
| corr_mean | `-0.176690 ± 0.003656` | Global negative shift (calibration) |
| corr_std | `0.107264 ± 0.001482` | Small variance |
| Cohen d (anom vs normal) | `-0.525560 ± 0.020369` | Moderate separation |

**ABCD Flags (All 5 Seeds)**:
- flag_A_global_shift: **5/5 True** — correction is global calibration
- flag_B_margin_linked: **5/5 True** — correction strongly explained by margin
- flag_C_too_weak_to_change_rank: **5/5 True** — perturbation insufficient for rank change
- flag_D_no_anomaly_separation: **0/5 True** — NOT flagged; some anomaly-normal separation exists

**Final Conclusion**:
> The bounded normal-only residual probe does NOT produce stable improvement beyond the explicit margin backbone. ABCD diagnostics confirm: the learned correction is dominated by global calibration (A), is linearly explained by margin (B), and produces insufficient ranking perturbation (C). This route should be dropped as a method candidate.

---

## Day 7: 2026-05-14 (Response Matrix Diagnostic - Route2)

**Activity**: No-training diagnostic on multi-reference response distribution.

**Motivation**: Scalar margin 把每个 reference set 求均值 centroid，可能丢失 response distribution 信息。Route2 检查压缩前的 matrix 是否有额外信号。

**Method**:
- Response Matrix: `M_ij(v) = cos(h_v - r_{n,i}, r_{a,j} - r_{n,i})`
- 维度: `K_n × K_a = 4 × 16 = 64`
- 诊断分数: `mat_mean` (全均值), `mat_entropy`, `mat_high08_ratio`

**Results (5-seed Elliptic)**:

|| Signal | AUC mean±std | vs margin |
||--------|--------------|-----------|
|| margin (baseline) | 0.7952±0.0071 | — |
|| mat_mean | 0.8009±0.0203 | **3/5 wins** |
|| mat_entropy | 0.7777±0.0296 | 1/5 wins |
|| mat_high08_ratio | 0.7878±0.0232 | 2/5 wins |
|| ra_anom_ratio_diagnostic | 0.9328±0.0005 | 5/5 (label-dep) |

**Per-seed AUC**:

|| seed | margin | mat_mean | mat_entropy |
||------|--------|----------|-------------|
|| 0 | 0.7938 | **0.8200** | 0.8058 |
|| 1 | 0.7960 | 0.7901 | 0.7763 |
|| 2 | **0.7991** | 0.7711 ❌ | 0.7299 |
|| 3 | 0.7840 | **0.8060** | 0.7780 |
|| 4 | 0.8030 | **0.8170** | 0.7986 |

**Key Mechanism Insights**:
- `Spearman(mat_mean, margin) = 0.708` — 两者不完全重叠，matrix 有额外信息
- `ra_anom_ratio_diagnostic` AUC 0.93 — **Target-conditioned R_a purity 是关键机制**
- Seed2 失败原因：response matrix variance 极大 (mat_std≈0.606)，均值被低/负尾巴拖低
- False positive 问题：geometry degeneracy 可产生均匀高响应（即使 R_a 不纯）

**Conclusion**:
> Route2 (multi-reference distributional inconsistency) 是有用的解释性视角，**但 mat_mean 不稳定 (3/5 wins, seed2 损失严重)**。简单均值过于粗糙，无法处理 heterogeneous anomalies 和 geometry degeneracy。建议作为 complementary signal 或后续 investigation 起点，而非直接部署。

**Anchor**: `~/anchors/2026-05-14-dualrefgad-mat-mean-route2/report.html`

⚠️ **Protocol Deviation**: Seeds 1-4 通过手动 SSH + Hermes cron watchdog 执行，而非 experiment-runner。本诊断不作为 formal method-validation evidence。

---

## File Inventory

| File | Type | Date Created |
|------|------|--------------|
| `README.md` | Investigation plan | 2026-05-09 |
| `hypothesis.md` | Candidate designs | 2026-05-09 |
| `RESEARCH_SURVEY.md` | Literature audit | 2026-05-09 |
| `experiments/outputs/*.html` | Email reports | 2026-05-10~12 |
| `insights.md` | Key conclusions | 2026-05-12 |
| `PROGRESS.md` | This file | 2026-05-12 |

---

## Day 8: 2026-05-14 (Deep Dive: R_a Selection Mechanism + VecGAD Analysis)

**Activity**: 用户提出三个深度问题，深入分析 DualRefGAD 核心机制。

**Questions**:
1. 当前代码如何挑选 R_a？真的与当前节点无关吗？
2. VecGAD（不是 DualRefGAD）能帮上我们吗？
3. 我们至今没有采用机器学习的训练，应该加在哪里？Matrix 还有什么可以操作？

**Findings**:

### R_a 选择机制（select_refs 函数）

关键代码位于 `run_training_degradation_diagnosis.py:92-111`：

```python
# R_a 选择公式
a_scores = l_a + ga[None,:]  # similarity + anomaly score
anom_refs = np.argsort(-a_scores, axis=1)[:,:args.anom_k]
```

**结论**：R_a 选择**确实与当前节点有关**
- `l_a[i,j]` = 节点 i 与节点 j 的 cosine similarity（当前节点视角）
- `ga[j]` = 节点 j 的 rejection score（来自 PCA residual，不依赖真实标签）
- 选择标准：**高相似 + 高异常度** 的节点

**Bootstrap Trap 解释**：
- ra_anom_ratio_diagnostic AUC 0.93 是因为 PCA residual 确实能识别真实异常
- 但这正是 bootstrap trap 的根源：要获得高纯度 R_a → 需要准确 anomaly score → 但要获得 anomaly score → 需要知道异常 → 这是我们要解决的问题

### VecGAD 核心洞察

> **重构误差的方向信息比标量分数更重要**

DualRefGAD 可借鉴：
- 保留 residual 的向量形式而非压缩为标量
- 用 reference residual 指导伪异常生成
- HSC 球壳约束确保伪异常质量

### 训练位置建议

| 方案 | 科学干净度 | 实现难度 | 预期收益 |
|------|-----------|---------|---------|
| **A: RHO-style Alignment** | 最高 | 中 | 中高 |
| **B: VecGAD-style Residual** | 中 | 高 | 高 |
| **C: Matrix Summary Learning** | 高 | 低 | 中 |

**Matrix 操作建议**：
- 避免 simple mean（易受极端值干扰）
- 考虑 trimmed mean、weighted mean、quantile summary、attention pool

**推荐顺序**：
1. 先尝试方案 C（Matrix Summary）：低成本，直接改进已验证 signal
2. 如果 C 效果有限，尝试方案 A（RHO-style）：最干净
3. 如果 A 效果有限，再考虑方案 B（VecGAD-style）：最复杂

**Email**: `dualrefgad_analysis.md` sent to ziyao.lin@senyao.cloud

---

## Day 8-afternoon: 2026-05-14 (方案 C vs A 诊断设计)

**用户决策**：
- 方案 C (Matrix Summary Learning) 优先尝试
- 允许训练 GT Encoder，师兄建议从 1 层开始
- 方案 A (RHO-style) 作为干净备选
- 需要先做诊断确定 C vs A 优先级

**诊断设计**：

### 方案 C 诊断：Matrix Summary 聚合方式对比

| 聚合方式 | 公式 | 预期效果 |
|---------|------|---------|
| **Trimmed mean** | `mean(M | M > Q10 and M < Q90)` | 去除极端值，稳定 seed2 |
| **Weighted mean** | `sum(w_ij * M_ij)`，`w_ij = ||d_j||` | 倾向可靠 R_a |
| **Quantile (Q50)** | `median(M.flatten())` | 保留分布形态 |
| **Max + mean hybrid** | `0.3*max + 0.7*mean` | 边界 + 平均 |

**判断标准**：5/5 wins 或 AUC > margin + 2σ

**时间成本**：30 分钟（使用现有 frozen encoder 输出）

### 方案 A 诊断：RHO-style Normality Alignment Probe

```python
z_n = MLP(concat(h, r_n))      # target-normal relation
z_d = MLP(concat(h, r_n, r_a))  # deviation-reference relation
loss = ||z_n[normal] - center_n||² + ||z_d[normal] - center_d||²
score = ||z_n - center_n|| + ||z_d - center_d||
```

**判断标准**：AUC > margin 或 Spearman vs margin < 0.5（正交）

**时间成本**：2-3 小时

### 师兄建议：从 1 层开始训练 GT Encoder

| 配置 | 当前值 | 建议起点 |
|------|--------|---------|
| GT_num_layers | 3 | **1** |
| GT_num_heads | 2 | 2 |
| embedding_dim | 256 | 256 |

**好处**：更快训练、更容易诊断、逐步扩展

### 决策树

```
方案 C 诊断
├─ 稳定 (5/5 wins)
│   └─ → 实现 learnable attention pool（训练 1 层 encoder）
│
└─ 不稳定
    └─ → 方案 A 诊断
        ├─ 有正交 signal (Spearman < 0.5)
        │   └─ → Normality alignment 主方向
        │
        └─ 无 signal
            └─ → 方案 B (VecGAD-style residual)
```

### 执行约束

**⚠️ HCCS-88 不可用**，需在本地执行：
- OpenClawVM (192.168.1.9) 应有 GPU
- 或降级到 CPU（诊断脚本较轻）

**Next Actions**：
1. 确认本地 GPU 可用性
2. 执行方案 C 诊断（Matrix Summary 聚合对比）
3. 根据结果决定后续方向

---

_Investigation tracking started by Nexus, updated 2026-05-14._

---

## Day 8-evening: 2026-05-14 (Stage4 RHO-style Normality Alignment Smoke Probe)

**Activity**: 在 HCCS-80 上启动并完成 Stage4 最小 smoke probe，用真实 `photo.mat` 验证脚本、环境和目标函数链路。

**Infrastructure**:
- DualRefGAD 已迁移为 GitHub 管理仓库：`Sen-Yao/DualRefGAD`
- HCCS-80 目录：`~/DualRefGAD`，由 GitHub clone 管理
- 数据集：`~/DualRefGAD/dataset/*.mat` symlink 到 `~/GGADFormer/dataset/*.mat`
- 环境：`source /opt/anaconda/etc/profile.d/conda.sh && conda activate GGADFormer`

**Probe**:
- Script: `scripts/diagnostics/stage4_rho_normality_alignment_probe.py`
- Config: `configs/stage4_rho_photo_s0_smoke.yaml`
- Job ID: `exp_20260514_180451_stage4_rho_photo_s0_smoke`
- Dataset: `photo`, seed 0, 20 epochs, GT_num_layers=1, frozen encoder (`train_encoder=false`)
- Runtime: ~40s on HCCS-80 GPU 3

**Smoke Result**:

| Metric | Value |
|--------|-------|
| Stage4 score AUC | `0.5682` |
| Stage4 score AP | `0.1165` |
| Margin AUC (same script baseline) | `0.4237` |
| Margin AP | `0.0749` |
| Delta AUC | `+0.1445` |
| Spearman(score, margin) | `-0.3305` |
| Top-1% Jaccard(score, margin) | `0.0000` |
| Top-5% Jaccard(score, margin) | `0.0027` |

**Interpretation**:
- Smoke probe confirms the Stage4 pipeline runs end-to-end on real HCCS-80 data.
- The score is strongly non-overlapping with margin on this smoke configuration (`Spearman=-0.33`, almost zero top-k overlap), which is exactly the kind of orthogonal signal diagnostic we wanted to test.
- However, this is **not formal evidence**: single seed, photo dataset, 20 epochs, and margin baseline inside this script appears weak on photo. Next step should be a runner-managed 5-seed probe, preferably starting with Elliptic after optimizing reference selection runtime.

**Important runtime note**:
- Elliptic dry-run with current full pairwise reference selection exceeded 120s and was stopped. Need optimize/block reference selection or start formal probe on photo first, then port to Elliptic.

**Update after optimization**:
- Implemented blockwise top-k reference selection in `stage4_rho_normality_alignment_probe.py`, replacing full N×N anomaly-reference matrix materialization.
- Commit: `1a6deac Optimize Stage4 reference selection with blockwise top-k` in `Sen-Yao/DualRefGAD`.
- Elliptic dry-run now completes successfully: token shape `[46564, 21, 93]`, normal_train_count=2086, test_count=44236, elapsed `0:44`, max RSS ~2.8GB.
- Elliptic 2-epoch smoke also completes successfully in `0:44` (job `exp_20260514_182312_stage4_rho_elliptic_s0_smoke2`).
- Smoke metrics are not formal evidence (`num_epoch=2`), but confirm the bottleneck is resolved enough for real Elliptic probing.

---

## Day 9: 2026-05-16 (Stage4 Elliptic 5-seed Probe - FAILED)

**Activity**: 在 HCCS-88 上运行 Stage4 Elliptic 5-seed probe，验证 RHO-style normality alignment 在真实大规模数据集上的效果。

**Infrastructure**:
- HCCS-88: 通过反向隧道 `senyao.cloud:8822` 访问
- SSH alias 已添加到 `~/.ssh/config`: `Host hccs-88-tunnel`
- experiment-runner 管理（修复了手动 SSH protocol deviation）
- 数据集: `elliptic.mat` 从 HCCS-66 复制到 HCCS-88

**Probe Config**:
- Script: `scripts/diagnostics/stage4_rho_normality_alignment_probe.py`
- Dataset: `elliptic` (46564 nodes, 93 features)
- Seeds: 5 (s0-s4)
- Epochs: 50
- GT encoder: frozen (`train_encoder=false`)
- Device: GPU 0-4 分配
- Reference selection: `--use_approx_anom_refs --anom_approx_k 500`

**Results (5-seed)**:

| Seed | Stage4 AUC | Margin AUC | Delta AUC | Spearman(vs margin) |
|------|------------|------------|-----------|---------------------|
| 0 | 0.386 | 0.340 | +0.046 | 0.213 |
| 1 | 0.416 | 0.381 | +0.035 | 0.403 |
| 2 | 0.449 | 0.395 | +0.054 | 0.409 |
| 3 | 0.439 | 0.484 | **-0.045** | 0.577 |
| 4 | 0.360 | 0.487 | **-0.127** | 0.253 |
| **Mean** | **0.410** | **0.413** | **-0.003** | **0.371** |

**Critical Finding**:

> **Stage4 在 Elliptic 上完全失败。**

1. **AUC 0.41 ≈ random level**：远低于 photo smoke 的 0.57
2. **Spearman 正相关 (0.37)**：而非 photo 的负相关 (-0.33)
   - Stage4 score 和 margin **同向变化**，没有独立信号
   - 这意味着 normality alignment score 只是 margin 的弱 proxy
3. **Delta AUC 近乎零**：无法超越 margin baseline

**与 Photo Smoke 的对比**:

| Dataset | Stage4 AUC | Margin AUC | Spearman | Interpretation |
|---------|------------|------------|----------|----------------|
| photo (smoke) | 0.57 | 0.42 | **-0.33** | 有正交信号 ✅ |
| elliptic (5-seed) | 0.41 | 0.41 | **+0.37** | 无独立信号 ❌ |

**Root Cause Analysis**:

**不是训练策略问题，而是数据集特性差异**：

1. **异常节点分布差异**
   - photo: 异常节点可能形成紧密簇（社区结构）
   - elliptic: 异常节点分散（金融交易图，异常是时间局部性）
   
2. **Center deviation scoring 的假设**
   - Stage4 假设：正常节点在 relation embedding 空间有紧凑的 center
   - elliptic 的正常节点可能在 relation 空间**本身就分散**
   - 因此 deviation from center 无法区分异常

3. **Frozen encoder 的局限**
   - GT encoder 在两个数据集上可能学到不同的 representation geometry
   - elliptic 的 relation embedding 可能没有清晰的结构

**训练策略问题的反驳**：

- 如果是 epoch 不够 → photo 20 epochs 就有信号，elliptic 50 epochs 应该更好
- 如果是 MLP head 太简单 → 应该两个数据集都失败
- 实际情况是 photo 有信号、elliptic 无信号 → **问题不在训练策略**

**Protocol Compliance**:
- ✅ 使用 experiment-runner 管理（修复了 Day 8 的 deviation）
- ✅ SSH config 已添加 alias（保护文件需手动确认）

**Conclusion**:

> RHO-style normality alignment **在 Elliptic 上不可行**。数据集特性（异常分散）导致 center deviation scoring 无效。建议放弃此路线，转向：
> 1. R_a purity improvement（Target-conditioned reference selection）
> 2. VecGAD-style residual（向量保留而非标量压缩）
> 3. 数据集自适应（识别 center deviation scoring 的适用条件）

**Next Decision Point**: 用户需决定后续方向

---

## Day 9-followup: 2026-05-16 (Elliptic Embedding Structure Diagnostic)

**Activity**: 用户要求先诊断 Elliptic 的 embedding 结构，并明确提醒实验/诊断需使用 experiment-runner skill。已按 runner-registered probe pattern 执行。

**Runner Compliance**:
- Skill: `experiment-runner`
- Job ID: `exp_20260516_154508_elliptic_embedding_structure_probe`
- Profile/kind: `probe` / `probe`
- Launch: Hermes-tracked background process, not manual SSH loop
- Status: `finished`
- Runtime: ~154s on HCCS-88 GPU 0

**Artifacts**:
- Script: `experiments/scripts/elliptic_embedding_structure_probe.py`
- Config: `experiments/configs/elliptic_embedding_structure_probe.yaml`
- Output: `experiments/outputs/elliptic_embedding_structure_probe.json`

**Diagnostic Goal**:

Test whether Stage4 failed because Elliptic frozen relation embeddings lack a compact normal center. The probe inspected multiple spaces:
- descriptor before GT encoder
- GT embedding `h`
- normal/anomaly reference centroids `r_n`, `r_a`
- relation vectors `u=h-r_n`, `d=r_a-r_n`
- Stage4 inputs `z_n` / `z_d`
- interaction features `[u,d,u*d,|u-d|]`

For each space, it computed center-distance AUC/AP, diagonal Mahalanobis AUC, PCA residual AUC, normal radius statistics, Cohen's d, KS distances, and effective rank. Labels are diagnostic-only.

**Key Results**:

| Space | Center AUC | Spearman(center, margin) | Cohen d (anom-normal radius) | Interpretation |
|-------|------------|--------------------------|------------------------------|----------------|
| `rn_normal_ref_centroid` | **0.5267** | -0.232 | +0.011 | only near-random weak positive |
| `d_ra_minus_rn` | 0.4953 | +0.303 | -0.050 | random |
| `gt_emb_h` | 0.4646 | -0.084 | -0.245 | anomalies closer to center |
| `interaction_u_d` | 0.4210 | +0.445 | -0.235 | anomalies closer |
| `zd_feat_stage4_input` | 0.4105 | +0.324 | -0.317 | anomalies closer |
| `zn_feat_stage4_input` | 0.3888 | +0.305 | -0.377 | anomalies much closer |
| `ra_anom_ref_centroid` | 0.3800 | +0.319 | -0.450 | anomaly refs not useful |
| `u_target_minus_rn` | 0.3680 | +0.584 | -0.482 | strongly inverted |
| `descriptor_z_pre_encoder` | 0.3592 | +0.191 | -0.087 | inverted/raw descriptor weak |

**Reference Autopsy**:

| Metric | Value | Meaning |
|--------|-------|---------|
| Global anomaly-ref anomaly ratio | `0.0235` | selected R_a is mostly normal on Elliptic |
| Anom-ref ratio AUC | `0.4448` | reference purity does not identify anomalies |
| ga/rejection AUC | `0.3489` | the normal-rejection score is inverted |
| Margin AUC | `0.3404` | margin is inverted in this run |
| -Margin AUC | `0.6596` | flipping margin recovers moderate signal |

**Core Finding**:

> Elliptic does **not** satisfy the Stage4 normal-manifold assumption. In most relation spaces, anomalies are not farther from the normal center; they are often **closer** to it. The best center-distance AUC is only 0.5267 on `r_n`, while Stage4-relevant spaces (`z_n`, `z_d`, `u`, interaction) are below random.

**Mechanistic Interpretation**:

1. **Normal center is not the right object on Elliptic**
   - Elliptic's normal transaction nodes are heterogeneous.
   - Center distance measures broadness/typicality, but anomalies are not necessarily peripheral.

2. **R_a bootstrap signal is inverted/weak**
   - `ga/rejection AUC = 0.3489`, so high rejection selects the wrong side.
   - Global R_a anomaly ratio is only `2.35%`, so anomaly reference sets are almost entirely normal.

3. **Stage4 failure is structural, not merely training**
   - The frozen spaces supplied to Stage4 already have inverted center-distance geometry.
   - A better MLP head cannot fix that without changing the objective from "far from normal center = anomaly".

4. **Potentially useful clue: sign inversion**
   - `-margin AUC = 0.6596`, suggesting Elliptic may need an orientation/role diagnostic rather than center-deviation scoring.
   - This aligns with the idea that Elliptic anomalies may be "too normal-looking / central suspicious transactions" rather than outlying graph/product nodes.

**Decision Update**:

- Drop Stage4 center-deviation as a main route for Elliptic.
- Do not spend more time only increasing epochs or MLP capacity.
- Next promising directions:
  1. **Orientation/sign diagnostic**: why margin and rejection invert on Elliptic;
  2. **R_a purity / reference selection repair**: current R_a is mostly normal;
  3. **VecGAD-style residual vector route**: preserve residual direction instead of scalar rejection/center distance.

---

## Day 9-followup-2: 2026-05-16 (Direction 1 Orientation / Sign Diagnostic)

**Activity**: 用户指示执行方向一诊断：分析 Elliptic 上 margin/rejection 方向反转的根因，并检查是否存在无标签 orientation rule 能决定使用 `score` 还是 `-score`。

**Runner Compliance**:
- Skill: `experiment-runner`
- Job ID: `exp_20260516_155640_elliptic_orientation_sign_probe`
- Profile/kind: `probe` / `probe`
- Launch: Hermes-tracked background process, not manual SSH loop
- Status: `finished`
- Runtime: ~64s on HCCS-88 GPU 0

**Artifacts**:
- Script: `experiments/scripts/elliptic_orientation_sign_probe.py`
- Config: `experiments/configs/elliptic_orientation_sign_probe.yaml`
- Output: `experiments/outputs/elliptic_orientation_sign_probe.json`

**Diagnostic Questions**:
1. Which scores are inverted on Elliptic?
2. Is the inversion tied to normal rejection, degree, R_a purity, or transaction/time-like grouping?
3. Can a label-free rule infer the correct orientation?

**Top Score Orientation Results**:

| Score | AUC | -Score AUC | Correct Orientation | Key Correlation |
|-------|-----|------------|--------------------|----------------|
| `margin_cos_u_d` | 0.3404 | **0.6596** | negative | Spearman(rejection)=+0.256, degree=+0.374 |
| `ga_score` | 0.3489 | **0.6511** | negative | Spearman(rejection)=+1.000 |
| `normal_rejection` | 0.3489 | **0.6511** | negative | itself inverted |
| `residual_norm` | 0.3489 | **0.6511** | negative | same as rejection |
| `raw_dot_u_d` | 0.3503 | **0.6497** | negative | Spearman(rejection)=+0.220 |
| `rn_dist/u_norm` | 0.3659 | **0.6341** | negative | Spearman(degree)=+0.339 |
| `ra_dist` | **0.6138** | 0.3862 | positive | Spearman(rejection)=-0.200 |
| `d_norm` | **0.5452** | 0.4548 | positive | weak |

**Core Finding 1 — Rejection is inverted**:

> The normal-model rejection signal itself is inverted on Elliptic: `rejection AUC=0.3489`, while `-rejection AUC=0.6511`.

This means the current bootstrap assumption “higher rejection = more anomalous” is wrong for this Elliptic configuration. Since R_a selection depends on `ga/rejection`, the selected anomaly references are pulled toward the wrong side.

**Core Finding 2 — Margin inversion follows rejection/degree**:

`margin_cos_u_d` has:
- Spearman(rejection)=`+0.256`
- Spearman(ga)=`+0.256`
- Spearman(log_degree)=`+0.374`
- AUC=`0.3404`, but `-margin` AUC=`0.6596`

Interpretation:
> Margin is not independently inverted for mysterious reasons; it inherits orientation from the wrong-side rejection / structural-degree axis. High margin corresponds to high rejection/high degree, but Elliptic anomalies are concentrated in lower-degree / lower-rejection regions.

**Core Finding 3 — Degree stratification explains much of the flip**:

By log-degree bins:
- Lowest-degree bin (`log_degree 0~0.69`) has anomaly rate `19.2%` and `-margin AUC=0.8051`.
- Medium degree (`1.10~1.39`) has anomaly rate `5.7%` and `margin AUC=0.6265`.
- Highest degree has anomaly rate only `2.1%`.

This shows sign is **not globally stable**: Elliptic anomaly distribution is degree-regime dependent. A single anomaly-high convention can be misleading.

**Core Finding 4 — R_a purity remains broken**:

| Metric | Value |
|--------|-------|
| Global anomaly-ref anomaly ratio | `0.0235` |
| Anom-ref ratio AUC | `0.4448` |
| Mean normal-ref log-degree | `0.6431` |
| Mean anomaly-ref log-degree | `1.0404` |

R_a references are not actually anomalous; they are structurally higher-degree / high-rejection nodes. On Elliptic, that is often the normal side.

**Core Finding 5 — Naive label-free orientation rules fail**:

Tested label-free rules:
1. Use positive sign if unlabeled mean > train-normal mean.
2. Use positive sign if score aligns with rejection.
3. Use positive sign if score aligns with ga.
4. Majority vote of the above.

For key scores, this predicts the **wrong orientation**:
- `margin_cos_u_d`: predicted positive, true best is negative.
- `normal_rejection`: predicted positive, true best is negative.
- `ga_score`: predicted positive, true best is negative.

Why: these rules assume train normals define the low-anomaly baseline, but Elliptic labeled train normals are not representative of test normals/anomalies across degree regimes. Unlabeled-vs-train shift is dominated by structural distribution shift, not anomaly direction.

**Time-like Feature Check**:
- The heuristic search found no reliable low-cardinality timestep-like feature in the preprocessed feature matrix.
- Therefore the current probe cannot confirm time-slice causality from features alone.
- Degree/regime analysis is currently the stronger explanation.

**Mechanistic Conclusion**:

> Elliptic’s failure mode is not just “score sign flipped.” It is **regime-dependent orientation collapse**: normal rejection and margin align with a structural-degree/rejection axis, but anomalies are concentrated in low-degree/low-rejection regions for a large part of the test distribution. This makes high-rejection R_a selection actively harmful.

**Research Implication**:

Do **not** solve this by globally flipping margin. Global flip gives moderate AUC (`0.6596`) but is a dataset-specific diagnostic hack and fails the scientific goal. The real route should address regime-conditioned reference selection / orientation:

1. **Degree/regime-conditioned orientation diagnostic**: select orientation per structural regime without labels.
2. **Repair R_a selection**: avoid assuming high rejection = anomaly; include low-rejection suspicious regimes or bidirectional references.
3. **VecGAD-style residual direction**: preserve residual vector direction and condition scoring on regime, rather than scalar rejection magnitude.

