# Retrieval Reference Tokenization 探究

## 背景

本探究围绕 Dual-Sequence Reference Tokenization 展开：为每个节点构造 self token、normal-reference sequence 与 anomaly-reference sequence，并将它们送入现有 VecGAD 的 TransformerEncoder，以验证全图 reference tokenization 是否能为 GAD 提供更有效的 token 语义。

当前实现遵循最小侵入原则：不直接修改主 `run.py` / `VecGAD.py` 训练逻辑，而是通过独立 runner 与 probe 脚本验证 token 构造、reference 选择与异常指标质量。

## 当前阶段结论

1. **最小闭环已跑通**：`run_dual_sequence_reference.py` 可在 Photo 上构造 `[N, 21, d]` token 序列并送入 `VecGAD.TransformerEncoder`，输出 embedding 维度正常。
2. **normal-reference 已基本站住**：Photo 上 normal-reference 平均纯度约 0.806；四分数审计中 `G_n` 与 `L_n` 分别达到 1.0000 与 0.8931 的正常纯度。
3. **anomaly-reference 是核心瓶颈**：Photo 上 anomaly-reference 平均纯度仅约 0.054；`G_a` / `L_a` 单独纯度也较弱。
4. **盲目丰富局部异常表示 `h_v` 收益有限**：`[q,c]` 扩展到 `[q,c,NDC,a]`，以及进一步加入 patch mismatch / inconsistency 后，未显著提升 anomaly-reference purity。
5. **VecGAD-style normal failure 有价值**：normal-space failure 与现有 anomaly affinity 组合，在 Photo 的 AUC/AP 和中等 top-k purity 上明显提升；Elliptic 上 normal center deviation AUC 很高但 top-k 不稳定。
6. **显式复杂加权组合不适合作为主方法**：三源组合 `anomaly affinity + normal density failure + structure context failure` 有实验价值，但过于启发式且引入权重超参数。
7. **当前最重要的新原则**：诊断实验显示 `q*c` 候选池内存在高纯异常候选，normal-deviation 能在候选池内显著重排。因此下一步应抽象为更优雅的单阶段原则：**Support as eligibility, deviation as ranking**。

## 关键文件

### 脚本

- `experiments/scripts/probe_ga_candidates.py`：测试 VecGAD-inspired `G_a` 候选。
- `experiments/scripts/probe_priority_ga.py`：测试 Normal Density Failure / Structure-Attribute Disagreement / LOF-style 指标。
- `experiments/scripts/probe_rank_twofactor_ga.py`：测试无权重 `Normal-Deviation × Anomaly-Support`。
- `experiments/scripts/probe_support_first_rerank.py`：诊断性 support-first rerank 实验。
- `run_dual_sequence_reference.py`：独立 Dual-Sequence Reference Tokenization runner（位于 VoxG 根目录）。

### 输出

- `experiments/outputs/photo_ga_candidates.json`
- `experiments/outputs/elliptic_ga_candidates.json`
- `experiments/outputs/photo_priority_ga.json`
- `experiments/outputs/elliptic_priority_ga.json`
- `experiments/outputs/photo_rank_twofactor_ga.json`
- `experiments/outputs/elliptic_rank_twofactor_ga.json`
- `experiments/outputs/photo_support_first_rerank.json`
- `experiments/outputs/elliptic_support_first_rerank.json`

## 下一步

不要把显式双阶段 top-M rerank 作为最终方法。应将它作为诊断结果，反推一个无显式 candidate-size 超参数的单阶段 score，使其自然表达：异常支持决定资格，正常偏离决定排序。
