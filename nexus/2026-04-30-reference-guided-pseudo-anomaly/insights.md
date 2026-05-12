# Insights

## 2026-04-30 开题洞见

Dual-reference tokenization 的价值不应停在 reference selection。真正的问题是如何让训练 objective 学会利用 reference geometry。

前序实验显示：

- `R_a(v)` 在异常目标节点上有较高异常命中率；
- `shuffled_ra` 降低性能，说明 target-reference 对应关系重要；
- self-residual objective 的 best epoch 多为 0；
- global ref-guided 在 Photo 上强正向，在 Elliptic 上负向。

因此，新方向应聚焦：**target-specific reference residual pseudo anomaly**。

## 设计原则

1. 保持单一 BCE objective；
2. pseudo anomaly direction 来自 `R_a(i)-R_n(i)`；
3. 不引入复杂多损失；
4. 用 mini 验证机制，再做 5-seed。

## 2026-04-30 Phase 1 manual-run exploratory results

**Run type**: manual exploratory run, retained by user approval. These results are useful for mechanism diagnosis, but formal sweeps/training must follow the skill workflow.

| Dataset | Objective | Test AUC | Test AP | Best Epoch | Interpretation |
|---|---:|---:|---:|---:|---|
| Photo | self_residual | 0.5799 | 0.1209 | 0 | baseline; training does not amplify geometry |
| Photo | global_ref_guided | 0.7646 | 0.3967 | 200 | reference-guided synthesis is strongly positive |
| Photo | target_ref_guided | **0.7649** | 0.3585 | 70 | matches global AUC, converges earlier |
| Elliptic | self_residual | 0.6900 | 0.1518 | 0 | baseline geometry is already useful |
| Elliptic | global_ref_guided | 0.5360 | 0.0984 | 160 | global direction is harmful |
| Elliptic | target_ref_guided | **0.7197** | **0.1853** | 0 | fixes global-direction failure, but training still does not amplify after epoch 0 |

### Interpretation

Target-specific reference residual is currently the strongest and cleanest pseudo anomaly direction among tested variants. It preserves the Photo gain and repairs the Elliptic degradation caused by a global reference direction. However, Elliptic still selects epoch 0, suggesting that target-specific direction improves initial geometry but the optimization dynamics still need diagnosis.

### VecGAD-style reconstruction residual note

The current tokenization-theory script does not implement VecGADs decoder-based reconstruction error path. Therefore, a true ablation is not currently available. Possible next steps are:

## 2026-04-30 Phase 1 manual-run exploratory results

**Run type**: manual exploratory run, retained by user approval. These results are useful for mechanism diagnosis, but formal sweeps/training must follow the skill workflow.

| Dataset | Objective | Test AUC | Test AP | Best Epoch | Interpretation |
|---|---:|---:|---:|---:|---|
| Photo | self_residual | 0.5799 | 0.1209 | 0 | baseline; training does not amplify geometry |
| Photo | global_ref_guided | 0.7646 | 0.3967 | 200 | reference-guided synthesis is strongly positive |
| Photo | target_ref_guided | **0.7649** | 0.3585 | 70 | matches global AUC, converges earlier |
| Elliptic | self_residual | 0.6900 | 0.1518 | 0 | baseline geometry is already useful |
| Elliptic | global_ref_guided | 0.5360 | 0.0984 | 160 | global direction is harmful |
| Elliptic | target_ref_guided | **0.7197** | **0.1853** | 0 | fixes global-direction failure, but training still does not amplify after epoch 0 |

### Interpretation

Target-specific reference residual is currently the strongest and cleanest pseudo anomaly direction among tested variants. It preserves the Photo gain and repairs the Elliptic degradation caused by a global reference direction. However, Elliptic still selects epoch 0, suggesting that target-specific direction improves initial geometry but the optimization dynamics still need diagnosis.

### VecGAD-style reconstruction residual note

The current tokenization-theory script does not implement VecGAD's decoder-based reconstruction error path. Therefore, a true VecGAD reconstruction-residual ablation is not currently available. Possible next steps are:

1. **No-op / unavailable control**: explicitly record that true VecGAD residual is unavailable in the current architecture.
2. **Proxy residual**: use an existing non-learned residual such as PCA residual or reference residual as a proxy, but this is not a VecGAD reconstruction-error ablation.
3. **Minimal decoder addition**: add a token decoder and reconstruction projection only to produce residual direction, while keeping the objective as single BCE. This would be a new architectural component and must be treated as a separate controlled ablation.

## 2026-04-30 Photo formal sweep via sweep-monitor skill

**Sweep ID**: `sp1r1v6i`  
**WandB**: https://wandb.ai/HCCS/VoxG/sweeps/sp1r1v6i  
**Run type**: formal grid sweep launched via `sweep-monitor` skill  
**Dataset**: Photo  
**Objective**: `target_ref_guided`  
**Grid**: `pseudo_beta ∈ {0.1, 0.2, 0.3}`, `seed ∈ {0,1,2,3,4}`  
**Total runs**: 15/15 completed

| pseudo_beta | AUC mean±std | AP mean±std | Interpretation |
|---:|---:|---:|---|
| 0.1 | 0.6395±0.0822 | **0.2192±0.0775** | best AP, unstable AUC |
| 0.2 | 0.6215±0.0481 | 0.2007±0.0716 | weaker |
| 0.3 | **0.6474±0.0350** | 0.1752±0.0437 | best AUC, lowest variance |

### Interpretation

The formal 5-seed Photo sweep does not reproduce the strong seed=0 exploratory result (`AUC≈0.7649`) as a stable mean. `target_ref_guided` remains mechanistically promising, but the Photo result is seed-sensitive. The clean formal conclusion is therefore:

- single-seed target-specific reference residual can strongly amplify Photo geometry;
- 5-seed stability is not yet sufficient;
- `pseudo_beta=0.3` gives the best mean AUC and lowest AUC variance among tested values;
- `pseudo_beta=0.1` gives the best mean AP.

### Monitoring issue note

The sweep itself was launched through the `sweep-monitor` skill, but the monitor cron did not report automatically because of timezone/timeout issues in the current skill implementation. Result grouping was also manually verified because the default monitor grouping does not include `pseudo_beta` / `objective_mode`.
