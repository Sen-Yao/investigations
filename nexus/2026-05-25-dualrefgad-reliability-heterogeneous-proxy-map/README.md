# DualRefGAD Reliability / Heterogeneous Proxy Map

> 创建时间：2026-05-25 20:07 CST  
> 状态：🟡 活跃 / 新探究  
> 上游探究：`2026-05-21-dualrefgad-constraint-calibrated-reference-relation`  
> 第一阶段实验：`cleg3_oracle_to_proxy_candidate_signal_map` runner-registered pure probe

## 1. 中心问题

上游 C-LEG3 strict reproduction audit 已确认：在固定 `old_exact_080_regime` 后，`mat_mean` 的定义没有漂移，严格复现下 `mat_mean` AUC 为 `0.8009 ± 0.0182`，`margin` AUC 为 `0.7952 ± 0.0064`。但 Step-2 / LSD autopsy 同时暴露了一个关键矛盾：

> `mat_mean` 能清理大量 margin-only normal false positives，但也会丢失一部分 anomaly-reference 支持较高、只是矩阵支持异质的 true anomalies。

因此，本探究不直接启动 trainable head，而先问一个更窄的问题：

> 能否在不使用训练的 pure probe 中，把 oracle boundary categories 映射到无标签 proxy，并同时测试 row/column reliability 与 heterogeneous-support handling 的无训练 readout？

## 2. 研究边界

本探究是 **oracle-to-proxy 映射探究**，不是新方法训练探究。

- 不训练 learned reliability gate、Transformer head、Matrix AE 或 pseudo-anomaly scorer。
- anomaly labels 只用于 oracle category 定义、AUC/AP/autopsy 和候选 proxy 诊断，不用于训练、早停、阈值选择或最终方法声明。
- 固定 C-LEG3 / `old_exact_080_regime`、`normal_k=4`、`anom_k=16`，优先继承 strict reproduction 的 split discipline。
- 保存 split fingerprints 与公式 invariants；若存在 global RNG/threaded drift 风险，优先顺序执行 seeds。
- 输出必须能回答“哪些 proxy 值得进入 shallow reliability gate”，而不是只给一个 AUC leaderboard。

## 3. 上游证据约束

| 证据 | 结论 | 对本探究的约束 |
|---|---|---|
| Step-1 decomposition | `mat_mean` 曾达 `0.8168 ± 0.0054`，强于 `margin` | `mat_mean` 是必须击败或解释的强基线 |
| Step-2 failure autopsy | removed FP 低 anomaly-reference purity、高 dispersion；lost anomalies 高异质支持 | 需要显式建模 reliability 与 heterogeneous support |
| LSD oracle autopsy | aggregate AUC gap 变小，但 discordant categories 仍有机制信号 | 不能只看平均 AUC，需要 oracle-to-proxy category 映射 |
| Strict reproduction audit | `mat_mean = response_matrix.mean(axis=(1,2))` 精确成立，split/RNG discipline 是关键 | 新 probe 必须保存 split fingerprint 与公式检查 |

## 4. 第一阶段 probe 问题

`cleg3_oracle_to_proxy_candidate_signal_map` 要回答：

1. 哪些无标签 proxy 能区分四类边界节点：`rescued_anomalies`、`removed_false_positives`、`lost_anomalies`、`introduced_false_positives`？
2. row reliability / column reliability 是否能解释 `mat_mean` 的 false-positive cleanup？
3. heterogeneous-support readout 是否能减少 lost true anomalies，而不把 removed false positives 放回来？
4. 候选 readout 是否只是 margin、degree、rejection 或 reference-density shortcut？
5. 是否存在一个足够短、稳定、可解释的无训练 readout，值得进入下一步 shallow reliability gate？

## 5. 候选信号族

### 5.1 Row / Column Reliability

核心对象来自 response matrix `M(v)`：

- row mean distribution：每个 normal reference 对所有 anomaly references 的平均支持；
- column mean distribution：每个 anomaly reference 被所有 normal references 支持的程度；
- reliability 可用低 dispersion、高一致性、低 single-row/single-column dominance 来近似。

候选无训练 readout：

- trimmed/quantile mean：降低极端 entry 对平均值的支配；
- row/column consensus score：奖励多行多列共同支持；
- reliability-weighted mean：用 row/column 稳定性对 `M` 加权，但权重必须由无标签矩阵统计生成。

### 5.2 Heterogeneous-support Handling

目标不是简单惩罚异质性，因为 lost anomalies 正是“支持异质但可能真实”的节点。probe 要区分两种异质性：

- bad heterogeneity：removed false positives，通常 anomaly-reference diagnostic ratio 低、matrix dispersion 高；
- useful heterogeneity：lost anomalies，可能 anomaly-reference diagnostic ratio 较高但支持集中在部分 rows/cols。

候选无训练 readout：

- mixture-style support：允许一部分 high-support row/column 形成有效证据，而不是要求全矩阵均匀高；
- top-row / top-column robust pooling：看强支持是否跨多个 reference，而非单点峰值；
- consensus-minus-fragmentation：保留局部一致支持，惩罚孤立 spike。

## 6. Continuation gate

进入下一步 shallow reliability gate 的最低条件：

1. 至少一个无训练 readout 在 5 seeds 上接近或超过 strict `mat_mean`，或在 AUC 不升的情况下显著改善 lost anomaly / removed FP trade-off；
2. 该 readout 与 `margin`、`mat_mean` 的 Spearman/top-k overlap 不接近 1，能证明不是简单单调变换；
3. proxy 对 oracle categories 的区分方向稳定：removed FP 与 lost anomaly 不能被同一种“高异质性=坏”规则混在一起；
4. degree/rejection/reference-density shortcut 检查不过强；
5. 输出能给出可训练版本的 target proposal：例如 normal-only reference-dropout consistency、row/column consensus stability，或 mixture-support regularity。

停止或回退条件：

- 所有候选都只是 `mat_mean` / `margin` 的单调重写；
- heterogeneous support 无法在无标签统计上区分 lost anomalies 与 removed false positives；
- 最强候选主要由 degree、rejection 或 reference-density 解释；
- seed 间方向不稳定，且 split fingerprints 指向 protocol drift。

## 7. 预期产物

- `experiments/scripts/cleg3_oracle_to_proxy_candidate_signal_map.py`
- `experiments/configs/cleg3_oracle_to_proxy_candidate_signal_map.yaml`
- `experiments/outputs/cleg3_oracle_to_proxy_candidate_signal_map.json`
- `experiments/outputs/cleg3_oracle_to_proxy_candidate_signal_map.progress.json`
- `experiments/logs/cleg3_oracle_to_proxy_candidate_signal_map.log`
- `PROGRESS.md` / `insights.md` 更新

---
*Created: 2026-05-25 | Hermes / research-investigation skill*
