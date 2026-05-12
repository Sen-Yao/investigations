# Insights

## 2026-05-06 Initial positioning

This investigation starts from a user clarification:

```text
Dual-reference should be fixed global precomputed information.
The learnable part is GT: how it learns to judge anomaly degree based on fixed dual-reference.
```

This avoids two earlier issues:

1. learning/updating dual-reference can introduce bootstrap confirmation bias;
2. converting R_a into pseudo anomaly labels is unreliable.

The method direction should therefore be:

```text
fixed dual-reference evidence + 5% labeled-normal calibration -> learnable GT anomaly judgment
```

not:

```text
learnable dual-reference retrieval
```

and not:

```text
R_a-as-positive pseudo anomaly learning
```

## 2026-05-06 Mini Experiment 1: frozen fixed-dual-reference GT prototype audit

Experiment finished successfully:

- Job: `exp_20260506_150446_fixed_ref_gt_proto_seed0`
- Sweep: `kem2425f`
- WandB: https://wandb.ai/HCCS/VoxG/sweeps/kem2425f
- Runs: 4/4 finished, failed=0
- Dataset: Elliptic
- Seed: 0
- Fixed dual-reference config: current Elliptic best (`hop_attr / pca_residual / label_gate / normal_soft_or / descriptor_similarity`)
- Prototype sweep: `proto_k = [1, 2, 4, 8]`

### Results

| proto_k | single_center_l2 AUC/AP | multi_proto_l2 AUC/AP | local_knn_normal_l2 AUC/AP | ref_delta_dist AUC/AP | profile_fusion_probe AUC/AP | epoch0_head AUC/AP |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2564 / 0.0621 | 0.2564 / 0.0621 | 0.4071 / 0.0762 | 0.6089 / 0.1141 | 0.4691 / 0.0845 | 0.4877 / 0.0863 |
| 2 | 0.2564 / 0.0621 | 0.2156 / 0.0587 | 0.4071 / 0.0762 | 0.6089 / 0.1141 | 0.4691 / 0.0845 | 0.4877 / 0.0863 |
| 4 | 0.2564 / 0.0621 | 0.2221 / 0.0622 | 0.4071 / 0.0762 | 0.6089 / 0.1141 | 0.4691 / 0.0845 | 0.4877 / 0.0863 |
| 8 | 0.2564 / 0.0621 | 0.2217 / 0.0623 | 0.4071 / 0.0762 | 0.6089 / 0.1141 | 0.4691 / 0.0845 | 0.4877 / 0.0863 |

Reference retrieval quality remained unchanged:

| metric | value |
|---|---:|
| normal_ref_normal_ratio | 1.0000 |
| anom_ref_anom_ratio_on_anom_nodes | 0.6837 |

### Interpretation

1. Frozen GT embedding space does **not** provide a useful normal-prototype anomaly signal under the current random/untrained GT encoder.
2. Single-center normal distance is strongly reversed (`AUC=0.2564`), meaning anomalies are closer to the labeled-normal center than normals under this frozen embedding geometry.
3. Multi-prototype KMeans does not fix the issue. Increasing prototype count from 1 to 8 remains below random (`AUC≈0.216-0.222`). This means the problem is not simply that normality is multi-modal in this frozen embedding space.
4. Local kNN to labeled normals improves over center/prototypes but remains below random (`AUC=0.4071`).
5. Reference distance delta remains the best frozen probe (`AUC=0.6089`), confirming that reference evidence has signal, but it is not encoded as a normal-prototype distance in the current GT embedding.
6. The naive profile fusion of normal-center distance and reference delta also degrades (`AUC=0.4691`), because the reversed normal-center signal corrupts the weak reference signal.

### Conclusion

The current frozen/random GT embedding is not a valid space for normal-prototype anomaly judgment. Fixed dual-reference evidence is informative, but GT must be trained to interpret that evidence; simply building prototypes in the initial GT embedding space is not sufficient.

This result shifts the next step away from frozen prototype scoring and toward a **trainable GT judgment head/objective** conditioned on fixed dual-reference.

Recommended next diagnostic:

```text
Train a lightweight GT judgment head using 5% labeled-normal calibration plus relation-level anti-collapse/corruption, while keeping R_n/R_a fixed.
```

Key design constraint:

```text
Do not learn/update dual-reference; learn only the judgment function f_GT(v, R_n(v), R_a(v)).
```

## 2026-05-06 Phase 2 Mini Experiment: fixed dual-reference GT consistency training

### Runner/GPU note

Checked `experiment-runner` skill semantics after user asked whether the skill defaults to one GPU.

Conclusion:

```text
The skill does not default to one GPU.
```

`launch-sweep` defaults to `--gpu-mode auto`, which selects all candidate GPUs that pass the filter (`memory.free >= 2GB`, `utilization <= 85%`), unless the caller explicitly limits the number of agents.

This mini run used only one GPU because the launch command explicitly passed:

```text
--max-agents 1
```

This was an intentional caller-side limit for mini diagnostics, not a profile/skill default. The `mini` profile itself only sets validation/reporting requirements and does not specify GPU count.

### Initial failed attempt

First sweep failed due to implementation bug, not method failure:

- Job: `exp_20260506_161242_fixed_ref_gt_consistency_seed0`
- Sweep: `06pswvsv`
- Failure: `TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.`
- Cause: CUDA tensor was used as an index for numpy reference arrays inside `build_pair_batch`.
- Fix: added `_idx_to_numpy()` and converted tensor indices to CPU numpy before numpy indexing.

### Fixed run

- Job: `exp_20260506_163049_fixed_ref_gt_consistency_seed0_fix1`
- Sweep: `b8m7124p`
- WandB: https://wandb.ai/HCCS/VoxG/sweeps/b8m7124p
- Runs: 2/2 finished, failed=0
- Dataset: Elliptic
- Seed: 0
- Objective: valid vs corrupted fixed-reference context on labeled-normal nodes
- Mode: frozen GT embedding + trainable lightweight pair judgment head
- Variants: `num_epoch = [50, 100]`

### Results

| num_epoch | best_val_auc | best_val_ap | best_test_auc | best_test_ap | final_test_auc | final_test_ap | final_train_auc | final_score_std |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.7546 | 0.2097 | 0.7452 | 0.2096 | 0.7404 | 0.2059 | 0.9810 | 2.3051 |
| 100 | 0.7619 | 0.2188 | 0.7546 | 0.2180 | 0.7277 | 0.1830 | 0.9925 | 2.7410 |

Reference retrieval quality remained:

| metric | value |
|---|---:|
| normal_ref_normal_ratio | 1.0000 |
| anom_ref_anom_ratio_on_anom_nodes | 0.6837 |

### Interpretation

This is a positive signal for the Phase 2 direction.

Compared with Mini Experiment 1 frozen prototype scoring:

- frozen normal prototype distance failed (`AUC≈0.22-0.26`);
- frozen reference delta probe was modest (`AUC=0.6089`);
- trainable relation-consistency judgment reached `best_test_auc≈0.745-0.755` and `best_test_ap≈0.210-0.218`.

The result indicates that fixed dual-reference evidence is not directly useful as a Euclidean normal-prototype space, but a trainable judgment head can learn useful normal-calibrated relation consistency from 5% labeled-normal supervision.

The 100-epoch run improved best validation/test AUC but final test AUC dropped vs best, suggesting possible mild overfitting after the best epoch. Early stopping by validation AUC is needed for future runs.

### Next recommended step

Run a small stability sweep over seeds `[0,1,2,3,4]` and possibly a few regularization values for the head, while still keeping references fixed and GT encoder frozen. If stable, move to `train_gt` variants.

## 2026-05-06 Phase 2 Step A: 5-seed stability sweep launch

### Sweep configuration

- Config file: `configs/stability_fixed_ref_gt_consistency_5seed.yaml`
- Mode: frozen GT embedding + trainable lightweight judgment head
- Objective: valid vs corrupted fixed-reference context on labeled-normal nodes
- Seeds: `[0, 1, 2, 3, 4]`
- Epochs: `[50, 100]`
- Total runs: 10
- Dataset: Elliptic
- Train rate: 5% labeled-normal only

### Launch details

- Job ID: `exp_20260506_172531_fixed_ref_gt_consistency_5seed`
- Sweep ID: `wj0tjy96`
- WandB: https://wandb.ai/HCCS/VoxG/sweeps/wj0tjy96
- Profile: mini
- Agents started: 3
- GPUs selected: 4, 7, 5
- GPU mode: auto (no max-agents limit)
- Cron: ✅ (every 10m)

### GPU policy note

This launch follows the updated experiment-runner GPU policy:

```text
Default: use all idle/low-utilization GPUs
No --max-agents limit for real multi-run sweeps
Check GPU status first via --gpu-mode auto
Start one agent per selected GPU
```

Selected GPUs based on auto filter:

```text
memory.free >= 2GB
utilization <= 85%
sorted by free memory / utilization
```

This is the first sweep after skill modification, demonstrating:

```text
multi-GPU parallelism by default
wall-clock time minimization for multi-run stability validation
```

### Purpose

Verify that Phase 2 consistency training result is stable across seeds, not a seed=0 artifact.

Key metrics to watch:

| metric | purpose |
|---|---|
| best_test_auc mean±std | stability across seeds |
| best_test_ap mean±std | anomaly detection quality |
| best_epoch distribution | early stopping pattern |
| final vs best gap | overfitting check |
| final_score_std | anti-collapse behavior |

If stable, this validates:

```text
fixed dual-reference evidence is trainable via relation-consistency objective
head-only frozen encoder approach has reproducible signal
ready to move to next stage (ablations + trainable GT)
```


## 2026-05-06 Phase 2 Step B: ablation sweep and academic report

### Ablation sweep launched

- Job ID: `exp_20260506_174720_fixed_ref_gt_consistency_ablation_5seed`
- Sweep ID: `mzl1af6z`
- WandB: https://wandb.ai/HCCS/VoxG/sweeps/mzl1af6z
- Profile: mini
- Total runs: 40
- Grid:
  - `seed = [0,1,2,3,4]`
  - `num_epoch = [50,100]`
  - `ablation_mode = [full, no_ra, shuffled_ra, fixed_labeled_normal]`
- GPU mode: auto
- Agents started: 3
- GPUs selected: 4,7,5
- No `--max-agents` cap was used.
- Cron monitor: enabled, every 10m.

### Ablation purpose

The stability sweep confirmed head-only relation consistency is reproducibly positive, but it does not yet prove that the signal specifically comes from the dual-reference relation. The ablation sweep tests whether `R_a` contributes structural evidence beyond normal-reference consistency.

Interpretation plan:

| comparison | interpretation |
|---|---|
| `full > no_ra` | anomaly-like reference contributes beyond normal reference |
| `full > shuffled_ra` | node-specific `R_a(v)` matching matters |
| `full > fixed_labeled_normal` | method is not merely a labeled-normal bank shortcut |
| `full ≈ no_ra/shuffled_ra` | current gain may be normal-consistency shortcut; rethink `R_a` construction before trainable GT |

### Academic email report sent

An academic-style HTML report was sent via the email skill to `ziyao.lin@senyao.cloud`.

- Subject: `🧪 固定双参考引导的 GT Judgment Learning：阶段性方法框架与消融计划`
- Content type: `html`
- Template style: email skill academic report template, chapter-numbered structure.
- Report covered:
  1. Problem repositioning: pseudo anomaly generation → judgment function learning.
  2. Phase 1 frozen prototype negative result.
  3. Phase 2 head-only consistency 5-seed positive result.
  4. Current fixed evidence + trainable judgment framework.
  5. Ablation design and interpretation criteria.
  6. Next-stage decision rules for moving to trainable GT encoder.


## 2026-05-06 Closure: stability and ablation results

### Phase 2 Step A: 5-seed stability result

Sweep:

- Job ID: `exp_20260506_172531_fixed_ref_gt_consistency_5seed`
- Sweep ID: `wj0tjy96`
- WandB: https://wandb.ai/HCCS/VoxG/sweeps/wj0tjy96

Aggregated over 5 seeds:

| num_epoch | n | best_test_auc | best_test_ap | final_test_auc | final_test_ap | final_train_auc | final_score_std |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 5 | 0.7352±0.0122 | 0.1880±0.0153 | 0.7297±0.0128 | 0.1812±0.0161 | 0.9796±0.0034 | 2.1478±0.2532 |
| 100 | 5 | 0.7439±0.0174 | 0.1986±0.0217 | 0.7190±0.0079 | 0.1674±0.0096 | 0.9922±0.0004 | 2.4478±0.2758 |

Interpretation:

- Head-only relation consistency is reproducibly positive; the seed-0 result was not accidental.
- 100 epochs reaches a higher best AUC/AP but worse final AUC/AP, indicating overfitting after the best validation point.
- Future versions should use validation early stopping and report both best and final metrics.

### Phase 2 Step B: completed ablation result

Sweep:

- Job ID: `exp_20260506_174720_fixed_ref_gt_consistency_ablation_5seed`
- Sweep ID: `mzl1af6z`
- WandB: https://wandb.ai/HCCS/VoxG/sweeps/mzl1af6z
- Total: 40 runs = 4 ablations × 5 seeds × 2 epoch settings.

Aggregated by ablation mode:

| ablation_mode | n | best_val_auc | best_val_ap | best_test_auc | best_test_ap | final_test_auc | final_test_ap | final_train_auc | final_score_std | normal_ref_normal_ratio | anom_ref_anom_ratio_on_anom_nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_labeled_normal | 10 | 0.7254±0.0258 | 0.1803±0.0271 | 0.7219±0.0252 | 0.1820±0.0267 | 0.6998±0.0329 | 0.1601±0.0199 | 0.9740±0.0121 | 1.7750±0.2110 | 1.0000±0.0000 | 0.6782±0.0106 |
| full | 10 | 0.7419±0.0162 | 0.1910±0.0189 | 0.7384±0.0147 | 0.1919±0.0184 | 0.7234±0.0125 | 0.1735±0.0152 | 0.9858±0.0066 | 2.2965±0.2909 | 1.0000±0.0000 | 0.6782±0.0106 |
| no_ra | 10 | 0.5620±0.0286 | 0.1136±0.0081 | 0.5513±0.0197 | 0.1124±0.0058 | 0.5268±0.0136 | 0.1049±0.0050 | 0.8987±0.0508 | 2.0014±0.5931 | 1.0000±0.0000 | 0.0000±0.0000 |
| shuffled_ra | 10 | 0.5711±0.0219 | 0.1298±0.0191 | 0.5689±0.0309 | 0.1306±0.0275 | 0.5318±0.0139 | 0.1028±0.0036 | 0.9314±0.0301 | 2.1446±0.5725 | 1.0000±0.0000 | 0.1100±0.0027 |

Key findings:

1. `full` strongly outperforms `no_ra` (`best_test_auc +0.1871`), showing that deviation-side reference evidence contributes beyond normal-side consistency.
2. `full` strongly outperforms `shuffled_ra` (`best_test_auc +0.1695`), showing that the node-specific matching of `R_a(v)` matters. `R_a` is not merely a global anomaly-like pool.
3. `full` modestly outperforms `fixed_labeled_normal` (`best_test_auc +0.0165`), showing that labeled-normal anchors are already strong and that the current context-replacement objective may not fully release the value of `R_a`.

Final interpretation:

The fixed dual-reference evidence structure is valid. `R_n` and `R_a` should be treated as two-sided evidence, not labels. `R_a` is best described as answering: "if the node is not normal-like, does its deviation have a similar reference?" GT/Transformer should learn the relation between the target node and the two sides of evidence, rather than using attention itself as explanation.

The context-replacement consistency objective should be treated as a diagnostic probe. It successfully discovered a trainable signal, but it should not be frozen as the permanent training strategy for DualRefGAD.

### Closure decision

This investigation is considered closed for its original question:

```text
Can fixed dual-reference evidence support trainable anomaly judgment?
```

Answer:

```text
Yes. The evidence structure is valid and reproducibly learnable, but the next bottleneck is score-head design.
```

Next investigation:

```text
2026-05-06-dualrefgad-two-sided-score-head
```

Focus:

```text
How should GT-interacted target / normal-side / deviation-side tokens be read out and combined into an anomaly score?
```
