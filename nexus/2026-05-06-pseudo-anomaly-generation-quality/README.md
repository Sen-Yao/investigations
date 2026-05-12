# Pseudo Anomaly Generation Quality

**创建日期**: 2026-05-06  
**状态**: Planning / Diagnosis  
**主题**: 研究 dual-reference relation 如何转化为有效 anomaly learning signal；诊断当前 pseudo anomaly generation 为什么没有提供正学习信号。

## 背景

前序 investigations：

- `2026-04-30-reference-guided-pseudo-anomaly`
- `2026-05-05-elliptic-training-degradation`

当前已有关键结果：

### 1. Elliptic full train_all formal sweep

- Sweep ID: `w47qp7ah`
- Objective: `target_ref_guided`
- pseudo_beta: `{0.1, 0.2, 0.3}`
- seeds: `[0,1,2,3,4]`

最佳组：

| method | final AUC | final AP | best_test_auc | best_test_ap |
|---|---:|---:|---:|---:|
| full train_all, beta=0.2 | 0.5359±0.0618 | 0.0997±0.0200 | 0.6836±0.0229 | 0.1530±0.0149 |

### 2. Elliptic head-only frozen geometry 5-seed

- Sweep ID: `h21wc31z`
- diagnosis_mode: `train_head_only`
- pseudo_beta: `0.2`
- seeds: `[0,1,2,3,4]`

| method | final AUC | final AP | best_test_auc | best_test_ap | pseudo_auc |
|---|---:|---:|---:|---:|---:|
| frozen geometry + head-only | 0.6649±0.0826 | 0.1536±0.0459 | 0.7514±0.0243 | 0.2566±0.0590 | 0.5154±0.0002 |

## Motivation

上述结果说明：

1. dual-reference geometry 对 Elliptic 是有用的；
2. end-to-end training 会破坏该 geometry；
3. frozen geometry + head-only 显著优于 full train_all；
4. 但 `pseudo_auc≈0.515`，说明当前 pseudo anomaly BCE 任务几乎没有形成有效 synthetic separability。

因此，当前核心问题不再是简单调 `pseudo_beta` 或训练方式，而是：

> Reference geometry works; current pseudo anomaly generation does not yet provide a reliable positive learning signal.

中文：

> reference geometry 有效；当前伪异常生成还没有提供可靠正学习信号。

## 核心研究问题

Current dual-reference retrieval provides useful anomaly geometry, but why does converting it into pseudo anomalies fail to provide positive learning signal?

中文：

> dual-reference retrieval 已经提供了有效异常几何，但为什么把它转换成伪异常样本后，不能提供有效正学习信号？

## 当前 pseudo generation

当前 `target_ref_guided` 生成方式：

```python
rn_i = mean(emb[normal_refs[i]])
ra_i = mean(emb[anom_refs[i]])
direction_i = normalize(ra_i - rn_i)
pseudo_i = emb[i] + beta * direction_i
```

隐含假设：

```text
真实异常 ≈ normal embedding 沿 target-specific R_a - R_n 方向平移
```

本 investigation 要检查这个假设是否成立。

## 非目标

暂时不做：

- 直接追 SOTA；
- 大规模 formal sweep；
- 加复杂 loss / decoder；
- 在未诊断前提出新 objective。

本阶段先做 pseudo quality audit。
