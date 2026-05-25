# DualRefGAD Route2.5: Normal-only Matrix Autoencoder Probe

> Created: 2026-05-19  
> Project: DualRefGAD / Elliptic  
> Status: active diagnostic investigation

## Research question

DualRefGAD 过去的 Route2 发现：多参考响应矩阵里可能存在 margin 之外的信息，但简单标量聚合（例如 `mat_mean`）存在 seed instability。这个 investigation 测试一个更具体的问题：

> 完整 response matrix `M(v) ∈ R^{K_n × K_a}` 的 normal-only pattern 是否能通过小型 autoencoder 学到，并用 reconstruction error 区分异常节点？

其中：

```text
M_ij(v) = cos(h_v - r_{n,i}, r_{a,j} - r_{n,i})
```

默认 `K_n=4, K_a=16`，因此 `flatten(M) ∈ R^64`。

## Why this is Route2.5

Route2 是 response matrix scalar aggregation：`mean(M)`, `high08_ratio`, `top5_mean` 等。Route2.5 不再假设某个手工统计量足够，而是测试 normal-only response distribution 是否有可学习结构。

It still starts as a Matrix AE probe, but after the subsequent orientation/regime, old-setting alignment, and AMRF diagnostics it has effectively become a **Route2.5 response-matrix mechanism record**. Treat the later sections as part of the same evidence chain, not as a separate method claim.

It仍然不是最终方法，而是低成本诊断：

- 如果 AE reconstruction error 没有信号，说明 full matrix pattern 大概率不值得继续建模；
- 如果 AE 有信号但只是在学 degree/rejection，则转向 regime diagnostic；
- 如果 AE 接近或超过最强 scalar baseline 且互补，则值得推进到 denoising/regime-conditioned decoder。

## Protocol

- Dataset: real Elliptic only.
- Seed: start with seed 0 probe.
- Encoder/reference selection: frozen upstream VecGAD encoder; no end-to-end training.
- Training data: labeled-normal training nodes only.
- Labels: evaluation/autopsy only，不参与训练与 checkpoint selection。
- AE candidates: `64 → 32 → latent → 32 → 64`, latent in `{4,8,16}`.
- Score: per-node matrix reconstruction MSE.
- Comparisons: margin, -margin, mat_mean, -mat_mean, mat_high08_ratio, normal rejection, residual norm, degree.

## Decision rules

- `PROMOTE`: best AE AUC > strongest scalar baseline AUC + 0.02.
- `PROMOTE_CAUTION`: AE 接近 strongest scalar baseline（within 0.02）且与 margin 低相关。
- `DROP`: AE AUC < 0.58.
- `DROP_OR_REPAIR`: AE 强相关 degree/rejection，说明可能只是 regime proxy。

## Files

- `hypothesis.md` — scientific hypotheses and candidate interpretations.
- `PROGRESS.md` — chronological execution record.
- `insights.md` — updated after probe completion.
- `experiments/scripts/route25_matrix_autoencoder_probe.py` — probe implementation.
- `experiments/configs/route25_matrix_ae_elliptic_seed0_probe.yaml` — runner profile config.
