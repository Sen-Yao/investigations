#!/usr/bin/env python3
"""
Delta vs Offset 本质关系分析

核心问题：Delta 和 Offset 可以互相推导，但 MI/熵差距大，为什么？

数学关系：
- Delta_k = hop_k - hop_{k-1}
- Offset_k = hop_k - hop_0
- Offset_k = Sum_{i=1}^{k} Delta_i（累积）

约束：alpha=0，邻接矩阵双边归一化
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from pathlib import Path
import json
from datetime import datetime
from scipy.stats import ks_2samp

def load_dataset(dataset: str, data_path: str):
    mat_file = Path(data_path) / f"{dataset}.mat"
    data = sio.loadmat(mat_file)
    features = data.get("Attributes", data.get("X", data.get("features")))
    labels = data.get("Label", data.get("Y", data.get("labels")))
    adj = data.get("Network", data.get("A", data.get("adj")))
    
    if sp.issparse(features):
        features = features.toarray()
    if sp.issparse(adj):
        adj = adj.toarray()
    
    labels = labels.flatten()
    if labels.min() > 0:
        labels = labels - labels.min()
    
    return features.astype(np.float32), labels.astype(np.int64), adj.astype(np.float32)

def compute_hop_features_alpha0(features, adj, K=6):
    """alpha=0，双边归一化"""
    N, D = features.shape
    degree = adj.sum(axis=1)
    degree[degree == 0] = 1
    d_inv_sqrt = np.power(degree, -0.5)
    adj_norm = np.diag(d_inv_sqrt) @ adj @ np.diag(d_inv_sqrt)
    
    hop_features = np.zeros((N, K + 1, D), dtype=np.float32)
    hop_features[:, 0, :] = features
    
    X = features.copy()
    for k in range(K):
        X = adj_norm @ X
        hop_features[:, k + 1, :] = X
    
    return hop_features

def compute_delta(hop_features):
    """Delta_k = hop_k - hop_{k-1}"""
    return hop_features[:, 1:, :] - hop_features[:, :-1, :]

def compute_offset(hop_features):
    """Offset_k = hop_k - hop_0"""
    return hop_features[:, 1:, :] - hop_features[:, 0:1, :]

def compute_offset_from_delta(delta):
    """Offset_k = Sum_{i=1}^{k} Delta_i（累积）"""
    return np.cumsum(delta, axis=1)

def compute_delta_from_offset(offset):
    """Delta_k = Offset_k - Offset_{k-1}"""
    delta = np.zeros_like(offset)
    delta[:, 0, :] = offset[:, 0, :]  # Delta_1 = Offset_1
    delta[:, 1:, :] = offset[:, 1:, :] - offset[:, :-1, :]
    return delta

def main():
    print("="*60)
    print("Delta vs Offset 本质关系分析")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载数据
    print("加载 Photo 数据集...")
    features, labels, adj = load_dataset("Photo", "/root/gpufree-data/linziyao/VoxG/dataset")
    N, D = features.shape
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    print(f"节点数: {N}, 特征维度: {D}")
    print()
    
    # 计算 Hop 特征
    print("计算 Hop 特征 (alpha=0, 双边归一化)...")
    hop_features = compute_hop_features_alpha0(features, adj, K=6)
    print()
    
    # ========== 分析1：数学关系验证 ==========
    print("="*60)
    print("分析1：数学关系验证")
    print("="*60)
    
    # 计算两种方式
    delta_direct = compute_delta(hop_features)
    offset_direct = compute_offset(hop_features)
    
    # 从 Delta 推导 Offset
    offset_from_delta = compute_offset_from_delta(delta_direct)
    
    # 从 Offset 推导 Delta
    delta_from_offset = compute_delta_from_offset(offset_direct)
    
    # 验证等价性
    offset_diff = np.abs(offset_direct - offset_from_delta).max()
    delta_diff = np.abs(delta_direct - delta_from_offset).max()
    
    print(f"\nOffset 直接计算 vs Delta累积推导:")
    print(f"  最大差异: {offset_diff:.2e}")
    print(f"  是否等价: {'✅ 是' if offset_diff < 1e-5 else '❌ 否'}")
    
    print(f"\nDelta 直接计算 vs Offset差分推导:")
    print(f"  最大差异: {delta_diff:.2e}")
    print(f"  是否等价: {'✅ 是' if delta_diff < 1e-5 else '❌ 否'}")
    print()
    
    # ========== 分析2：统计特性对比 ==========
    print("="*60)
    print("分析2：统计特性对比")
    print("="*60)
    
    # 计算范数
    delta_norms = np.linalg.norm(delta_direct, axis=2)  # [N, K]
    offset_norms = np.linalg.norm(offset_direct, axis=2)  # [N, K]
    
    print(f"\n范数统计:")
    print(f"{'Hop':<10} {'Delta Norm (mean)':<20} {'Offset Norm (mean)':<20}")
    print("-"*60)
    for k in range(6):
        print(f"hop_{k+1:<5} {delta_norms[:, k].mean():<20.4f} {offset_norms[:, k].mean():<20.4f}")
    
    # KS 检验对比
    print(f"\nKS 检验对比 (正/异常区分力):")
    print(f"{'Hop':<10} {'Delta KS':<15} {'Offset KS':<15} {'差异':<10}")
    print("-"*60)
    
    ks_results = []
    for k in range(6):
        delta_ks, _ = ks_2samp(delta_norms[normal_mask, k], delta_norms[anomaly_mask, k])
        offset_ks, _ = ks_2samp(offset_norms[normal_mask, k], offset_norms[anomaly_mask, k])
        ks_results.append({
            "hop": k+1,
            "delta_ks": float(delta_ks),
            "offset_ks": float(offset_ks),
            "diff": float(delta_ks - offset_ks)
        })
        print(f"hop_{k+1:<5} {delta_ks:<15.4f} {offset_ks:<15.4f} {delta_ks - offset_ks:+.4f}")
    
    # ========== 分析3：信息论视角 ==========
    print()
    print("="*60)
    print("分析3：信息论视角")
    print("="*60)
    
    # 计算方差（信息量的代理指标）
    delta_var = np.var(delta_direct, axis=0).mean()  # 各维度平均方差
    offset_var = np.var(offset_direct, axis=0).mean()
    
    # 计算与标签的相关性
    from scipy.stats import pointbiserialr
    
    delta_corr = []
    offset_corr = []
    for k in range(6):
        # 取范数作为标量
        d_corr, _ = pointbiserialr(labels, delta_norms[:, k])
        o_corr, _ = pointbiserialr(labels, offset_norms[:, k])
        delta_corr.append(abs(d_corr))
        offset_corr.append(abs(o_corr))
    
    print(f"\n与标签的相关性 (|r|):")
    print(f"{'Hop':<10} {'Delta Corr':<15} {'Offset Corr':<15}")
    print("-"*50)
    for k in range(6):
        print(f"hop_{k+1:<5} {delta_corr[k]:<15.4f} {offset_corr[k]:<15.4f}")
    
    print(f"\n方差对比:")
    print(f"  Delta 平均方差: {delta_var:.4f}")
    print(f"  Offset 平均方差: {offset_var:.4f}")
    print(f"  比值: {offset_var / delta_var:.2f}x")
    
    # ========== 核心洞察 ==========
    print()
    print("="*60)
    print("核心洞察")
    print("="*60)
    
    avg_delta_ks = np.mean([r["delta_ks"] for r in ks_results])
    avg_offset_ks = np.mean([r["offset_ks"] for r in ks_results])
    avg_delta_corr = np.mean(delta_corr)
    avg_offset_corr = np.mean(offset_corr)
    
    print(f"\n区分力对比 (平均 KS):")
    print(f"  Delta: {avg_delta_ks:.4f}")
    print(f"  Offset: {avg_offset_ks:.4f}")
    print(f"  差异: {avg_delta_ks - avg_offset_ks:+.4f}")
    
    print(f"\n与标签相关性 (平均 |r|):")
    print(f"  Delta: {avg_delta_corr:.4f}")
    print(f"  Offset: {avg_offset_corr:.4f}")
    
    # 保存结果
    results = {
        "dataset": "Photo",
        "timestamp": datetime.now().isoformat(),
        "math_verification": {
            "offset_from_delta_diff": float(offset_diff),
            "delta_from_offset_diff": float(delta_diff),
            "mathematically_equivalent": bool(offset_diff < 1e-5 and delta_diff < 1e-5)
        },
        "ks_comparison": ks_results,
        "variance_comparison": {
            "delta_variance": float(delta_var),
            "offset_variance": float(offset_var),
            "ratio": float(offset_var / delta_var)
        },
        "label_correlation": {
            "delta_corr_mean": float(avg_delta_corr),
            "offset_corr_mean": float(avg_offset_corr)
        }
    }
    
    output_file = Path("/root/gpufree-data/linziyao/VoxG/nexus/investigations/2026-04-01-delta-offset-equivalence/equivalence_analysis.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存")

if __name__ == "__main__":
    main()
