#!/usr/bin/env python3
"""
Offset 与异常检测性能关系分析

关键约束：
- alpha=0（纯邻居聚合）
- 邻接矩阵对称归一化 D^{-0.5}AD^{-0.5}
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from pathlib import Path
import json
from datetime import datetime

def load_dataset(dataset: str, data_path: str):
    """加载数据集"""
    mat_file = Path(data_path) / f"{dataset}.mat"
    if not mat_file.exists():
        raise FileNotFoundError(f"Dataset not found: {mat_file}")
    
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
    
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)
    adj = adj.astype(np.float32)
    
    return features, labels, adj

def compute_hop_features_alpha0(features, adj, K=6):
    """
    计算 K-hop 特征（alpha=0，纯邻居聚合）
    邻接矩阵对称归一化 D^{-0.5}AD^{-0.5}
    """
    N, D = features.shape
    
    # 对称归一化 D^{-0.5}AD^{-0.5}
    degree = adj.sum(axis=1)
    degree[degree == 0] = 1
    d_inv_sqrt = np.power(degree, -0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    
    print(f"归一化验证:")
    print(f"  对角元素之和: {np.diag(adj_norm).sum():.4f} (应为 0)")
    print(f"  行和范围: [{adj_norm.sum(axis=1).min():.4f}, {adj_norm.sum(axis=1).max():.4f}]")
    
    # 计算 K-hop 特征
    hop_features = np.zeros((N, K + 1, D), dtype=np.float32)
    hop_features[:, 0, :] = features
    
    X = features.copy()
    for k in range(K):
        # alpha=0: 纯邻居聚合，无 PPR 残差
        X = adj_norm @ X
        hop_features[:, k + 1, :] = X
    
    return hop_features

def compute_offset(hop_features):
    """计算 Offset: offset_k = hop_k - hop_0"""
    return hop_features[:, 1:, :] - hop_features[:, 0:1, :]

def compute_offset_statistics(offsets, labels):
    """计算 Offset 统计量"""
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    results = {}
    
    for k in range(offsets.shape[1]):
        offset_k = offsets[:, k, :]
        
        # 正常节点统计
        normal_offsets = offset_k[normal_mask]
        normal_norms = np.linalg.norm(normal_offsets, axis=1)
        
        # 异常节点统计
        anomaly_offsets = offset_k[anomaly_mask]
        anomaly_norms = np.linalg.norm(anomaly_offsets, axis=1)
        
        # KS 检验
        from scipy.stats import ks_2samp
        ks_stat, ks_pval = ks_2samp(normal_norms, anomaly_norms)
        
        results[f"hop_{k+1}"] = {
            "normal_norm_mean": float(normal_norms.mean()),
            "normal_norm_std": float(normal_norms.std()),
            "anomaly_norm_mean": float(anomaly_norms.mean()),
            "anomaly_norm_std": float(anomaly_norms.std()),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "separation_ratio": float(anomaly_norms.mean() / normal_norms.mean()) if normal_norms.mean() > 0 else 0
        }
    
    return results

def main():
    print("="*60)
    print("Offset 与异常检测性能关系分析")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载数据
    print("加载 Photo 数据集...")
    features, labels, adj = load_dataset("Photo", "/root/gpufree-data/linziyao/VoxG/dataset")
    N, D = features.shape
    print(f"节点数: {N}, 特征维度: {D}")
    print(f"正常节点: {(labels==0).sum()}, 异常节点: {(labels==1).sum()}")
    print()
    
    # 计算 Hop 特征 (alpha=0)
    print("计算 Hop 特征 (alpha=0, 对称归一化)...")
    hop_features = compute_hop_features_alpha0(features, adj, K=6)
    print()
    
    # 计算 Offset
    print("计算 Offset...")
    offsets = compute_offset(hop_features)
    print(f"Offset 形状: {offsets.shape}")
    print()
    
    # 分析 Offset 统计量
    print("分析 Offset 统计量...")
    stats = compute_offset_statistics(offsets, labels)
    
    print("\nOffset 范数统计:")
    print(f"{'Hop':<10} {'Normal Mean':<15} {'Anomaly Mean':<15} {'KS Stat':<10} {'Separation':<10}")
    print("-"*60)
    for k, v in stats.items():
        print(f"{k:<10} {v['normal_norm_mean']:<15.4f} {v['anomaly_norm_mean']:<15.4f} {v['ks_statistic']:<10.4f} {v['separation_ratio']:<10.4f}")
    
    # 保存结果
    results = {
        "dataset": "Photo",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "alpha": 0,
            "normalization": "D^{-0.5}AD^{-0.5}",
            "K": 6
        },
        "statistics": stats
    }
    
    output_file = Path("~/VoxG/nexus/investigations/2026-04-01-offset-anomaly-relation/offset_analysis.json")
    output_file.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(output_file.expanduser(), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存: {output_file}")
    
    # 关键发现
    print("\n" + "="*60)
    print("关键发现:")
    print("="*60)
    
    # 找出最有区分力的 hop
    best_hop = max(stats.items(), key=lambda x: x[1]["ks_statistic"])
    print(f"1. 最有区分力的 Hop: {best_hop[0]}")
    print(f"   KS 统计量: {best_hop[1]['ks_statistic']:.4f}")
    print(f"   正常/异常分离比: {best_hop[1]['separation_ratio']:.4f}")
    
    # 分析趋势
    ks_stats = [v["ks_statistic"] for v in stats.values()]
    if ks_stats[-1] > ks_stats[0]:
        print(f"2. 趋势: KS 统计量随 Hop 增加而增大")
    else:
        print(f"2. 趋势: KS 统计量随 Hop 增加而减小")

if __name__ == "__main__":
    main()
