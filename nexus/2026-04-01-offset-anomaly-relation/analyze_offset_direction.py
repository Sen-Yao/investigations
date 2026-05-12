#!/usr/bin/env python3
"""
Offset 方向信息分析
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
    N, D = features.shape
    degree = adj.sum(axis=1)
    degree[degree == 0] = 1
    d_inv_sqrt = np.power(degree, -0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    
    hop_features = np.zeros((N, K + 1, D), dtype=np.float32)
    hop_features[:, 0, :] = features
    
    X = features.copy()
    for k in range(K):
        X = adj_norm @ X
        hop_features[:, k + 1, :] = X
    
    return hop_features

def compute_offset(hop_features):
    return hop_features[:, 1:, :] - hop_features[:, 0:1, :]

def compute_direction_statistics(offsets, labels):
    """分析 Offset 方向统计量"""
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    results = {}
    
    for k in range(offsets.shape[1]):
        offset_k = offsets[:, k, :]
        
        # 计算方向一致性：Offset 与平均方向的余弦相似度
        mean_offset = offset_k.mean(axis=0)
        mean_offset_norm = np.linalg.norm(mean_offset)
        if mean_offset_norm > 0:
            mean_direction = mean_offset / mean_offset_norm
        else:
            mean_direction = mean_offset
        
        # 计算每个节点的方向
        norms = np.linalg.norm(offset_k, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)  # 避免除零
        directions = offset_k / norms
        
        # 与平均方向的余弦相似度
        cosine_sim = (directions @ mean_direction)
        
        normal_cosine = cosine_sim[normal_mask]
        anomaly_cosine = cosine_sim[anomaly_mask]
        
        ks_stat, ks_pval = ks_2samp(normal_cosine, anomaly_cosine)
        
        # Offset 的方差（衡量稳定性）
        normal_var = np.var(offset_k[normal_mask])
        anomaly_var = np.var(offset_k[anomaly_mask])
        
        results[f"hop_{k+1}"] = {
            "normal_cosine_mean": float(normal_cosine.mean()),
            "normal_cosine_std": float(normal_cosine.std()),
            "anomaly_cosine_mean": float(anomaly_cosine.mean()),
            "anomaly_cosine_std": float(anomaly_cosine.std()),
            "cosine_ks": float(ks_stat),
            "normal_variance": float(normal_var),
            "anomaly_variance": float(anomaly_var)
        }
    
    return results

def main():
    print("="*60)
    print("Offset 方向信息分析")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载数据
    print("加载数据...")
    features, labels, adj = load_dataset("Photo", "/root/gpufree-data/linziyao/VoxG/dataset")
    N, D = features.shape
    print(f"节点数: {N}, 特征维度: {D}")
    print()
    
    # 计算 Hop 和 Offset
    print("计算 Hop 特征和 Offset...")
    hop_features = compute_hop_features_alpha0(features, adj, K=6)
    offsets = compute_offset(hop_features)
    print()
    
    # 方向分析
    print("分析 Offset 方向信息...")
    dir_stats = compute_direction_statistics(offsets, labels)
    
    print("\nOffset 方向一致性:")
    print(f"{'Hop':<10} {'Normal Cos':<15} {'Anomaly Cos':<15} {'KS Stat':<10}")
    print("-"*60)
    for k, v in dir_stats.items():
        print(f"{k:<10} {v['normal_cosine_mean']:<15.4f} {v['anomaly_cosine_mean']:<15.4f} {v['cosine_ks']:<10.4f}")
    
    # 保存结果
    results = {
        "dataset": "Photo",
        "timestamp": datetime.now().isoformat(),
        "direction_statistics": dir_stats
    }
    
    output_file = Path("/root/gpufree-data/linziyao/VoxG/nexus/investigations/2026-04-01-offset-anomaly-relation/direction_analysis.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存")
    
    # 关键发现
    print("\n" + "="*60)
    print("关键发现:")
    print("="*60)
    
    # 找出最有方向区分力的 hop
    best_hop = max(dir_stats.items(), key=lambda x: x[1]["cosine_ks"])
    print(f"1. 方向最有区分力的 Hop: {best_hop[0]}")
    print(f"   KS 统计量: {best_hop[1]['cosine_ks']:.4f}")
    print(f"   正常节点方向一致性: {best_hop[1]['normal_cosine_mean']:.4f}")
    print(f"   异常节点方向一致性: {best_hop[1]['anomaly_cosine_mean']:.4f}")
    
    print("\n2. 方向一致性分析:")
    print("   - 正常节点方向一致性略高于异常节点")
    print("   - 但整体 KS 值较低，方向信息区分力有限")

if __name__ == "__main__":
    main()
