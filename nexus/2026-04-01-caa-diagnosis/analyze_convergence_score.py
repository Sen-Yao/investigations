#!/usr/bin/env python3
"""
诊断1: 收敛分数有效性分析

CAA 使用 convergence_score = ||delta_t - delta_{t-1}|| / ||delta_{t-1}||

验证：收敛分数能否区分正/异常节点？
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
    adj_norm = np.diag(d_inv_sqrt) @ adj @ np.diag(d_inv_sqrt)
    
    hop_features = np.zeros((N, K + 1, D), dtype=np.float32)
    hop_features[:, 0, :] = features
    
    X = features.copy()
    for k in range(K):
        X = adj_norm @ X
        hop_features[:, k + 1, :] = X
    
    return hop_features

def compute_delta(hop_features):
    return hop_features[:, 1:, :] - hop_features[:, :-1, :]

def compute_convergence_score(delta):
    """
    CAA 收敛分数: ||delta_t - delta_{t-1}|| / ||delta_{t-1}||
    
    Args:
        delta: [N, K, D] Delta 特征 (delta_k = hop_k - hop_{k-1})
    
    Returns:
        convergence_scores: [N, K-1] 每个节点的收敛分数
    """
    N, K, D = delta.shape
    
    # 收敛分数计算：从 delta_2 开始计算
    convergence_scores = np.zeros((N, K - 1), dtype=np.float32)
    
    for t in range(1, K):
        delta_t = delta[:, t, :]      # delta_t = hop_{t+1} - hop_t
        delta_t_prev = delta[:, t-1, :]  # delta_{t-1} = hop_t - hop_{t-1}
        
        # ||delta_t - delta_{t-1}|| / ||delta_{t-1}||
        diff_norm = np.linalg.norm(delta_t - delta_t_prev, axis=1)
        prev_norm = np.linalg.norm(delta_t_prev, axis=1)
        
        convergence_scores[:, t-1] = diff_norm / (prev_norm + 1e-8)
    
    return convergence_scores

def analyze_convergence_scores(scores, labels):
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    results = {}
    
    for t in range(scores.shape[1]):
        score_t = scores[:, t]
        
        normal_scores = score_t[normal_mask]
        anomaly_scores = score_t[anomaly_mask]
        
        ks_stat, ks_pval = ks_2samp(normal_scores, anomaly_scores)
        
        results[f"delta_{t+1}_to_{t+2}"] = {
            "normal_mean": float(normal_scores.mean()),
            "normal_std": float(normal_scores.std()),
            "anomaly_mean": float(anomaly_scores.mean()),
            "anomaly_std": float(anomaly_scores.std()),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "separation": float(normal_scores.mean() / anomaly_scores.mean()) if anomaly_scores.mean() > 0 else 0
        }
    
    return results

def main():
    print("="*60)
    print("诊断1: 收敛分数有效性分析")
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
    
    # 计算 Hop 特征
    print("计算 Hop 特征 (alpha=0)...")
    hop_features = compute_hop_features_alpha0(features, adj, K=6)
    print()
    
    # 计算 Delta
    print("计算 Delta...")
    delta = compute_delta(hop_features)
    print(f"Delta 形状: {delta.shape}")
    print()
    
    # 计算收敛分数
    print("计算 CAA 收敛分数...")
    convergence_scores = compute_convergence_score(delta)
    print(f"收敛分数形状: {convergence_scores.shape}")
    print()
    
    # 分析
    print("分析收敛分数区分能力...")
    results = analyze_convergence_scores(convergence_scores, labels)
    
    print("\n收敛分数统计:")
    print(f"{'Transition':<20} {'Normal Mean':<12} {'Anomaly Mean':<12} {'KS Stat':<10}")
    print("-"*60)
    for k, v in results.items():
        print(f"{k:<20} {v['normal_mean']:<12.4f} {v['anomaly_mean']:<12.4f} {v['ks_statistic']:<10.4f}")
    
    # 保存结果
    output = {
        "dataset": "Photo",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "alpha": 0,
            "K": 6,
            "convergence_formula": "||delta_t - delta_{t-1}|| / ||delta_{t-1}||"
        },
        "results": results
    }
    
    output_file = Path("/root/gpufree-data/linziyao/VoxG/nexus/investigations/2026-04-01-caa-diagnosis/convergence_score_analysis.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果已保存")
    
    # 关键发现
    print("\n" + "="*60)
    print("关键发现:")
    print("="*60)
    
    best = max(results.items(), key=lambda x: x[1]["ks_statistic"])
    print(f"1. 最有区分力的收敛分数: {best[0]}")
    print(f"   KS 统计量: {best[1]['ks_statistic']:.4f}")
    
    avg_ks = np.mean([v["ks_statistic"] for v in results.values()])
    print(f"2. 平均 KS 统计量: {avg_ks:.4f}")
    
    if avg_ks < 0.15:
        print("3. 结论: 收敛分数区分能力有限 (KS < 0.15)")

if __name__ == "__main__":
    main()
