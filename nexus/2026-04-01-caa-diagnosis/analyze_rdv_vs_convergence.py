#!/usr/bin/env python3
"""
诊断3: VecGAD RDV vs CAA 收敛分数对比

VecGAD 使用重构差异向量 (RDV): R_i = T_i - T_hat_i
CAA 使用收敛分数 (标量): ||delta_t - delta_{t-1}|| / ||delta_{t-1}||

关键问题: CAA 是否丢失了方向信息？
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

def main():
    print("="*60)
    print("诊断3: VecGAD RDV vs CAA 收敛分数对比")
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
    
    # 计算 Hop 和 Delta
    print("计算 Hop 和 Delta...")
    hop_features = compute_hop_features_alpha0(features, adj, K=6)
    delta = compute_delta(hop_features)
    print()
    
    # ========== 方案 A: CAA 收敛分数 (标量) ==========
    print("方案 A: CAA 收敛分数 (标量)")
    
    # 计算 Delta 范数（标量）
    delta_norms = np.linalg.norm(delta, axis=2)  # [N, K]
    
    # 计算 Delta 方向（归一化向量）
    delta_norms_expanded = delta_norms[:, :, np.newaxis]  # [N, K, 1]
    delta_direction = delta / (delta_norms_expanded + 1e-8)  # [N, K, D]
    
    # 收敛分数
    convergence_scores = np.zeros((N, 5), dtype=np.float32)
    for t in range(1, 6):
        diff_norm = np.linalg.norm(delta[:, t, :] - delta[:, t-1, :], axis=1)
        prev_norm = np.linalg.norm(delta[:, t-1, :], axis=1)
        convergence_scores[:, t-1] = diff_norm / (prev_norm + 1e-8)
    
    # KS 检验（收敛分数）
    ks_convergence = []
    for t in range(5):
        ks_stat, _ = ks_2samp(convergence_scores[normal_mask, t], 
                              convergence_scores[anomaly_mask, t])
        ks_convergence.append(ks_stat)
    
    print(f"  收敛分数平均 KS: {np.mean(ks_convergence):.4f}")
    print()
    
    # ========== 方案 B: VecGAD 方式（方向信息） ==========
    print("方案 B: Delta 方向信息（类似 VecGAD RDV）")
    
    # 分析 Delta 方向的区分力
    # 计算 Delta 方向的变化
    direction_change = delta[:, 1:, :] - delta[:, :-1, :]  # [N, K-1, D]
    
    # 方向变化的范数
    direction_change_norms = np.linalg.norm(direction_change, axis=2)  # [N, K-1]
    
    # KS 检验（方向变化范数）
    ks_direction = []
    for t in range(5):
        ks_stat, _ = ks_2samp(direction_change_norms[normal_mask, t],
                              direction_change_norms[anomaly_mask, t])
        ks_direction.append(ks_stat)
    
    print(f"  方向变化范数平均 KS: {np.mean(ks_direction):.4f}")
    print()
    
    # ========== 方案 C: Delta 范数本身 ==========
    print("方案 C: Delta 范数本身")
    
    ks_delta_norm = []
    for t in range(6):
        ks_stat, _ = ks_2samp(delta_norms[normal_mask, t],
                              delta_norms[anomaly_mask, t])
        ks_delta_norm.append(ks_stat)
    
    print(f"  Delta 范数平均 KS: {np.mean(ks_delta_norm):.4f}")
    print()
    
    # ========== 对比总结 ==========
    print("="*60)
    print("对比总结")
    print("="*60)
    
    print(f"\n{'方法':<30} {'平均 KS':<10} {'说明'}")
    print("-"*60)
    print(f"{'CAA 收敛分数 (标量)':<30} {np.mean(ks_convergence):<10.4f} 丢失方向")
    print(f"{'Delta 方向变化 (向量)':<30} {np.mean(ks_direction):<10.4f} 保留方向")
    print(f"{'Delta 范数本身':<30} {np.mean(ks_delta_norm):<10.4f} 基准")
    
    print("\n关键洞察:")
    if np.mean(ks_direction) > np.mean(ks_convergence):
        print("  1. 方向变化比收敛分数更有区分力！")
        print("  2. CAA 将 Delta 压缩为标量，丢失了方向信息")
    else:
        print("  1. 收敛分数更有区分力")
    
    # VecGAD 启示
    print("\nVecGAD 启示:")
    print("  - VecGAD 使用重构差异向量 (RDV)，保留方向信息")
    print("  - CAA 使用标量收敛分数，丢失方向信息")
    print("  - 这可能是 CAA 性能差的原因之一")
    
    # 保存结果
    results = {
        "dataset": "Photo",
        "timestamp": datetime.now().isoformat(),
        "comparison": {
            "CAA_convergence_score_KS": float(np.mean(ks_convergence)),
            "Delta_direction_change_KS": float(np.mean(ks_direction)),
            "Delta_norm_KS": float(np.mean(ks_delta_norm))
        },
        "detail": {
            "convergence_ks": [float(x) for x in ks_convergence],
            "direction_ks": [float(x) for x in ks_direction],
            "delta_norm_ks": [float(x) for x in ks_delta_norm]
        }
    }
    
    output_file = Path("/root/gpufree-data/linziyao/VoxG/nexus/investigations/2026-04-01-caa-diagnosis/rdv_vs_convergence.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存")

if __name__ == "__main__":
    main()
