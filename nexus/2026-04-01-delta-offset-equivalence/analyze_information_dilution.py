#!/usr/bin/env python3
"""
信息稀释分析：为什么累积导致区分力下降？

关键发现：
- Delta 范数递减：11.43 → 0.32
- Offset 范数稳定：11.43 → 11.33
- Offset 方差是 Delta 的 5.18 倍

假设：累积导致深层信息被稀释到高方差噪声中
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

def main():
    print("="*60)
    print("信息稀释分析")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载数据
    features, labels, adj = load_dataset("Photo", "/root/gpufree-data/linziyao/VoxG/dataset")
    N, D = features.shape
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    hop_features = compute_hop_features_alpha0(features, adj, K=6)
    
    # 计算 Delta 和 Offset
    delta = hop_features[:, 1:, :] - hop_features[:, :-1, :]
    offset = hop_features[:, 1:, :] - hop_features[:, 0:1, :]
    
    # ========== 核心分析：累积效应 ==========
    print("="*60)
    print("累积效应分析")
    print("="*60)
    
    # 分析每个 hop 的 Delta 信息贡献
    print(f"\nDelta 各层的信息贡献:")
    print(f"{'Hop':<10} {'Delta Norm':<15} {'贡献占比':<15}")
    print("-"*50)
    
    delta_norms = np.linalg.norm(delta, axis=2)
    total_delta_norm = delta_norms.sum(axis=1, keepdims=True)
    
    for k in range(6):
        contrib = delta_norms[:, k].mean() / delta_norms.sum(axis=1).mean()
        print(f"hop_{k+1:<5} {delta_norms[:, k].mean():<15.4f} {contrib*100:<15.2f}%")
    
    # 分析累积后的信噪比
    print(f"\n累积信噪比分析:")
    
    # 信号：各层 Delta 的区分力
    # 噪声：累积方差
    
    results = []
    for k in range(6):
        # 该层 Delta 的 KS
        delta_k = delta[:, k, :]
        delta_k_norm = np.linalg.norm(delta_k, axis=1)
        delta_ks, _ = ks_2samp(delta_k_norm[normal_mask], delta_k_norm[anomaly_mask])
        
        # 累积到该层的 Offset 的 KS
        offset_k = offset[:, k, :]
        offset_k_norm = np.linalg.norm(offset_k, axis=1)
        offset_ks, _ = ks_2samp(offset_k_norm[normal_mask], offset_k_norm[anomaly_mask])
        
        # 方差比
        delta_var = np.var(delta_k)
        offset_var = np.var(offset_k)
        
        # 信噪比：KS / sqrt(方差)
        delta_snr = delta_ks / np.sqrt(delta_var + 1e-8)
        offset_snr = offset_ks / np.sqrt(offset_var + 1e-8)
        
        results.append({
            "hop": k+1,
            "delta_ks": float(delta_ks),
            "offset_ks": float(offset_ks),
            "delta_var": float(delta_var),
            "offset_var": float(offset_var),
            "delta_snr": float(delta_snr),
            "offset_snr": float(offset_snr)
        })
    
    print(f"\n{'Hop':<10} {'Delta SNR':<15} {'Offset SNR':<15} {'比值':<10}")
    print("-"*60)
    for r in results:
        ratio = r["delta_snr"] / (r["offset_snr"] + 1e-8)
        print(f"hop_{r['hop']:<5} {r['delta_snr']:<15.4f} {r['offset_snr']:<15.4f} {ratio:.2f}x")
    
    # ========== 关键洞察 ==========
    print()
    print("="*60)
    print("关键洞察")
    print("="*60)
    
    avg_delta_snr = np.mean([r["delta_snr"] for r in results])
    avg_offset_snr = np.mean([r["offset_snr"] for r in results])
    
    print(f"\n信噪比对比:")
    print(f"  Delta 平均 SNR: {avg_delta_snr:.4f}")
    print(f"  Offset 平均 SNR: {avg_offset_snr:.4f}")
    print(f"  Delta 是 Offset 的 {avg_delta_snr / avg_offset_snr:.2f} 倍")
    
    print(f"\n信息稀释效应:")
    print("  1. Delta 是'增量'：每层独立，信息清晰")
    print("  2. Offset 是'累积'：深层信息被累积方差稀释")
    print("  3. 类似于信号处理中的'累积噪声'")
    
    print(f"\n数学解释:")
    print("  Offset_k = Delta_1 + Delta_2 + ... + Delta_k")
    print("  Var(Offset_k) = Sum Var(Delta_i) + 2*Sum Cov(Delta_i, Delta_j)")
    print("  累积导致方差增长，深层信息被'淹没'")
    
    # 保存结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "snr_analysis": results,
        "summary": {
            "avg_delta_snr": float(avg_delta_snr),
            "avg_offset_snr": float(avg_offset_snr),
            "snr_ratio": float(avg_delta_snr / avg_offset_snr)
        }
    }
    
    output_file = Path("/root/gpufree-data/linziyao/VoxG/nexus/investigations/2026-04-01-delta-offset-equivalence/snr_analysis.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果已保存")

if __name__ == "__main__":
    main()
