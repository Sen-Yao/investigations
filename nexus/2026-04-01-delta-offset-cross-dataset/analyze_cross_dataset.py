#!/usr/bin/env python3
"""
Delta vs Offset 跨数据集分析

目标：验证信息稀释效应是否跨数据集普适
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from pathlib import Path
import json
from datetime import datetime
from scipy.stats import ks_2samp

def load_dataset(dataset: str, data_path: str):
    dataset_map = {
        "photo": "Photo.mat",
        "tolokers": "Tolokers.mat",
        "elliptic": "Elliptic.mat"
    }
    
    mat_file = Path(data_path) / dataset_map.get(dataset.lower(), f"{dataset}.mat")
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

def compute_snr_analysis(hop_features, labels):
    """计算 Delta vs Offset 的 SNR 对比"""
    delta = hop_features[:, 1:, :] - hop_features[:, :-1, :]
    offset = hop_features[:, 1:, :] - hop_features[:, 0:1, :]
    
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    results = []
    
    for k in range(min(6, delta.shape[1])):
        # Delta
        delta_k = delta[:, k, :]
        delta_norm = np.linalg.norm(delta_k, axis=1)
        delta_ks, _ = ks_2samp(delta_norm[normal_mask], delta_norm[anomaly_mask])
        delta_var = np.var(delta_k)
        delta_snr = delta_ks / np.sqrt(delta_var + 1e-8)
        
        # Offset
        offset_k = offset[:, k, :]
        offset_norm = np.linalg.norm(offset_k, axis=1)
        offset_ks, _ = ks_2samp(offset_norm[normal_mask], offset_norm[anomaly_mask])
        offset_var = np.var(offset_k)
        offset_snr = offset_ks / np.sqrt(offset_var + 1e-8)
        
        results.append({
            "hop": k+1,
            "delta_ks": float(delta_ks),
            "offset_ks": float(offset_ks),
            "delta_var": float(delta_var),
            "offset_var": float(offset_var),
            "delta_snr": float(delta_snr),
            "offset_snr": float(offset_snr),
            "snr_ratio": float(delta_snr / (offset_snr + 1e-8))
        })
    
    return results

def main():
    print("="*70)
    print("Delta vs Offset 跨数据集分析")
    print("="*70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    datasets = ["Photo", "Tolokers", "Elliptic"]
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"分析 {dataset} 数据集")
        print("="*70)
        
        try:
            features, labels, adj = load_dataset(dataset, "/root/gpufree-data/linziyao/VoxG/dataset")
            N, D = features.shape
            print(f"节点数: {N}, 特征维度: {D}")
            
            print("计算 Hop 特征...")
            hop_features = compute_hop_features_alpha0(features, adj, K=6)
            
            print("计算 SNR 对比...")
            results = compute_snr_analysis(hop_features, labels)
            
            all_results[dataset] = {
                "N": int(N),
                "D": int(D),
                "results": results
            }
            
            # 打印结果
            print(f"\n{'Hop':<8} {'Delta SNR':<12} {'Offset SNR':<12} {'SNR比值':<10}")
            print("-"*50)
            for r in results:
                print(f"hop_{r['hop']:<4} {r['delta_snr']:<12.4f} {r['offset_snr']:<12.4f} {r['snr_ratio']:<10.2f}x")
            
            avg_delta_snr = np.mean([r["delta_snr"] for r in results])
            avg_offset_snr = np.mean([r["offset_snr"] for r in results])
            avg_snr_ratio = avg_delta_snr / (avg_offset_snr + 1e-8)
            
            print(f"\n平均 SNR:")
            print(f"  Delta: {avg_delta_snr:.4f}")
            print(f"  Offset: {avg_offset_snr:.4f}")
            print(f"  SNR 比值: {avg_snr_ratio:.2f}x")
            
            all_results[dataset]["summary"] = {
                "avg_delta_snr": float(avg_delta_snr),
                "avg_offset_snr": float(avg_offset_snr),
                "avg_snr_ratio": float(avg_snr_ratio)
            }
            
        except Exception as e:
            print(f"错误: {e}")
            all_results[dataset] = {"error": str(e)}
    
    # 跨数据集对比
    print(f"\n{'='*70}")
    print("跨数据集对比")
    print("="*70)
    
    print(f"\n{'数据集':<15} {'D':<10} {'Delta SNR':<12} {'Offset SNR':<12} {'SNR比值':<10}")
    print("-"*70)
    
    for dataset in datasets:
        if "summary" in all_results.get(dataset, {}):
            s = all_results[dataset]["summary"]
            print(f"{dataset:<15} {all_results[dataset]['D']:<10} {s['avg_delta_snr']:<12.4f} {s['avg_offset_snr']:<12.4f} {s['avg_snr_ratio']:<10.2f}x")
    
    # 关键洞察
    print(f"\n{'='*70}")
    print("关键洞察")
    print("="*70)
    
    ratios = [all_results[d]["summary"]["avg_snr_ratio"] for d in datasets if "summary" in all_results.get(d, {})]
    if ratios:
        print(f"\n1. SNR 比值范围: {min(ratios):.2f}x - {max(ratios):.2f}x")
        print(f"2. 平均 SNR 比值: {np.mean(ratios):.2f}x")
        
        if all(r > 1 for r in ratios):
            print(f"3. **信息稀释效应在所有数据集上都存在！**")
            print(f"   Delta 始终比 Offset 有更高的信噪比")
        
        # 分析与特征维度的关系
        print(f"\n4. 特征维度与 SNR 比值的关系:")
        for dataset in datasets:
            if "summary" in all_results.get(dataset, {}):
                D = all_results[dataset]["D"]
                ratio = all_results[dataset]["summary"]["avg_snr_ratio"]
                print(f"   {dataset} (D={D}): {ratio:.2f}x")
    
    # 保存结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "datasets": all_results
    }
    
    output_file = Path("/root/gpufree-data/linziyao/VoxG/nexus/investigations/2026-04-01-delta-offset-cross-dataset/cross_dataset_analysis.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果已保存")

if __name__ == "__main__":
    main()
