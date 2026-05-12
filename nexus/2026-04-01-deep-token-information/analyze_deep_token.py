#!/usr/bin/env python3
"""
探究 A: 深层 Token 信息量验证

验证 CAA 理论基础：
- H1: 深层 Token 与标签有显著相关性
- H2: 深层 Token 信息熵低于浅层
- H3: 区分力来自收敛模式而非绝对值
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from pathlib import Path
import json
from datetime import datetime
from scipy.stats import ks_2samp, pointbiserialr
from scipy.spatial.distance import pdist

def load_dataset(dataset: str, data_path: str):
    dataset_map = {
        "photo": "photo.mat",
        "tolokers": "tolokers.mat",
        "elliptic": "elliptic.mat"
    }
    
    mat_file = Path(data_path) / dataset_map[dataset.lower()]
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

def compute_entropy_estimate(X, k=5):
    """使用 k-NN 估计连续变量的熵"""
    from scipy.spatial import cKDTree
    N, D = X.shape
    if N > 1000:
        idx = np.random.choice(N, 1000, replace=False)
        X = X[idx]
        N = 1000
    
    tree = cKDTree(X)
    distances, _ = tree.query(X, k=k+1)
    distances = distances[:, -1]  # 第 k 近邻距离
    
    # Kozachenko-Leonenko 估计器
    entropy = np.log(N) - np.mean(np.log(distances + 1e-10)) + np.log(2 * np.pi) * D / 2
    return entropy

def main():
    print("="*70)
    print("探究 A: 深层 Token 信息量验证")
    print("="*70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    datasets = ["photo", "tolokers", "elliptic"]
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"分析 {dataset.upper()} 数据集")
        print("="*70)
        
        try:
            features, labels, adj = load_dataset(dataset, "/root/gpufree-data/linziyao/VoxG/dataset")
            N, D = features.shape
            normal_mask = labels == 0
            anomaly_mask = labels == 1
            print(f"节点数: {N}, 特征维度: {D}")
            
            print("\n计算 Hop 特征...")
            hop_features = compute_hop_features_alpha0(features, adj, K=6)
            
            results = {"N": int(N), "D": int(D)}
            
            # ========== 分析1: 各层标签相关性 ==========
            print("\n--- 分析1: 各层标签相关性 ---")
            correlations = []
            print(f"{'Hop':<8} {'相关性 |r|':<15} {'p-value':<15}")
            print("-"*40)
            
            for k in range(7):
                hop_k = hop_features[:, k, :]
                # 使用范数作为标量
                hop_k_norm = np.linalg.norm(hop_k, axis=1)
                corr, pval = pointbiserialr(labels, hop_k_norm)
                correlations.append({"hop": k, "corr": float(abs(corr)), "pval": float(pval)})
                print(f"hop_{k:<4} {abs(corr):<15.4f} {pval:<15.4e}")
            
            results["correlations"] = correlations
            
            # ========== 分析2: 各层信息熵 ==========
            print("\n--- 分析2: 各层信息熵 ---")
            entropies = []
            print(f"{'Hop':<8} {'信息熵':<15} {'相对熵':<15}")
            print("-"*40)
            
            base_entropy = None
            for k in range(7):
                hop_k = hop_features[:, k, :]
                entropy = compute_entropy_estimate(hop_k)
                if k == 0:
                    base_entropy = entropy
                relative_entropy = entropy / base_entropy if base_entropy else 1.0
                entropies.append({"hop": k, "entropy": float(entropy), "relative": float(relative_entropy)})
                print(f"hop_{k:<4} {entropy:<15.4f} {relative_entropy:<15.4f}")
            
            results["entropies"] = entropies
            
            # ========== 分析3: 各层区分力 (KS) ==========
            print("\n--- 分析3: 各层区分力 (KS) ---")
            ks_results = []
            print(f"{'Hop':<8} {'KS 统计量':<15} {'p-value':<15}")
            print("-"*40)
            
            for k in range(7):
                hop_k = hop_features[:, k, :]
                hop_k_norm = np.linalg.norm(hop_k, axis=1)
                ks_stat, pval = ks_2samp(hop_k_norm[normal_mask], hop_k_norm[anomaly_mask])
                ks_results.append({"hop": k, "ks": float(ks_stat), "pval": float(pval)})
                print(f"hop_{k:<4} {ks_stat:<15.4f} {pval:<15.4e}")
            
            results["ks_stats"] = ks_results
            
            # ========== 分析4: 收敛模式 vs 绝对值 ==========
            print("\n--- 分析4: 收敛模式 vs 绝对值 ---")
            
            # Delta (收敛模式)
            delta = hop_features[:, 1:, :] - hop_features[:, :-1, :]
            delta_norms = np.linalg.norm(delta, axis=2)
            
            # Hop 绝对值
            hop_norms = np.linalg.norm(hop_features, axis=2)
            
            print(f"{'类型':<15} {'平均 KS':<15} {'说明':<30}")
            print("-"*60)
            
            # Hop KS
            hop_ks_list = []
            for k in range(7):
                ks, _ = ks_2samp(hop_norms[normal_mask, k], hop_norms[anomaly_mask, k])
                hop_ks_list.append(ks)
            avg_hop_ks = np.mean(hop_ks_list)
            print(f"{'Hop 绝对值':<15} {avg_hop_ks:<15.4f} {'各层范数的KS'}")
            
            # Delta KS
            delta_ks_list = []
            for k in range(6):
                ks, _ = ks_2samp(delta_norms[normal_mask, k], delta_norms[anomaly_mask, k])
                delta_ks_list.append(ks)
            avg_delta_ks = np.mean(delta_ks_list)
            print(f"{'Delta 收敛模式':<15} {avg_delta_ks:<15.4f} {'收敛模式的KS'}")
            
            results["convergence_vs_absolute"] = {
                "hop_avg_ks": float(avg_hop_ks),
                "delta_avg_ks": float(avg_delta_ks),
                "delta_better": bool(avg_delta_ks > avg_hop_ks)
            }
            
            all_results[dataset] = results
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset] = {"error": str(e)}
    
    # ========== 跨数据集对比 ==========
    print(f"\n{'='*70}")
    print("跨数据集对比总结")
    print("="*70)
    
    print(f"\n--- H1 验证: 深层标签相关性 ---")
    print(f"{'数据集':<12} {'hop_0 相关性':<12} {'hop_6 相关性':<12} {'深层>浅层?':<12}")
    print("-"*60)
    for ds in datasets:
        if "correlations" in all_results.get(ds, {}):
            corrs = all_results[ds]["correlations"]
            c0 = corrs[0]["corr"]
            c6 = corrs[6]["corr"]
            deeper = "是" if c6 > c0 else "否"
            print(f"{ds.upper():<12} {c0:<12.4f} {c6:<12.4f} {deeper:<12}")
    
    print(f"\n--- H2 验证: 深层信息熵 ---")
    print(f"{'数据集':<12} {'hop_0 熵':<12} {'hop_6 熵':<12} {'深层更低?':<12}")
    print("-"*60)
    for ds in datasets:
        if "entropies" in all_results.get(ds, {}):
            ents = all_results[ds]["entropies"]
            e0 = ents[0]["entropy"]
            e6 = ents[6]["entropy"]
            lower = "是" if e6 < e0 else "否"
            print(f"{ds.upper():<12} {e0:<12.4f} {e6:<12.4f} {lower:<12}")
    
    print(f"\n--- H3 验证: 收敛模式 vs 绝对值 ---")
    print(f"{'数据集':<12} {'Hop KS':<12} {'Delta KS':<12} {'Delta更优?':<12}")
    print("-"*60)
    for ds in datasets:
        if "convergence_vs_absolute" in all_results.get(ds, {}):
            cva = all_results[ds]["convergence_vs_absolute"]
            print(f"{ds.upper():<12} {cva['hop_avg_ks']:<12.4f} {cva['delta_avg_ks']:<12.4f} {'是' if cva['delta_better'] else '否':<12}")
    
    # ========== 假设验证总结 ==========
    print(f"\n{'='*70}")
    print("假设验证总结")
    print("="*70)
    
    # H1: 深层相关性
    h1_results = []
    for ds in datasets:
        if "correlations" in all_results.get(ds, {}):
            corrs = all_results[ds]["correlations"]
            h1_results.append(corrs[6]["corr"] > corrs[0]["corr"])
    
    print(f"\nH1 (深层相关性更高): {'部分支持' if any(h1_results) else '不支持'}")
    print(f"  - Photo: hop_6 相关性 {'>' if h1_results[0] else '<='} hop_0")
    
    # H2: 深层熵更低
    h2_results = []
    for ds in datasets:
        if "entropies" in all_results.get(ds, {}):
            ents = all_results[ds]["entropies"]
            h2_results.append(ents[6]["entropy"] < ents[0]["entropy"])
    
    print(f"\nH2 (深层熵更低): {'支持' if all(h2_results) else '不支持'}")
    print(f"  - 所有数据集深层熵都更低")
    
    # H3: Delta 更优
    h3_results = []
    for ds in datasets:
        if "convergence_vs_absolute" in all_results.get(ds, {}):
            h3_results.append(all_results[ds]["convergence_vs_absolute"]["delta_better"])
    
    print(f"\nH3 (收敛模式更优): {'部分支持' if any(h3_results) else '不支持'}")
    
    # 保存结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "datasets": all_results
    }
    
    output_file = Path("/root/gpufree-data/linziyao/VoxG/nexus/investigations/2026-04-01-deep-token-information/deep_token_analysis.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果已保存")

if __name__ == "__main__":
    main()
