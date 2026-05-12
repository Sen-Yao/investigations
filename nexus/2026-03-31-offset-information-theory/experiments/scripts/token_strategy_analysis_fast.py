#!/usr/bin/env python3
"""
Token 策略信息论分析脚本（采样加速版）
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gamma
from sklearn.feature_selection import mutual_info_classif
import scipy.io as sio
import scipy.sparse as sp

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, sp.spmatrix]:
    data_paths = [
        f"/root/gpufree-data/linziyao/VoxG/dataset/{dataset_name}.mat",
        f"/root/gpufree-data/linziyao/VoxG/dataset/{dataset_name.capitalize()}.mat",
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            data = sio.loadmat(path)
            break
    else:
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")
    
    feature_keys = ['X', 'features', 'Attributes', 'attr', 'feature']
    for key in feature_keys:
        if key in data:
            feat = data[key]
            features = feat.toarray().astype(np.float32) if sp.issparse(feat) else np.array(feat, dtype=np.float32)
            break
    
    label_keys = ['y', 'label', 'Label', 'labels', 'Class', 'str_anomaly_label', 'attr_anomaly_label']
    for key in label_keys:
        if key in data:
            labels = data[key].flatten().astype(np.int32)
            break
    
    adj_keys = ['A', 'adj', 'Network', 'network', 'adjacency']
    for key in adj_keys:
        if key in data:
            adj = sp.csr_matrix(data[key])
            break
    
    labels = (labels != 0).astype(np.int32)
    print(f"数据集: {dataset_name}")
    print(f"节点数: {features.shape[0]}, 特征维度: {features.shape[1]}")
    print(f"正常节点: {(labels == 0).sum()}, 异常节点: {(labels == 1).sum()}")
    
    return features, labels, adj


def compute_hop_features(features: np.ndarray, adj: sp.spmatrix, k: int = 6) -> np.ndarray:
    N, D = features.shape
    adj_dense = adj.toarray().astype(np.float32)
    degree = adj_dense.sum(axis=1)
    degree[degree == 0] = 1
    d_inv_sqrt = np.power(degree, -0.5)
    adj_norm = np.diag(d_inv_sqrt) @ adj_dense @ np.diag(d_inv_sqrt)
    
    hop_features = np.zeros((N, k + 1, D), dtype=np.float32)
    hop_features[:, 0, :] = features
    
    X = features.copy()
    for t in range(k):
        X = adj_norm @ X
        hop_features[:, t + 1, :] = X
        print(f"  Hop {t+1} computed")
    
    return hop_features


def compute_token_strategies(hop_features: np.ndarray) -> Dict[str, np.ndarray]:
    hop_0 = hop_features[:, 0:1, :]
    
    hop_tokens = hop_features.copy()
    
    offset_tokens = hop_features.copy()
    offset_tokens[:, 1:, :] = hop_features[:, 1:, :] - hop_0
    
    delta_tokens = hop_features.copy()
    delta_tokens[:, 1:, :] = hop_features[:, 1:, :] - hop_features[:, :-1, :]
    
    return {"hop": hop_tokens, "offset": offset_tokens, "delta": delta_tokens}


def estimate_entropy_knn_fast(features: np.ndarray, k: int = 5, sample_size: int = 5000) -> float:
    """采样加速的 k-NN 熵估计"""
    N, D = features.shape
    
    if N > sample_size:
        np.random.seed(42)
        indices = np.random.choice(N, sample_size, replace=False)
        features_sample = features[indices]
    else:
        features_sample = features
    
    n_sample = features_sample.shape[0]
    k_actual = min(k, n_sample - 2)
    
    tree = cKDTree(features_sample)
    distances, _ = tree.query(features_sample, k=k_actual + 1)
    r = distances[:, -1]
    r = np.maximum(r, 1e-10)
    
    entropy = digamma(n_sample) - digamma(k_actual) + D * np.mean(np.log(r))
    return float(entropy)


def compute_mutual_info_fast(features: np.ndarray, labels: np.ndarray, sample_size: int = 5000) -> float:
    """采样加速的互信息估计"""
    N = features.shape[0]
    
    if N > sample_size:
        np.random.seed(42)
        indices = np.random.choice(N, sample_size, replace=False)
        features_sample = features[indices]
        labels_sample = labels[indices]
    else:
        features_sample = features
        labels_sample = labels
    
    mi_scores = mutual_info_classif(features_sample, labels_sample, 
                                     n_neighbors=min(5, features_sample.shape[0]-2),
                                     discrete_features=False, random_state=42)
    return float(np.sum(mi_scores))


def compute_fisher_score(features: np.ndarray, labels: np.ndarray) -> float:
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    mean_normal = np.mean(features[normal_mask], axis=0)
    mean_anomaly = np.mean(features[anomaly_mask], axis=0)
    var_normal = np.var(features[normal_mask], axis=0)
    var_anomaly = np.var(features[anomaly_mask], axis=0)
    
    between_class = (mean_normal - mean_anomaly) ** 2
    within_class = var_normal + var_anomaly + 1e-10
    
    return float(np.sum(between_class / within_class))


def analyze_strategy(tokens: np.ndarray, labels: np.ndarray, strategy_name: str, sample_size: int = 5000) -> Dict:
    N, num_tokens, D = tokens.shape
    
    tokens_flat = tokens.reshape(N, -1)
    
    print(f"  计算 {strategy_name} 整体熵 (采样 {sample_size})...")
    entropy_total = estimate_entropy_knn_fast(tokens_flat, k=5, sample_size=sample_size)
    
    print(f"  计算 {strategy_name} 互信息...")
    mi_total = compute_mutual_info_fast(tokens_flat, labels, sample_size=sample_size)
    
    print(f"  计算 {strategy_name} Fisher 分数...")
    fisher_total = compute_fisher_score(tokens_flat, labels)
    
    entropy_per_token = []
    mi_per_token = []
    fisher_per_token = []
    
    for t in range(num_tokens):
        print(f"  计算 Token {t}...")
        token_t = tokens[:, t, :]
        entropy_per_token.append({"token_idx": t, "entropy": estimate_entropy_knn_fast(token_t, k=5, sample_size=sample_size)})
        mi_per_token.append({"token_idx": t, "mi": compute_mutual_info_fast(token_t, labels, sample_size=sample_size)})
        fisher_per_token.append({"token_idx": t, "fisher": compute_fisher_score(token_t, labels)})
    
    return {
        "strategy": strategy_name,
        "shape": [N, num_tokens, D],
        "entropy_total": entropy_total,
        "mi_total": mi_total,
        "fisher_total": fisher_total,
        "entropy_per_token": entropy_per_token,
        "mi_per_token": mi_per_token,
        "fisher_per_token": fisher_per_token
    }


def main():
    parser = argparse.ArgumentParser(description="Token 策略信息论分析（快速版）")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--sample-size", type=int, default=5000, help="熵估计采样大小")
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Token 策略信息论分析（采样加速版）")
    print("=" * 60)
    print(f"参数: alpha=0, 双边归一化, k={args.k}, 采样大小={args.sample_size}")
    
    features, labels, adj = load_dataset(args.dataset)
    
    print(f"\n计算 Hop 特征 (alpha=0, 双边归一化, k={args.k})...")
    hop_features = compute_hop_features(features, adj, args.k)
    print(f"Hop 特征形状: {hop_features.shape}")
    
    print("\n构建三种 Token 策略...")
    token_strategies = compute_token_strategies(hop_features)
    
    print("\n分析 Token 策略...")
    results = {
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "k": args.k,
            "alpha": 0,
            "normalization": "D^{-1/2} A D^{-1/2}",
            "num_tokens": args.k + 1,
            "token_dim": features.shape[1],
            "sample_size": args.sample_size
        },
        "strategies": {}
    }
    
    for name in ["hop", "offset", "delta"]:
        print(f"\n{'='*40}")
        print(f"策略: {name.upper()}")
        print(f"{'='*40}")
        
        stats = analyze_strategy(token_strategies[name], labels, name, args.sample_size)
        results["strategies"][name] = stats
        
        print(f"整体熵: {stats['entropy_total']:.4f}")
        print(f"整体互信息: {stats['mi_total']:.4f}")
        print(f"整体 Fisher 分数: {stats['fisher_total']:.4f}")
    
    print("\n" + "=" * 60)
    print("策略对比")
    print("=" * 60)
    print(f"\n{'策略':<10s} {'熵':>12s} {'互信息':>12s} {'Fisher':>12s}")
    print("-" * 50)
    for name in ["hop", "offset", "delta"]:
        s = results["strategies"][name]
        print(f"{name:<10s} {s['entropy_total']:>12.4f} {s['mi_total']:>12.4f} {s['fisher_total']:>12.4f}")
    
    if args.output is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(output_dir), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"token_strategy_{args.dataset}_fast.json")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {args.output}")


if __name__ == "__main__":
    main()
