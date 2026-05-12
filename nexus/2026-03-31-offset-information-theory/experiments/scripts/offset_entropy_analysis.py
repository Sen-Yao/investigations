#!/usr/bin/env python3
"""
Offset 信息熵分析脚本

功能：
1. 计算原始 Hop 特征的信息熵
2. 计算 Offset 特征的信息熵
3. 计算 Delta 特征的信息熵
4. 对比不同特征类型的信息量

作者：Nexus
日期：2026-03-31
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
import scipy.io as sio
import scipy.sparse as sp

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Numpy Encoder for JSON
class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码器"""
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
    """
    加载数据集
    
    Args:
        dataset_name: 数据集名称 (photo, elliptic, tolokers)
    
    Returns:
        features: [N, D] 节点特征
        labels: [N] 标签 (0=正常, 1=异常)
        adj: [N, N] 邻接矩阵
    """
    # 数据集路径
    # 尝试多个可能的位置
    possible_paths = [
        f"~/gpufree-data/linziyao/VoxG/dataset/{dataset_name.capitalize()}.mat",
        f"~/gpufree-data/linziyao/VoxG/dataset/{dataset_name}.mat",
        f"~/VoxG/dataset/{dataset_name.capitalize()}.mat",
        f"~/VoxG/dataset/{dataset_name}.mat",
    ]
    
    data_dir = None
    for path in possible_paths:
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            data_dir = expanded
            break
    
    if data_dir is None:
        raise FileNotFoundError(f"数据集不存在，尝试路径: {possible_paths}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据集不存在: {data_dir}")
    
    # 加载 .mat 文件
    data = sio.loadmat(data_dir)
    
    # 提取特征和标签
    # 尝试多种可能的键名
    feature_keys = ['X', 'features', 'Attributes', 'attr', 'feature']
    features = None
    for key in feature_keys:
        if key in data:
            feat = data[key]
            # 处理稀疏矩阵
            if sp.issparse(feat):
                features = feat.toarray().astype(np.float32)
            else:
                features = np.array(feat, dtype=np.float32)
            break
    
    if features is None:
        raise KeyError(f"找不到特征矩阵，可用键: {list(data.keys())}")
    
    # 标签
    label_keys = ['y', 'label', 'Label', 'labels', 'Class', 'str_anomaly_label', 'attr_anomaly_label']
    labels = None
    for key in label_keys:
        if key in data:
            labels = data[key].flatten().astype(np.int32)
            break
    
    if labels is None:
        raise KeyError(f"找不到标签，可用键: {list(data.keys())}")
    
    # 邻接矩阵
    adj_keys = ['A', 'adj', 'Network', 'network', 'adjacency']
    adj = None
    for key in adj_keys:
        if key in data:
            adj = data[key]
            break
    
    if adj is None:
        raise KeyError(f"找不到邻接矩阵，可用键: {list(data.keys())}")
    
    # 转换为稀疏矩阵
    if sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    else:
        adj = sp.csr_matrix(adj)
    
    # 标签转换为二值（假设非零为异常）
    labels = (labels != 0).astype(np.int32)
    
    print(f"数据集: {dataset_name}")
    print(f"节点数: {features.shape[0]}")
    print(f"特征维度: {features.shape[1]}")
    print(f"正常节点: {(labels == 0).sum()}")
    print(f"异常节点: {(labels == 1).sum()}")
    
    return features, labels, adj


def compute_hop_features(features: np.ndarray, adj: sp.spmatrix, pp_k: int = 6, 
                          alpha: float = 0.1) -> np.ndarray:
    """
    计算 k-hop 特征 (PPR 风格聚合)
    
    X^{(k+1)} = (1-α) * A_norm @ X^{(k)} + α * X^{(0)}
    
    Args:
        features: [N, D] 节点特征
        adj: [N, N] 邻接矩阵
        pp_k: hop 数量
        alpha: PPR 参数
    
    Returns:
        hop_features: [N, K+1, D] hop 特征
    """
    N, D = features.shape
    
    # 转换为 dense 数组
    if sp.issparse(adj):
        adj_dense = adj.toarray().astype(np.float32)
    else:
        adj_dense = np.array(adj, dtype=np.float32)
    degree = adj_dense.sum(axis=1)
    degree[degree == 0] = 1
    d_inv_sqrt = np.power(degree, -0.5)
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = d_mat_inv_sqrt @ adj_dense @ d_mat_inv_sqrt
    
    # 初始化 hop 特征
    hop_features = np.zeros((N, pp_k + 1, D), dtype=np.float32)
    hop_features[:, 0, :] = features  # hop_0 = 原始特征
    
    X = features.copy()
    for k in range(pp_k):
        # PPR 风格聚合
        X = (1 - alpha) * adj_norm @ X + alpha * features
        hop_features[:, k + 1, :] = X
    
    return hop_features


def compute_offset(hop_features: np.ndarray) -> np.ndarray:
    """
    计算 Offset: offset_k = hop_k - hop_0
    
    Args:
        hop_features: [N, K+1, D] hop 特征
    
    Returns:
        offsets: [N, K, D] Offset 特征
    """
    hop_0 = hop_features[:, 0:1, :]  # [N, 1, D]
    offsets = hop_features[:, 1:, :] - hop_0  # [N, K, D]
    return offsets


def compute_delta(hop_features: np.ndarray) -> np.ndarray:
    """
    计算 Delta: delta_k = hop_k - hop_{k-1}
    
    Args:
        hop_features: [N, K+1, D] hop 特征
    
    Returns:
        deltas: [N, K, D] Delta 特征
    """
    deltas = hop_features[:, 1:, :] - hop_features[:, :-1, :]  # [N, K, D]
    return deltas


def estimate_entropy_knn(features: np.ndarray, k: int = 5, 
                           max_dim: int = 50) -> float:
    """
    使用 k-NN 方法估计连续特征的信息熵
    
    方法：Kozachenko-Leonenko 估计器
    
    H(X) ≈ ψ(N) - ψ(k) + log(c_d) + (d/N) * Σ log(r_i)
    
    其中 r_i 是第 k 近邻距离
    
    注意：对于高维特征，使用采样维度
    
    Args:
        features: [N, D] 特征矩阵
        k: k-NN 参数
        max_dim: 最大维度（超过则采样）
    
    Returns:
        entropy: 信息熵 (nat)
    """
    N, D = features.shape
    
    if N <= k:
        k = max(1, N - 1)
    
    # 对于高维特征，采样维度
    if D > max_dim:
        # 随机采样 max_dim 个维度
        np.random.seed(42)
        sample_dims = np.random.choice(D, max_dim, replace=False)
        features = features[:, sample_dims]
        D = max_dim
    
    # 构建 KD 树
    tree = cKDTree(features)
    
    # 查询 k+1 近邻（包含自身）
    distances, _ = tree.query(features, k=k+1)
    
    # 第 k 近邻距离（跳过自身，所以是第 k+1 个）
    r = distances[:, -1]
    
    # 避免零距离
    r = np.maximum(r, 1e-10)
    
    # Kozachenko-Leonenko 估计
    # 对于高维，c_d 会溢出，改用简化的估计
    # H ≈ ψ(N) - ψ(k) + D * mean(log(r))
    entropy = digamma(N) - digamma(k) + D * np.mean(np.log(r))
    
    return float(entropy)


def compute_entropy_per_dimension(features: np.ndarray, k: int = 5) -> np.ndarray:
    """
    计算每个维度的信息熵
    
    Args:
        features: [N, D] 特征矩阵
        k: k-NN 参数
    
    Returns:
        entropy_per_dim: [D] 每个维度的信息熵
    """
    N, D = features.shape
    entropy_per_dim = np.zeros(D)
    
    for d in range(D):
        # 单维度特征
        feat_d = features[:, d:d+1]
        entropy_per_dim[d] = estimate_entropy_knn(feat_d, k=min(k, N-1))
    
    return entropy_per_dim


def analyze_entropy(features: np.ndarray, labels: np.ndarray, 
                    feature_name: str) -> Dict:
    """
    分析特征的信息熵
    
    Args:
        features: [N, D] 或 [N, K, D] 特征矩阵
        labels: [N] 标签
        feature_name: 特征名称
    
    Returns:
        stats: 统计信息字典
    """
    # 处理 3D 特征
    if len(features.shape) == 3:
        features = features.reshape(features.shape[0], -1)
    
    N, D = features.shape
    
    # 整体熵
    entropy_total = estimate_entropy_knn(features)
    
    # 正常节点熵
    normal_mask = labels == 0
    if normal_mask.sum() > 5:
        entropy_normal = estimate_entropy_knn(features[normal_mask])
    else:
        entropy_normal = 0.0
    
    # 异常节点熵
    anomaly_mask = labels == 1
    if anomaly_mask.sum() > 5:
        entropy_anomaly = estimate_entropy_knn(features[anomaly_mask])
    else:
        entropy_anomaly = 0.0
    
    # 每个维度的熵（采样以加速）
    if D > 50:
        # 采样维度
        sample_dims = np.random.choice(D, min(50, D), replace=False)
        entropy_per_dim_sample = compute_entropy_per_dimension(features[:, sample_dims])
        entropy_per_dim = np.zeros(D)
        entropy_per_dim[sample_dims] = entropy_per_dim_sample
    else:
        entropy_per_dim = compute_entropy_per_dimension(features)
    
    return {
        "feature_name": feature_name,
        "N": N,
        "D": D,
        "entropy_total": entropy_total,
        "entropy_normal": entropy_normal,
        "entropy_anomaly": entropy_anomaly,
        "entropy_per_dim_mean": float(np.mean(entropy_per_dim)),
        "entropy_per_dim_std": float(np.std(entropy_per_dim)),
        "entropy_per_dim_max": float(np.max(entropy_per_dim)),
        "entropy_per_dim_min": float(np.min(entropy_per_dim)),
        "top_10_dim": [int(x) for x in np.argsort(entropy_per_dim)[-10:][::-1]],
    }


def main():
    parser = argparse.ArgumentParser(description="Offset 信息熵分析")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["photo", "elliptic", "tolokers"],
                        help="数据集名称")
    parser.add_argument("--pp_k", type=int, default=6, help="Hop 数量")
    parser.add_argument("--alpha", type=float, default=0.1, help="PPR 参数")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Offset 信息熵分析")
    print("=" * 60)
    
    # 加载数据集
    features, labels, adj = load_dataset(args.dataset)
    
    # 计算 hop 特征
    print("\n计算 Hop 特征...")
    hop_features = compute_hop_features(features, adj, args.pp_k, args.alpha)
    
    # 计算 Offset
    print("计算 Offset...")
    offsets = compute_offset(hop_features)
    
    # 计算 Delta
    print("计算 Delta...")
    deltas = compute_delta(hop_features)
    
    # 分析各种特征的熵
    print("\n分析信息熵...")
    
    results = {
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "pp_k": args.pp_k,
            "alpha": args.alpha,
        },
        "entropy_analysis": {}
    }
    
    # 1. 原始特征熵
    print("\n1. 原始特征熵...")
    stats_original = analyze_entropy(features, labels, "original")
    results["entropy_analysis"]["original"] = stats_original
    print(f"   总熵: {stats_original['entropy_total']:.4f}")
    print(f"   正常节点熵: {stats_original['entropy_normal']:.4f}")
    print(f"   异常节点熵: {stats_original['entropy_anomaly']:.4f}")
    
    # 2. Hop 特征熵（展平）
    print("\n2. Hop 特征熵...")
    hop_flat = hop_features.reshape(hop_features.shape[0], -1)
    stats_hop = analyze_entropy(hop_flat, labels, "hop_features")
    results["entropy_analysis"]["hop_features"] = stats_hop
    print(f"   总熵: {stats_hop['entropy_total']:.4f}")
    
    # 3. Offset 熵
    print("\n3. Offset 熵...")
    offset_flat = offsets.reshape(offsets.shape[0], -1)
    stats_offset = analyze_entropy(offset_flat, labels, "offset")
    results["entropy_analysis"]["offset"] = stats_offset
    print(f"   总熵: {stats_offset['entropy_total']:.4f}")
    print(f"   正常节点熵: {stats_offset['entropy_normal']:.4f}")
    print(f"   异常节点熵: {stats_offset['entropy_anomaly']:.4f}")
    
    # 4. Delta 熵
    print("\n4. Delta 熵...")
    delta_flat = deltas.reshape(deltas.shape[0], -1)
    stats_delta = analyze_entropy(delta_flat, labels, "delta")
    results["entropy_analysis"]["delta"] = stats_delta
    print(f"   总熵: {stats_delta['entropy_total']:.4f}")
    
    # 对比分析
    print("\n" + "=" * 60)
    print("信息熵对比")
    print("=" * 60)
    print(f"{'特征类型':<15s}: {'总熵':>10s} {'正常熵':>10s} {'异常熵':>10s}")
    print("-" * 50)
    for name in ["original", "hop_features", "offset", "delta"]:
        stats = results["entropy_analysis"][name]
        print(f"{name:<15s}: {stats['entropy_total']:>10.4f} "
              f"{stats['entropy_normal']:>10.4f} {stats['entropy_anomaly']:>10.4f}")
    
    # 信息增益
    print("\n" + "=" * 60)
    print("信息增益分析 (相对于原始特征)")
    print("=" * 60)
    H_original = results["entropy_analysis"]["original"]["entropy_total"]
    for name in ["hop_features", "offset", "delta"]:
        H_feature = results["entropy_analysis"][name]["entropy_total"]
        info_gain = H_feature - H_original
        print(f"{name}: {info_gain:+.4f} nat")
    
    # 保存结果
    if args.output is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(output_dir), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"entropy_{args.dataset}.json")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {args.output}")


if __name__ == "__main__":
    main()