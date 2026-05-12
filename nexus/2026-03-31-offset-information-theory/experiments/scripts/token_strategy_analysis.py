#!/usr/bin/env python3
"""
Token 策略信息论分析脚本（修正版）

三种 Token 策略：
- Hop:    [hop_0, hop_1, ..., hop_k]           (K+1)×D
- Offset: [hop_0, hop_1-hop_0, ..., hop_k-hop_0]  (K+1)×D
- Delta:  [hop_0, hop_1-hop_0, hop_2-hop_1, ..., hop_k-hop_{k-1}]  (K+1)×D

参数设置：
- alpha=0: 关闭残差（纯邻居聚合）
- 双边归一化: D^{-1/2} A D^{-1/2}

作者：Nexus
日期：2026-03-31
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
    
    # 加载 .mat 文件
    data = sio.loadmat(data_dir)
    
    # 提取特征
    feature_keys = ['X', 'features', 'Attributes', 'attr', 'feature']
    features = None
    for key in feature_keys:
        if key in data:
            feat = data[key]
            if sp.issparse(feat):
                features = feat.toarray().astype(np.float32)
            else:
                features = np.array(feat, dtype=np.float32)
            break
    
    if features is None:
        raise KeyError(f"找不到特征矩阵，可用键: {list(data.keys())}")
    
    # 提取标签
    label_keys = ['y', 'label', 'Label', 'labels', 'Class', 'str_anomaly_label', 'attr_anomaly_label']
    labels = None
    for key in label_keys:
        if key in data:
            labels = data[key].flatten().astype(np.int32)
            break
    
    if labels is None:
        raise KeyError(f"找不到标签，可用键: {list(data.keys())}")
    
    # 提取邻接矩阵
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


def compute_hop_features(features: np.ndarray, adj: sp.spmatrix, 
                          k: int = 6) -> np.ndarray:
    """
    计算 k-hop 特征（纯邻居聚合，alpha=0，双边归一化）
    
    X^{(t+1)} = D^{-1/2} A D^{-1/2} X^{(t)}
    
    Args:
        features: [N, D] 节点特征
        adj: [N, N] 邻接矩阵
        k: hop 数量
    
    Returns:
        hop_features: [N, K+1, D] hop 特征序列
    """
    N, D = features.shape
    
    # 双边归一化: D^{-1/2} A D^{-1/2}
    adj_dense = adj.toarray().astype(np.float32)
    degree = adj_dense.sum(axis=1)
    degree[degree == 0] = 1  # 避免除零
    d_inv_sqrt = np.power(degree, -0.5)
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = d_mat_inv_sqrt @ adj_dense @ d_mat_inv_sqrt
    
    # 初始化 hop 特征序列
    hop_features = np.zeros((N, k + 1, D), dtype=np.float32)
    hop_features[:, 0, :] = features  # Token 0: hop_0 = 原始特征
    
    X = features.copy()
    for t in range(k):
        # alpha=0: 纯邻居聚合，无残差
        X = adj_norm @ X
        hop_features[:, t + 1, :] = X
    
    return hop_features


def compute_token_strategies(hop_features: np.ndarray) -> Dict[str, np.ndarray]:
    """
    构建三种 Token 策略的特征序列
    
    三种策略都保持 (K+1)×D 维度：
    - Hop:    [hop_0, hop_1, ..., hop_k]
    - Offset: [hop_0, hop_1-hop_0, ..., hop_k-hop_0]
    - Delta:  [hop_0, hop_1-hop_0, hop_2-hop_1, ..., hop_k-hop_{k-1}]
    
    Args:
        hop_features: [N, K+1, D]
    
    Returns:
        dict with three strategies, each [N, K+1, D]
    """
    hop_0 = hop_features[:, 0:1, :]  # [N, 1, D]
    
    # Hop 策略: [hop_0, hop_1, ..., hop_k]
    hop_tokens = hop_features.copy()
    
    # Offset 策略: [hop_0, hop_1-hop_0, ..., hop_k-hop_0]
    # Token 0 保持 hop_0，Token 1~k 改为 hop_t - hop_0
    offset_tokens = hop_features.copy()
    offset_tokens[:, 1:, :] = hop_features[:, 1:, :] - hop_0
    
    # Delta 策略: [hop_0, hop_1-hop_0, hop_2-hop_1, ..., hop_k-hop_{k-1}]
    # Token 0 保持 hop_0，Token t 改为 hop_t - hop_{t-1}
    delta_tokens = hop_features.copy()
    delta_tokens[:, 1:, :] = hop_features[:, 1:, :] - hop_features[:, :-1, :]
    
    return {
        "hop": hop_tokens,
        "offset": offset_tokens,
        "delta": delta_tokens
    }


def estimate_entropy_knn(features: np.ndarray, k: int = 5, 
                           use_full_formula: bool = True) -> float:
    """
    使用 k-NN 方法估计连续特征的信息熵
    
    Kozachenko-Leonenko 估计器:
    H(X) ≈ ψ(N) - ψ(k) + log(c_d) + d * mean(log(r))
    
    Args:
        features: [N, D] 特征矩阵
        k: k-NN 参数
        use_full_formula: 是否使用完整公式（包含 log(c_d)）
    
    Returns:
        entropy: 信息熵 (nat)
    """
    N, D = features.shape
    
    if N <= k + 1:
        k = max(1, N - 2)
    
    # 构建 KD 树
    tree = cKDTree(features)
    
    # 查询 k+1 近邻（包含自身）
    distances, _ = tree.query(features, k=k+1)
    
    # 第 k 近邻距离（跳过自身）
    r = distances[:, -1]
    
    # 避免零距离
    r = np.maximum(r, 1e-10)
    
    # K-L 估计
    if use_full_formula and D <= 100:
        # 完整公式：包含 D 维单位球体积修正
        c_d = np.pi ** (D / 2) / gamma(D / 2 + 1)
        entropy = digamma(N) - digamma(k) + D * np.mean(np.log(r)) + np.log(c_d)
    else:
        # 简化公式：仅用于高维（避免数值溢出）
        entropy = digamma(N) - digamma(k) + D * np.mean(np.log(r))
    
    return float(entropy)


def estimate_entropy_per_token(tokens: np.ndarray, k: int = 5) -> List[Dict]:
    """
    估计每个 Token 位置的信息熵
    
    Args:
        tokens: [N, K+1, D] Token 序列
        k: k-NN 参数
    
    Returns:
        list of dicts: 每个 Token 的熵统计
    """
    N, num_tokens, D = tokens.shape
    results = []
    
    for t in range(num_tokens):
        token_t = tokens[:, t, :]  # [N, D]
        entropy = estimate_entropy_knn(token_t, k=min(k, N-2))
        results.append({
            "token_idx": t,
            "entropy": entropy,
            "dim": D
        })
    
    return results


def compute_mutual_info(features: np.ndarray, labels: np.ndarray,
                         n_neighbors: int = 5) -> float:
    """
    计算特征与标签的互信息
    
    Args:
        features: [N, D] 特征矩阵
        labels: [N] 标签
    
    Returns:
        mi: 互信息 (nat)
    """
    mi_scores = mutual_info_classif(features, labels, 
                                     n_neighbors=min(n_neighbors, features.shape[0]-2),
                                     discrete_features=False, 
                                     random_state=42)
    return float(np.sum(mi_scores))


def compute_fisher_score(features: np.ndarray, labels: np.ndarray) -> float:
    """
    计算 Fisher 分数
    
    Fisher = (μ_1 - μ_0)^2 / (σ_0^2 + σ_1^2)
    
    Args:
        features: [N, D] 特征矩阵
        labels: [N] 标签
    
    Returns:
        fisher: Fisher 总分
    """
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    mean_normal = np.mean(features[normal_mask], axis=0)
    mean_anomaly = np.mean(features[anomaly_mask], axis=0)
    
    var_normal = np.var(features[normal_mask], axis=0)
    var_anomaly = np.var(features[anomaly_mask], axis=0)
    
    between_class = (mean_normal - mean_anomaly) ** 2
    within_class = var_normal + var_anomaly + 1e-10
    
    fisher_scores = between_class / within_class
    
    return float(np.sum(fisher_scores))


def analyze_strategy(tokens: np.ndarray, labels: np.ndarray, 
                      strategy_name: str) -> Dict:
    """
    分析单个 Token 策略
    
    Args:
        tokens: [N, K+1, D] Token 序列
        labels: [N] 标签
        strategy_name: 策略名称
    
    Returns:
        stats: 统计信息字典
    """
    N, num_tokens, D = tokens.shape
    
    # 展平为 [N, (K+1)*D] 进行整体分析
    tokens_flat = tokens.reshape(N, -1)
    
    # 整体熵
    entropy_total = estimate_entropy_knn(tokens_flat)
    
    # 整体互信息
    mi_total = compute_mutual_info(tokens_flat, labels)
    
    # 整体 Fisher 分数
    fisher_total = compute_fisher_score(tokens_flat, labels)
    
    # 每个 Token 的熵
    entropy_per_token = estimate_entropy_per_token(tokens)
    
    # 每个 Token 的互信息
    mi_per_token = []
    for t in range(num_tokens):
        token_t = tokens[:, t, :]
        mi_t = compute_mutual_info(token_t, labels)
        mi_per_token.append({
            "token_idx": t,
            "mi": mi_t
        })
    
    # 每个 Token 的 Fisher 分数
    fisher_per_token = []
    for t in range(num_tokens):
        token_t = tokens[:, t, :]
        fisher_t = compute_fisher_score(token_t, labels)
        fisher_per_token.append({
            "token_idx": t,
            "fisher": fisher_t
        })
    
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
    parser = argparse.ArgumentParser(description="Token 策略信息论分析")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["photo", "elliptic", "tolokers"],
                        help="数据集名称")
    parser.add_argument("--k", type=int, default=6, help="Hop 数量 (Token 数量 = k+1)")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Token 策略信息论分析（修正版）")
    print("=" * 60)
    print(f"参数: alpha=0, 双边归一化, k={args.k}")
    
    # 加载数据集
    features, labels, adj = load_dataset(args.dataset)
    
    # 计算 hop 特征
    print(f"\n计算 Hop 特征 (alpha=0, 双边归一化, k={args.k})...")
    hop_features = compute_hop_features(features, adj, args.k)
    print(f"Hop 特征形状: {hop_features.shape}")
    
    # 构建三种 Token 策略
    print("\n构建三种 Token 策略...")
    token_strategies = compute_token_strategies(hop_features)
    
    # 分析每种策略
    print("\n分析 Token 策略...")
    results = {
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "k": args.k,
            "alpha": 0,
            "normalization": "D^{-1/2} A D^{-1/2}",
            "num_tokens": args.k + 1,
            "token_dim": features.shape[1]
        },
        "strategies": {}
    }
    
    for name, tokens in token_strategies.items():
        print(f"\n{'='*40}")
        print(f"策略: {name.upper()}")
        print(f"{'='*40}")
        
        stats = analyze_strategy(tokens, labels, name)
        results["strategies"][name] = stats
        
        print(f"整体熵: {stats['entropy_total']:.4f}")
        print(f"整体互信息: {stats['mi_total']:.4f}")
        print(f"整体 Fisher 分数: {stats['fisher_total']:.4f}")
        
        print(f"\n每 Token 信息熵:")
        for t_info in stats['entropy_per_token']:
            print(f"  Token {t_info['token_idx']}: {t_info['entropy']:.4f}")
    
    # 对比分析
    print("\n" + "=" * 60)
    print("策略对比")
    print("=" * 60)
    
    print(f"\n{'策略':<10s} {'熵':>12s} {'互信息':>12s} {'Fisher':>12s}")
    print("-" * 50)
    for name in ["hop", "offset", "delta"]:
        s = results["strategies"][name]
        print(f"{name:<10s} {s['entropy_total']:>12.4f} {s['mi_total']:>12.4f} {s['fisher_total']:>12.4f}")
    
    # 保存结果
    if args.output is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(output_dir), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"token_strategy_{args.dataset}.json")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {args.output}")


if __name__ == "__main__":
    main()