#!/usr/bin/env python3
"""
Offset 互信息分析脚本

功能：
1. 计算特征与标签的互信息
2. 计算特征之间的冗余度
3. 计算信息增益

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
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import KernelDensity
import scipy.io as sio
import scipy.sparse as sp

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入共享函数
from offset_entropy_analysis import (
    NumpyEncoder, load_dataset, compute_hop_features,
    compute_offset, compute_delta
)

# 确保所有脚本使用相同的路径修复
import scipy.sparse as sp


def estimate_mutual_info_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """
    估计离散变量的互信息
    
    Args:
        x: [N] 离散变量
        y: [N] 离散变量
    
    Returns:
        mi: 互信息 (nat)
    """
    # 使用 sklearn 的互信息计算
    mi = mutual_info_score(x, y)
    return float(mi)


def estimate_mutual_info_continuous(features: np.ndarray, labels: np.ndarray,
                                     n_neighbors: int = 5) -> float:
    """
    估计连续特征与离散标签的互信息
    
    使用 k-NN 方法
    
    Args:
        features: [N, D] 连续特征
        labels: [N] 离散标签
        n_neighbors: k-NN 参数
    
    Returns:
        mi: 互信息 (nat)
    """
    # 使用 sklearn 的 mutual_info_classif
    mi_scores = mutual_info_classif(features, labels, n_neighbors=n_neighbors,
                                     discrete_features=False, random_state=42)
    
    # 总互信息（各维度之和）
    mi_total = np.sum(mi_scores)
    
    return float(mi_total)


def estimate_mutual_info_continuous_continuous(x: np.ndarray, y: np.ndarray,
                                                n_neighbors: int = 5) -> float:
    """
    估计两个连续变量的互信息
    
    使用 k-NN 方法
    
    Args:
        x: [N, D1] 连续特征
        y: [N, D2] 连续特征
        n_neighbors: k-NN 参数
    
    Returns:
        mi: 互信息 (nat)
    """
    # 确保是 2D
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    # 使用 mutual_info_regression
    # 取 y 的第一维作为目标
    y_target = y[:, 0] if y.shape[1] > 0 else y.flatten()
    
    mi = mutual_info_regression(x, y_target, n_neighbors=n_neighbors,
                                 random_state=42)
    
    return float(np.sum(mi))


def compute_mi_per_dimension(features: np.ndarray, labels: np.ndarray,
                              n_neighbors: int = 5) -> np.ndarray:
    """
    计算每个维度与标签的互信息
    
    Args:
        features: [N, D] 特征矩阵
        labels: [N] 标签
        n_neighbors: k-NN 参数
    
    Returns:
        mi_per_dim: [D] 每个维度的互信息
    """
    mi_per_dim = mutual_info_classif(features, labels, n_neighbors=n_neighbors,
                                       discrete_features=False, random_state=42)
    return mi_per_dim


def analyze_mutual_information(features: np.ndarray, labels: np.ndarray,
                                feature_name: str, 
                                reference_features: np.ndarray = None,
                                reference_name: str = None) -> Dict:
    """
    分析特征的互信息
    
    Args:
        features: [N, D] 或 [N, K, D] 特征矩阵
        labels: [N] 标签
        feature_name: 特征名称
        reference_features: 参考特征（计算冗余度）
        reference_name: 参考特征名称
    
    Returns:
        stats: 统计信息字典
    """
    # 处理 3D 特征
    if len(features.shape) == 3:
        features = features.reshape(features.shape[0], -1)
    
    N, D = features.shape
    
    # 计算与标签的互信息
    print(f"   计算 {feature_name} 与标签的互信息...")
    mi_label = estimate_mutual_info_continuous(features, labels)
    
    # 计算每个维度的互信息
    print(f"   计算每个维度的互信息...")
    mi_per_dim = compute_mi_per_dimension(features, labels)
    
    # 排序找出最重要的维度
    sorted_indices = np.argsort(mi_per_dim)[::-1]
    
    stats = {
        "feature_name": feature_name,
        "N": N,
        "D": D,
        "mi_label_total": mi_label,
        "mi_label_per_dim_mean": float(np.mean(mi_per_dim)),
        "mi_label_per_dim_std": float(np.std(mi_per_dim)),
        "mi_label_per_dim_max": float(np.max(mi_per_dim)),
        "top_10_dim": [int(x) for x in sorted_indices[:10]],
        "top_10_mi": [float(mi_per_dim[x]) for x in sorted_indices[:10]],
    }
    
    # 如果有参考特征，计算冗余度
    if reference_features is not None:
        if len(reference_features.shape) == 3:
            reference_features = reference_features.reshape(reference_features.shape[0], -1)
        
        # 简化：只计算采样的冗余度
        print(f"   计算与 {reference_name} 的冗余度...")
        
        # 采样维度以加速
        sample_size = min(50, D, reference_features.shape[1])
        feat_sample = features[:, :sample_size]
        ref_sample = reference_features[:, :sample_size]
        
        # 计算互信息（近似）
        # 使用第一维作为代表
        mi_redundancy = estimate_mutual_info_continuous_continuous(
            feat_sample, ref_sample
        )
        
        stats["mi_redundancy"] = float(mi_redundancy)
        stats["reference_name"] = reference_name
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Offset 互信息分析")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["photo", "elliptic", "tolokers"],
                        help="数据集名称")
    parser.add_argument("--pp_k", type=int, default=6, help="Hop 数量")
    parser.add_argument("--alpha", type=float, default=0.1, help="PPR 参数")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Offset 互信息分析")
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
    
    # 分析各种特征的互信息
    print("\n分析互信息...")
    
    results = {
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "pp_k": args.pp_k,
            "alpha": args.alpha,
        },
        "mutual_information_analysis": {}
    }
    
    # 1. 原始特征的互信息
    print("\n1. 原始特征互信息...")
    stats_original = analyze_mutual_information(
        features, labels, "original"
    )
    results["mutual_information_analysis"]["original"] = stats_original
    print(f"   I(original; label) = {stats_original['mi_label_total']:.4f}")
    
    # 2. Offset 的互信息
    print("\n2. Offset 互信息...")
    stats_offset = analyze_mutual_information(
        offsets, labels, "offset",
        reference_features=features, reference_name="original"
    )
    results["mutual_information_analysis"]["offset"] = stats_offset
    print(f"   I(offset; label) = {stats_offset['mi_label_total']:.4f}")
    print(f"   I(offset; original) = {stats_offset.get('mi_redundancy', 0):.4f}")
    
    # 3. Delta 的互信息
    print("\n3. Delta 互信息...")
    stats_delta = analyze_mutual_information(
        deltas, labels, "delta",
        reference_features=features, reference_name="original"
    )
    results["mutual_information_analysis"]["delta"] = stats_delta
    print(f"   I(delta; label) = {stats_delta['mi_label_total']:.4f}")
    
    # 4. Hop 特征的互信息
    print("\n4. Hop 特征互信息...")
    stats_hop = analyze_mutual_information(
        hop_features, labels, "hop_features"
    )
    results["mutual_information_analysis"]["hop_features"] = stats_hop
    print(f"   I(hop; label) = {stats_hop['mi_label_total']:.4f}")
    
    # 计算信息增益
    print("\n" + "=" * 60)
    print("信息增益分析")
    print("=" * 60)
    
    mi_original = results["mutual_information_analysis"]["original"]["mi_label_total"]
    mi_offset = results["mutual_information_analysis"]["offset"]["mi_label_total"]
    mi_delta = results["mutual_information_analysis"]["delta"]["mi_label_total"]
    mi_hop = results["mutual_information_analysis"]["hop_features"]["mi_label_total"]
    
    info_gain_offset = mi_offset - mi_original
    info_gain_delta = mi_delta - mi_original
    info_gain_hop = mi_hop - mi_original
    
    print(f"{'特征':<15s}: {'I(X;label)':>12s} {'信息增益':>12s}")
    print("-" * 45)
    print(f"{'original':<15s}: {mi_original:>12.4f} {'-':>12s}")
    print(f"{'offset':<15s}: {mi_offset:>12.4f} {info_gain_offset:>+12.4f}")
    print(f"{'delta':<15s}: {mi_delta:>12.4f} {info_gain_delta:>+12.4f}")
    print(f"{'hop_features':<15s}: {mi_hop:>12.4f} {info_gain_hop:>+12.4f}")
    
    # 冗余度分析
    redundancy_offset = results["mutual_information_analysis"]["offset"].get("mi_redundancy", 0)
    redundancy_delta = results["mutual_information_analysis"]["delta"].get("mi_redundancy", 0)
    
    print("\n" + "=" * 60)
    print("冗余度分析")
    print("=" * 60)
    print(f"I(offset; original) = {redundancy_offset:.4f}")
    print(f"I(delta; original) = {redundancy_delta:.4f}")
    
    # 汇总结果
    results["summary"] = {
        "mi_original_label": mi_original,
        "mi_offset_label": mi_offset,
        "mi_delta_label": mi_delta,
        "info_gain_offset": info_gain_offset,
        "info_gain_delta": info_gain_delta,
        "redundancy_offset_original": redundancy_offset,
        "redundancy_delta_original": redundancy_delta,
    }
    
    # 保存结果
    if args.output is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(output_dir), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"mutual_info_{args.dataset}.json")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {args.output}")


if __name__ == "__main__":
    main()