#!/usr/bin/env python3
"""
Offset Fisher 判别分析脚本

功能：
1. 计算每个维度的 Fisher 分数
2. 分析正常/异常节点的类间可分性
3. 找出最具区分力的维度

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
import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入共享函数
from offset_entropy_analysis import (
    NumpyEncoder, load_dataset, compute_hop_features,
    compute_offset, compute_delta
)


def compute_fisher_score(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    计算每个维度的 Fisher 分数
    
    Fisher Score = (μ_1 - μ_0)^2 / (σ_0^2 + σ_1^2)
    
    Args:
        features: [N, D] 特征矩阵
        labels: [N] 标签 (0=正常, 1=异常)
    
    Returns:
        fisher_scores: [D] Fisher 分数
    """
    D = features.shape[1]
    
    # 分离正常和异常节点
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    # 计算均值
    mean_normal = np.mean(features[normal_mask], axis=0)  # [D]
    mean_anomaly = np.mean(features[anomaly_mask], axis=0)  # [D]
    
    # 计算方差
    var_normal = np.var(features[normal_mask], axis=0)  # [D]
    var_anomaly = np.var(features[anomaly_mask], axis=0)  # [D]
    
    # Fisher 分数
    between_class = (mean_normal - mean_anomaly) ** 2
    within_class = var_normal + var_anomaly + 1e-10
    
    fisher_scores = between_class / within_class
    
    return fisher_scores


def compute_multiclass_fisher_score(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    计算多类 Fisher 分数（更稳健的版本）
    
    Fisher Score = Σ n_c * (μ_c - μ)^2 / Σ n_c * σ_c^2
    
    Args:
        features: [N, D] 特征矩阵
        labels: [N] 标签
    
    Returns:
        fisher_scores: [D] Fisher 分数
    """
    N, D = features.shape
    
    # 全局均值
    global_mean = np.mean(features, axis=0)  # [D]
    
    # 分离正常和异常节点
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    n_normal = normal_mask.sum()
    n_anomaly = anomaly_mask.sum()
    
    # 类均值
    mean_normal = np.mean(features[normal_mask], axis=0)
    mean_anomaly = np.mean(features[anomaly_mask], axis=0)
    
    # 类方差
    var_normal = np.var(features[normal_mask], axis=0)
    var_anomaly = np.var(features[anomaly_mask], axis=0)
    
    # 分子：类间方差
    between_class = (n_normal * (mean_normal - global_mean) ** 2 + 
                     n_anomaly * (mean_anomaly - global_mean) ** 2)
    
    # 分母：类内方差
    within_class = n_normal * var_normal + n_anomaly * var_anomaly + 1e-10
    
    fisher_scores = between_class / within_class
    
    return fisher_scores


def analyze_fisher(features: np.ndarray, labels: np.ndarray,
                    feature_name: str) -> Dict:
    """
    分析特征的 Fisher 判别能力
    
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
    
    # 分离正常和异常节点
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    n_normal = normal_mask.sum()
    n_anomaly = anomaly_mask.sum()
    
    # 计算 Fisher 分数
    fisher_scores = compute_fisher_score(features, labels)
    
    # 排序
    sorted_indices = np.argsort(fisher_scores)[::-1]
    
    # 计算均值差异
    mean_normal = np.mean(features[normal_mask], axis=0)
    mean_anomaly = np.mean(features[anomaly_mask], axis=0)
    mean_diff = mean_anomaly - mean_normal
    
    # 计算方差比率
    var_normal = np.var(features[normal_mask], axis=0)
    var_anomaly = np.var(features[anomaly_mask], axis=0)
    var_ratio = var_anomaly / (var_normal + 1e-10)
    
    stats = {
        "feature_name": feature_name,
        "N": N,
        "D": D,
        "n_normal": int(n_normal),
        "n_anomaly": int(n_anomaly),
        "fisher_score_mean": float(np.mean(fisher_scores)),
        "fisher_score_std": float(np.std(fisher_scores)),
        "fisher_score_max": float(np.max(fisher_scores)),
        "fisher_score_sum": float(np.sum(fisher_scores)),
        "top_10_dim": [int(x) for x in sorted_indices[:10]],
        "top_10_fisher": [float(fisher_scores[x]) for x in sorted_indices[:10]],
        "mean_diff_top10": [float(mean_diff[x]) for x in sorted_indices[:10]],
        "var_ratio_top10": [float(var_ratio[x]) for x in sorted_indices[:10]],
    }
    
    return stats


def plot_fisher_comparison(results: Dict, output_path: str):
    """
    绘制 Fisher 分数对比图
    
    Args:
        results: 分析结果
        output_path: 输出路径
    """
    feature_names = ["original", "offset", "delta"]
    fisher_means = []
    fisher_sums = []
    
    for name in feature_names:
        if name in results["fisher_analysis"]:
            fisher_means.append(results["fisher_analysis"][name]["fisher_score_mean"])
            fisher_sums.append(results["fisher_analysis"][name]["fisher_score_sum"])
        else:
            fisher_means.append(0)
            fisher_sums.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 子图 1: 平均 Fisher 分数
    ax1 = axes[0]
    x = np.arange(len(feature_names))
    bars = ax1.bar(x, fisher_means, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax1.set_xlabel('Feature Type')
    ax1.set_ylabel('Mean Fisher Score')
    ax1.set_title('Mean Fisher Score by Feature Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_names)
    
    # 添加数值标签
    for bar, val in zip(bars, fisher_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 子图 2: 总 Fisher 分数
    ax2 = axes[1]
    bars = ax2.bar(x, fisher_sums, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax2.set_xlabel('Feature Type')
    ax2.set_ylabel('Total Fisher Score')
    ax2.set_title('Total Fisher Score by Feature Type')
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_names)
    
    for bar, val in zip(bars, fisher_sums):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   图表已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Offset Fisher 判别分析")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["photo", "elliptic", "tolokers"],
                        help="数据集名称")
    parser.add_argument("--pp_k", type=int, default=6, help="Hop 数量")
    parser.add_argument("--alpha", type=float, default=0.1, help="PPR 参数")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--plot", action="store_true", help="是否生成图表")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Offset Fisher 判别分析")
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
    
    # 分析各种特征的 Fisher 分数
    print("\n分析 Fisher 判别能力...")
    
    results = {
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "pp_k": args.pp_k,
            "alpha": args.alpha,
        },
        "fisher_analysis": {}
    }
    
    # 1. 原始特征
    print("\n1. 原始特征 Fisher 分析...")
    stats_original = analyze_fisher(features, labels, "original")
    results["fisher_analysis"]["original"] = stats_original
    print(f"   平均 Fisher 分数: {stats_original['fisher_score_mean']:.4f}")
    print(f"   最大 Fisher 分数: {stats_original['fisher_score_max']:.4f}")
    
    # 2. Offset
    print("\n2. Offset Fisher 分析...")
    stats_offset = analyze_fisher(offsets, labels, "offset")
    results["fisher_analysis"]["offset"] = stats_offset
    print(f"   平均 Fisher 分数: {stats_offset['fisher_score_mean']:.4f}")
    print(f"   最大 Fisher 分数: {stats_offset['fisher_score_max']:.4f}")
    
    # 3. Delta
    print("\n3. Delta Fisher 分析...")
    stats_delta = analyze_fisher(deltas, labels, "delta")
    results["fisher_analysis"]["delta"] = stats_delta
    print(f"   平均 Fisher 分数: {stats_delta['fisher_score_mean']:.4f}")
    print(f"   最大 Fisher 分数: {stats_delta['fisher_score_max']:.4f}")
    
    # 对比分析
    print("\n" + "=" * 60)
    print("Fisher 分数对比")
    print("=" * 60)
    print(f"{'特征类型':<15s}: {'平均分数':>12s} {'总分数':>12s} {'最大分数':>12s}")
    print("-" * 55)
    for name in ["original", "offset", "delta"]:
        stats = results["fisher_analysis"][name]
        print(f"{name:<15s}: {stats['fisher_score_mean']:>12.4f} "
              f"{stats['fisher_score_sum']:>12.2f} {stats['fisher_score_max']:>12.4f}")
    
    # 信息增益计算
    fs_original = results["fisher_analysis"]["original"]["fisher_score_sum"]
    fs_offset = results["fisher_analysis"]["offset"]["fisher_score_sum"]
    fs_delta = results["fisher_analysis"]["delta"]["fisher_score_sum"]
    
    print("\n" + "=" * 60)
    print("Fisher 分数增益")
    print("=" * 60)
    print(f"Offset 增益: {(fs_offset - fs_original) / fs_original * 100:+.2f}%")
    print(f"Delta 增益: {(fs_delta - fs_original) / fs_original * 100:+.2f}%")
    
    # 汇总
    results["summary"] = {
        "fisher_sum_original": fs_original,
        "fisher_sum_offset": fs_offset,
        "fisher_sum_delta": fs_delta,
        "fisher_gain_offset_pct": (fs_offset - fs_original) / fs_original * 100,
        "fisher_gain_delta_pct": (fs_delta - fs_original) / fs_original * 100,
    }
    
    # 绘制图表
    if args.plot:
        print("\n生成图表...")
        plot_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(os.path.dirname(plot_dir), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"fisher_comparison_{args.dataset}.png")
        plot_fisher_comparison(results, plot_path)
    
    # 保存结果
    if args.output is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(output_dir), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"fisher_{args.dataset}.json")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {args.output}")


if __name__ == "__main__":
    main()