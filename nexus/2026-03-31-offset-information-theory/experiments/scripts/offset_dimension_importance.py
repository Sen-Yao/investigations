#!/usr/bin/env python3
"""
Offset 维度重要性分析脚本

功能：
1. 分析每个维度的信息贡献
2. 找出最重要的维度
3. 生成可视化

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
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.sparse as sp

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入共享函数
from offset_entropy_analysis import (
    NumpyEncoder, load_dataset, compute_hop_features,
    compute_offset, compute_delta
)


def compute_dimension_importance(features: np.ndarray, labels: np.ndarray,
                                   method: str = "mutual_info") -> np.ndarray:
    """
    计算每个维度的重要性
    
    Args:
        features: [N, D] 特征矩阵
        labels: [N] 标签
        method: 计算方法 (mutual_info, variance, fisher)
    
    Returns:
        importance: [D] 重要性分数
    """
    if method == "mutual_info":
        # 互信息
        importance = mutual_info_classif(features, labels, 
                                          n_neighbors=5, 
                                          discrete_features=False,
                                          random_state=42)
    elif method == "variance":
        # 方差（绝对差异）
        normal_mask = labels == 0
        anomaly_mask = labels == 1
        
        mean_normal = np.mean(features[normal_mask], axis=0)
        mean_anomaly = np.mean(features[anomaly_mask], axis=0)
        
        importance = np.abs(mean_anomaly - mean_normal)
    elif method == "fisher":
        # Fisher 分数
        normal_mask = labels == 0
        anomaly_mask = labels == 1
        
        mean_normal = np.mean(features[normal_mask], axis=0)
        mean_anomaly = np.mean(features[anomaly_mask], axis=0)
        
        var_normal = np.var(features[normal_mask], axis=0)
        var_anomaly = np.var(features[anomaly_mask], axis=0)
        
        between_class = (mean_normal - mean_anomaly) ** 2
        within_class = var_normal + var_anomaly + 1e-10
        
        importance = between_class / within_class
    else:
        raise ValueError(f"未知方法: {method}")
    
    return importance


def analyze_dimension_importance(features: np.ndarray, labels: np.ndarray,
                                   feature_name: str, 
                                   methods: List[str] = ["mutual_info", "fisher"]) -> Dict:
    """
    分析维度重要性
    
    Args:
        features: [N, D] 或 [N, K, D] 特征矩阵
        labels: [N] 标签
        feature_name: 特征名称
        methods: 计算方法列表
    
    Returns:
        stats: 统计信息字典
    """
    # 处理 3D 特征
    original_shape = features.shape
    if len(features.shape) == 3:
        features = features.reshape(features.shape[0], -1)
    
    N, D = features.shape
    
    stats = {
        "feature_name": feature_name,
        "original_shape": list(original_shape),
        "flattened_dim": D,
        "methods": {}
    }
    
    all_importance = []
    
    for method in methods:
        print(f"   计算 {method} 重要性...")
        importance = compute_dimension_importance(features, labels, method)
        
        sorted_indices = np.argsort(importance)[::-1]
        
        stats["methods"][method] = {
            "importance_mean": float(np.mean(importance)),
            "importance_std": float(np.std(importance)),
            "importance_max": float(np.max(importance)),
            "top_10_dim": [int(x) for x in sorted_indices[:10]],
            "top_10_importance": [float(importance[x]) for x in sorted_indices[:10]],
        }
        
        all_importance.append(importance)
    
    # 综合排名（多方法平均）
    if len(all_importance) > 1:
        # 归一化后平均
        normalized = []
        for imp in all_importance:
            imp_min, imp_max = imp.min(), imp.max()
            if imp_max > imp_min:
                normalized.append((imp - imp_min) / (imp_max - imp_min))
            else:
                normalized.append(np.zeros_like(imp))
        
        combined_importance = np.mean(normalized, axis=0)
        combined_sorted = np.argsort(combined_importance)[::-1]
        
        stats["combined"] = {
            "top_10_dim": [int(x) for x in combined_sorted[:10]],
            "top_10_importance": [float(combined_importance[x]) for x in combined_sorted[:10]],
        }
    
    return stats


def plot_dimension_importance(results: Dict, output_path: str):
    """
    绘制维度重要性对比图
    
    Args:
        results: 分析结果
        output_path: 输出路径
    """
    feature_names = ["original", "offset", "delta"]
    
    # 获取 MI 数据
    mi_means = []
    for name in feature_names:
        if name in results["dimension_importance"]:
            mi_means.append(
                results["dimension_importance"][name]
                .get("methods", {})
                .get("mutual_info", {})
                .get("importance_mean", 0)
            )
        else:
            mi_means.append(0)
    
    # 获取 Fisher 数据
    fisher_means = []
    for name in feature_names:
        if name in results["dimension_importance"]:
            fisher_means.append(
                results["dimension_importance"][name]
                .get("methods", {})
                .get("fisher", {})
                .get("importance_mean", 0)
            )
        else:
            fisher_means.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    # 子图 1: 互信息
    ax1 = axes[0]
    bars = ax1.bar(x, mi_means, width, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax1.set_xlabel('Feature Type')
    ax1.set_ylabel('Mean Mutual Information')
    ax1.set_title('Mean MI per Dimension by Feature Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_names)
    
    for bar, val in zip(bars, mi_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 子图 2: Fisher
    ax2 = axes[1]
    bars = ax2.bar(x, fisher_means, width, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax2.set_xlabel('Feature Type')
    ax2.set_ylabel('Mean Fisher Score')
    ax2.set_title('Mean Fisher Score per Dimension by Feature Type')
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_names)
    
    for bar, val in zip(bars, fisher_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   图表已保存: {output_path}")


def plot_top_dimensions(results: Dict, feature_name: str, output_path: str):
    """
    绘制 Top 维度重要性柱状图
    
    Args:
        results: 分析结果
        feature_name: 特征名称
        output_path: 输出路径
    """
    if feature_name not in results["dimension_importance"]:
        print(f"   未找到 {feature_name} 的数据")
        return
    
    data = results["dimension_importance"][feature_name]
    
    # 获取 MI 和 Fisher 的 top 维度
    mi_data = data.get("methods", {}).get("mutual_info", {})
    fisher_data = data.get("methods", {}).get("fisher", {})
    
    if not mi_data and not fisher_data:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图 1: MI Top 10
    if mi_data:
        ax1 = axes[0]
        dims = [f"D{d}" for d in mi_data.get("top_10_dim", [])]
        values = mi_data.get("top_10_importance", [])
        
        bars = ax1.barh(range(len(dims)), values, color='#3498db')
        ax1.set_yticks(range(len(dims)))
        ax1.set_yticklabels(dims)
        ax1.set_xlabel('Mutual Information')
        ax1.set_title(f'Top 10 Dimensions - {feature_name} (MI)')
        ax1.invert_yaxis()
    
    # 子图 2: Fisher Top 10
    if fisher_data:
        ax2 = axes[1]
        dims = [f"D{d}" for d in fisher_data.get("top_10_dim", [])]
        values = fisher_data.get("top_10_importance", [])
        
        bars = ax2.barh(range(len(dims)), values, color='#e74c3c')
        ax2.set_yticks(range(len(dims)))
        ax2.set_yticklabels(dims)
        ax2.set_xlabel('Fisher Score')
        ax2.set_title(f'Top 10 Dimensions - {feature_name} (Fisher)')
        ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   图表已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Offset 维度重要性分析")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["photo", "elliptic", "tolokers"],
                        help="数据集名称")
    parser.add_argument("--pp_k", type=int, default=6, help="Hop 数量")
    parser.add_argument("--alpha", type=float, default=0.1, help="PPR 参数")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--plot", action="store_true", help="是否生成图表")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Offset 维度重要性分析")
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
    
    # 分析维度重要性
    print("\n分析维度重要性...")
    
    results = {
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "pp_k": args.pp_k,
            "alpha": args.alpha,
        },
        "dimension_importance": {}
    }
    
    # 1. 原始特征
    print("\n1. 原始特征维度重要性...")
    stats_original = analyze_dimension_importance(features, labels, "original")
    results["dimension_importance"]["original"] = stats_original
    
    # 2. Offset
    print("\n2. Offset 维度重要性...")
    stats_offset = analyze_dimension_importance(offsets, labels, "offset")
    results["dimension_importance"]["offset"] = stats_offset
    
    # 3. Delta
    print("\n3. Delta 维度重要性...")
    stats_delta = analyze_dimension_importance(deltas, labels, "delta")
    results["dimension_importance"]["delta"] = stats_delta
    
    # 打印结果
    print("\n" + "=" * 60)
    print("维度重要性对比")
    print("=" * 60)
    
    for name in ["original", "offset", "delta"]:
        data = results["dimension_importance"][name]
        mi_mean = data.get("methods", {}).get("mutual_info", {}).get("importance_mean", 0)
        fisher_mean = data.get("methods", {}).get("fisher", {}).get("importance_mean", 0)
        print(f"\n{name}:")
        print(f"  MI 均值: {mi_mean:.4f}")
        print(f"  Fisher 均值: {fisher_mean:.4f}")
        
        # Top 5 维度
        top_dims = data.get("methods", {}).get("mutual_info", {}).get("top_10_dim", [])[:5]
        print(f"  Top 5 维度: {top_dims}")
    
    # 绘制图表
    if args.plot:
        print("\n生成图表...")
        plot_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(os.path.dirname(plot_dir), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # 对比图
        plot_path = os.path.join(plot_dir, f"dimension_importance_{args.dataset}.png")
        plot_dimension_importance(results, plot_path)
        
        # 各特征 Top 维度图
        for name in ["offset", "delta"]:
            plot_path = os.path.join(plot_dir, f"top_dims_{name}_{args.dataset}.png")
            plot_top_dimensions(results, name, plot_path)
    
    # 保存结果
    if args.output is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(output_dir), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"dimension_importance_{args.dataset}.json")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {args.output}")


if __name__ == "__main__":
    main()