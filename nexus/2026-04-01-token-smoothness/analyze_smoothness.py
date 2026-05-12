#!/usr/bin/env python3
"""
Token 平滑度与异常检测关系分析脚本

研究假设：
- H1: 深层 Token 方差更低（更平滑）
- H2: 平滑度与区分力负相关
- H3: 异常检测需要高频信号（高方差）

关键约束：
- alpha=0（纯邻居聚合）
- 邻接矩阵双边归一化 D^{-0.5}AD^{-0.5}

作者：Nexus
日期：2026-04-01
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
from scipy import stats
import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt


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
        f"/root/gpufree-data/linziyao/VoxG/dataset/{dataset_name.capitalize()}.mat",
        f"/root/gpufree-data/linziyao/VoxG/dataset/{dataset_name}.mat",
        os.path.expanduser(f"~/gpufree-data/linziyao/VoxG/dataset/{dataset_name.capitalize()}.mat"),
        os.path.expanduser(f"~/gpufree-data/linziyao/VoxG/dataset/{dataset_name}.mat"),
        os.path.expanduser(f"~/VoxG/dataset/{dataset_name.capitalize()}.mat"),
        os.path.expanduser(f"~/VoxG/dataset/{dataset_name}.mat"),
    ]
    
    data_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            data_dir = path
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


def compute_variance_per_layer(hop_features: np.ndarray) -> np.ndarray:
    """
    计算每个 Token 层的特征方差（平滑度指标）
    
    Args:
        hop_features: [N, K+1, D]
    
    Returns:
        variances: [K+1] 每个 Token 层的总方差
    """
    K_plus_1 = hop_features.shape[1]
    variances = np.zeros(K_plus_1)
    
    for k in range(K_plus_1):
        # 计算该层所有节点所有维度的方差
        layer_features = hop_features[:, k, :]  # [N, D]
        variances[k] = np.var(layer_features)
    
    return variances


def compute_variance_per_node_per_layer(hop_features: np.ndarray) -> np.ndarray:
    """
    计算每个节点在每个 Token 层的特征方差
    
    Args:
        hop_features: [N, K+1, D]
    
    Returns:
        variances: [N, K+1] 每个节点每层的方差
    """
    N, K_plus_1, D = hop_features.shape
    variances = np.zeros((N, K_plus_1))
    
    for k in range(K_plus_1):
        # 每个节点在 D 维上的方差
        layer_features = hop_features[:, k, :]  # [N, D]
        variances[:, k] = np.var(layer_features, axis=1)
    
    return variances


def compute_ks_statistic(hop_features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算每个 Token 层的 KS 统计量（区分力指标）
    
    KS 统计量衡量正常节点和异常节点特征分布的差异
    
    Args:
        hop_features: [N, K+1, D]
        labels: [N] 标签
    
    Returns:
        ks_stats: [K+1] 每个 Token 层的平均 KS 统计量
        p_values: [K+1] 每个 Token 层的平均 p 值
    """
    K_plus_1 = hop_features.shape[1]
    ks_stats = np.zeros(K_plus_1)
    p_values = np.zeros(K_plus_1)
    
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    for k in range(K_plus_1):
        layer_features = hop_features[:, k, :]  # [N, D]
        
        # 对每个维度计算 KS 统计量，然后取平均
        ks_per_dim = []
        p_per_dim = []
        for d in range(layer_features.shape[1]):
            feat_normal = layer_features[normal_mask, d]
            feat_anomaly = layer_features[anomaly_mask, d]
            
            ks, p = stats.ks_2samp(feat_normal, feat_anomaly)
            ks_per_dim.append(ks)
            p_per_dim.append(p)
        
        ks_stats[k] = np.mean(ks_per_dim)
        p_values[k] = np.mean(p_per_dim)
    
    return ks_stats, p_values


def compute_fisher_score(hop_features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    计算每个 Token 层的 Fisher 分数（区分力指标）
    
    Fisher Score = (μ_1 - μ_0)^2 / (σ_0^2 + σ_1^2)
    
    Args:
        hop_features: [N, K+1, D]
        labels: [N] 标签
    
    Returns:
        fisher_scores: [K+1] 每个 Token 层的总 Fisher 分数
    """
    K_plus_1 = hop_features.shape[1]
    fisher_scores = np.zeros(K_plus_1)
    
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    for k in range(K_plus_1):
        layer_features = hop_features[:, k, :]  # [N, D]
        
        mean_normal = np.mean(layer_features[normal_mask], axis=0)
        mean_anomaly = np.mean(layer_features[anomaly_mask], axis=0)
        
        var_normal = np.var(layer_features[normal_mask], axis=0)
        var_anomaly = np.var(layer_features[anomaly_mask], axis=0)
        
        between_class = (mean_normal - mean_anomaly) ** 2
        within_class = var_normal + var_anomaly + 1e-10
        
        fisher_scores[k] = np.sum(between_class / within_class)
    
    return fisher_scores


def compute_frequency_characteristics(hop_features: np.ndarray) -> Dict:
    """
    分析 Token 层的频率特性
    
    深层 Token 应该更"低频"，表现为：
    1. 相邻层的差异变小
    2. 特征变化率下降
    
    Args:
        hop_features: [N, K+1, D]
    
    Returns:
        freq_stats: 频率特性统计
    """
    K_plus_1 = hop_features.shape[1]
    
    # 计算相邻层的差异
    layer_diffs = []
    for k in range(K_plus_1 - 1):
        diff = hop_features[:, k+1, :] - hop_features[:, k, :]
        layer_diffs.append(np.mean(np.abs(diff)))
    
    # 计算特征变化率（L2 范数）
    layer_norms = []
    for k in range(K_plus_1):
        norms = np.linalg.norm(hop_features[:, k, :], axis=1)
        layer_norms.append(np.mean(norms))
    
    # 计算频谱能量（通过 DFT 分析）
    # 将 hop 序列视为时间序列，分析频谱
    spectral_energy = []
    for n in range(hop_features.shape[0]):
        # 对每个节点的 hop 序列计算 DFT
        node_features = hop_features[n, :, :]  # [K+1, D]
        # 对每个维度计算 DFT，然后平均
        dft_results = np.fft.fft(node_features, axis=0)
        # 低频能量（前 2 个频率分量）
        low_freq_energy = np.sum(np.abs(dft_results[:2, :]) ** 2)
        spectral_energy.append(low_freq_energy)
    
    return {
        "layer_diffs": layer_diffs,  # 相邻层差异
        "layer_norms": layer_norms,  # 每层特征范数
        "spectral_energy_low": np.mean(spectral_energy),  # 低频能量
    }


def analyze_smoothness_discrimination_correlation(variances: np.ndarray, 
                                                    ks_stats: np.ndarray,
                                                    fisher_scores: np.ndarray) -> Dict:
    """
    分析平滑度与区分力的相关性
    
    Args:
        variances: [K+1] 每个 Token 层的方差
        ks_stats: [K+1] 每个 Token 层的 KS 统计量
        fisher_scores: [K+1] 每个 Token 层的 Fisher 分数
    
    Returns:
        correlation_stats: 相关性统计
    """
    # 方差 vs KS 统计量
    corr_variance_ks, p_variance_ks = stats.pearsonr(variances, ks_stats)
    spearman_variance_ks, sp_variance_ks = stats.spearmanr(variances, ks_stats)
    
    # 方差 vs Fisher 分数
    corr_variance_fisher, p_variance_fisher = stats.pearsonr(variances, fisher_scores)
    spearman_variance_fisher, sp_variance_fisher = stats.spearmanr(variances, fisher_scores)
    
    return {
        "variance_vs_ks": {
            "pearson_r": float(corr_variance_ks),
            "pearson_p": float(p_variance_ks),
            "spearman_r": float(spearman_variance_ks),
            "spearman_p": float(sp_variance_ks),
        },
        "variance_vs_fisher": {
            "pearson_r": float(corr_variance_fisher),
            "pearson_p": float(p_variance_fisher),
            "spearman_r": float(spearman_variance_fisher),
            "spearman_p": float(sp_variance_fisher),
        }
    }


def analyze_dataset(dataset_name: str, k: int = 6) -> Dict:
    """
    分析单个数据集的 Token 平滑度与异常检测关系
    
    Args:
        dataset_name: 数据集名称
        k: hop 数量
    
    Returns:
        results: 分析结果
    """
    print(f"\n{'='*60}")
    print(f"分析数据集: {dataset_name}")
    print(f"{'='*60}")
    
    # 加载数据集
    features, labels, adj = load_dataset(dataset_name)
    N, D = features.shape
    
    # 计算 hop 特征
    print(f"\n计算 Hop 特征 (alpha=0, 双边归一化, k={k})...")
    hop_features = compute_hop_features(features, adj, k)
    print(f"Hop 特征形状: {hop_features.shape}")
    
    # ========== H1 验证：深层 Token 方差更低 ==========
    print("\n[假设 H1] 深层 Token 方差更低（更平滑）")
    variances = compute_variance_per_layer(hop_features)
    
    print("\n各层方差:")
    for i, v in enumerate(variances):
        print(f"  Token {i}: {v:.6f}")
    
    # 计算方差下降率
    variance_ratios = []
    for i in range(1, len(variances)):
        ratio = variances[i] / variances[i-1] if variances[i-1] > 0 else 0
        variance_ratios.append(ratio)
    
    print("\n方差比率 (layer_i / layer_{i-1}):")
    for i, r in enumerate(variance_ratios):
        print(f"  Token {i+1}/Token {i}: {r:.4f}")
    
    # 拟合方差随层数变化的趋势
    layer_indices = np.arange(len(variances))
    slope, intercept, r_value, p_value, std_err = stats.linregress(layer_indices, variances)
    
    h1_result = {
        "variances": variances.tolist(),
        "variance_ratios": variance_ratios,
        "variance_trend": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
        },
        "conclusion": "H1 支持" if slope < 0 else "H1 不支持",
        "evidence": f"方差随层数{'下降' if slope < 0 else '上升'} (slope={slope:.6f}, p={p_value:.4f})"
    }
    print(f"\n结论: {h1_result['conclusion']}")
    print(f"证据: {h1_result['evidence']}")
    
    # ========== H2 验证：平滑度与区分力负相关 ==========
    print("\n[假设 H2] 平滑度与区分力负相关")
    
    # 计算区分力指标
    ks_stats, ks_p_values = compute_ks_statistic(hop_features, labels)
    fisher_scores = compute_fisher_score(hop_features, labels)
    
    print("\n各层 KS 统计量:")
    for i, (ks, p) in enumerate(zip(ks_stats, ks_p_values)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  Token {i}: KS={ks:.4f}, p={p:.4f} {sig}")
    
    print("\n各层 Fisher 分数:")
    for i, f in enumerate(fisher_scores):
        print(f"  Token {i}: {f:.4f}")
    
    # 分析相关性
    corr_stats = analyze_smoothness_discrimination_correlation(variances, ks_stats, fisher_scores)
    
    print("\n相关性分析:")
    print(f"  方差 vs KS (Pearson r={corr_stats['variance_vs_ks']['pearson_r']:.4f}, "
          f"p={corr_stats['variance_vs_ks']['pearson_p']:.4f})")
    print(f"  方差 vs Fisher (Pearson r={corr_stats['variance_vs_fisher']['pearson_r']:.4f}, "
          f"p={corr_stats['variance_vs_fisher']['pearson_p']:.4f})")
    
    # H2 判断标准：方差与区分力负相关
    h2_variance_ks_supported = corr_stats['variance_vs_ks']['pearson_r'] < 0 and corr_stats['variance_vs_ks']['pearson_p'] < 0.05
    h2_variance_fisher_supported = corr_stats['variance_vs_fisher']['pearson_r'] < 0 and corr_stats['variance_vs_fisher']['pearson_p'] < 0.05
    
    h2_result = {
        "ks_stats": ks_stats.tolist(),
        "ks_p_values": ks_p_values.tolist(),
        "fisher_scores": fisher_scores.tolist(),
        "correlation": corr_stats,
        "conclusion": "H2 支持" if (h2_variance_ks_supported or h2_variance_fisher_supported) else "H2 不支持",
        "evidence": f"方差-KS相关={corr_stats['variance_vs_ks']['pearson_r']:.4f}, 方差-Fisher相关={corr_stats['variance_vs_fisher']['pearson_r']:.4f}"
    }
    print(f"\n结论: {h2_result['conclusion']}")
    print(f"证据: {h2_result['evidence']}")
    
    # ========== H3 验证：异常检测需要高频信号 ==========
    print("\n[假设 H3] 异常检测需要高频信号（高方差）")
    
    # 频率特性分析
    freq_stats = compute_frequency_characteristics(hop_features)
    
    print("\n频率特性:")
    print(f"  相邻层差异: {freq_stats['layer_diffs']}")
    print(f"  每层范数: {freq_stats['layer_norms']}")
    print(f"  低频能量: {freq_stats['spectral_energy_low']:.4f}")
    
    # 分析哪一层对异常检测最有效
    best_ks_layer = np.argmax(ks_stats)
    best_fisher_layer = np.argmax(fisher_scores)
    highest_variance_layer = np.argmax(variances)
    
    # H3 判断：最佳区分层是否为低层（高方差层）
    h3_ks_supported = best_ks_layer <= 2  # 0, 1, 2 层为高频层
    h3_fisher_supported = best_fisher_layer <= 2
    
    h3_result = {
        "frequency_stats": freq_stats,
        "best_ks_layer": int(best_ks_layer),
        "best_fisher_layer": int(best_fisher_layer),
        "highest_variance_layer": int(highest_variance_layer),
        "conclusion": "H3 支持" if (h3_ks_supported or h3_fisher_supported) else "H3 不支持",
        "evidence": f"最佳KS层={best_ks_layer}, 最佳Fisher层={best_fisher_layer}, 最高方差层={highest_variance_layer}"
    }
    print(f"\n结论: {h3_result['conclusion']}")
    print(f"证据: {h3_result['evidence']}")
    
    # ========== 汇总结果 ==========
    results = {
        "dataset": dataset_name,
        "N": N,
        "D": D,
        "k": k,
        "alpha": 0,
        "normalization": "D^{-1/2} A D^{-1/2}",
        "H1_variance_decrease": h1_result,
        "H2_smoothness_discrimination_correlation": h2_result,
        "H3_high_frequency_needed": h3_result,
        "summary": {
            "h1_supported": slope < 0,
            "h2_supported": h2_variance_ks_supported or h2_variance_fisher_supported,
            "h3_supported": h3_ks_supported or h3_fisher_supported,
        }
    }
    
    return results


def generate_summary_table(all_results: List[Dict]) -> str:
    """生成汇总表格"""
    table = """
## 各层方差对比表

| 数据集 | D | Token 0 | Token 1 | Token 2 | Token 3 | Token 4 | Token 5 | Token 6 | 趋势 |
|--------|---|---------|---------|---------|---------|---------|---------|---------|------|
"""
    for r in all_results:
        variances = r["H1_variance_decrease"]["variances"]
        trend = "↓" if r["summary"]["h1_supported"] else "↑"
        table += f"| {r['dataset']} | {r['D']} |"
        for v in variances:
            table += f" {v:.4f} |"
        table += f" {trend} |\n"
    
    table += """
## 方差-KS 相关性结果

| 数据集 | 方差-KS (r) | 方差-KS (p) | 方差-Fisher (r) | 方差-Fisher (p) | H2 结论 |
|--------|-------------|-------------|-----------------|-----------------|---------|
"""
    for r in all_results:
        corr = r["H2_smoothness_discrimination_correlation"]["correlation"]
        h2 = "支持" if r["summary"]["h2_supported"] else "不支持"
        table += f"| {r['dataset']} | {corr['variance_vs_ks']['pearson_r']:.4f} | {corr['variance_vs_ks']['pearson_p']:.4f} | {corr['variance_vs_fisher']['pearson_r']:.4f} | {corr['variance_vs_fisher']['pearson_p']:.4f} | {h2} |\n"
    
    table += """
## 最佳区分层分析

| 数据集 | 最佳 KS 层 | 最佳 Fisher 层 | 最高方差层 | H3 结论 |
|--------|-----------|---------------|-----------|---------|
"""
    for r in all_results:
        h3 = r["H3_high_frequency_needed"]
        conclusion = "支持" if r["summary"]["h3_supported"] else "不支持"
        table += f"| {r['dataset']} | {h3['best_ks_layer']} | {h3['best_fisher_layer']} | {h3['highest_variance_layer']} | {conclusion} |\n"
    
    return table


def main():
    parser = argparse.ArgumentParser(description="Token 平滑度与异常检测关系分析")
    parser.add_argument("--datasets", type=str, default="photo,tolokers,elliptic",
                        help="数据集列表，逗号分隔")
    parser.add_argument("--k", type=int, default=6, help="Hop 数量")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    
    args = parser.parse_args()
    
    datasets = [d.strip() for d in args.datasets.split(",")]
    
    print("=" * 60)
    print("Token 平滑度与异常检测关系分析")
    print("=" * 60)
    print(f"参数: alpha=0, 双边归一化 D^{{-1/2}} A D^{{-1/2}}, k={args.k}")
    print(f"数据集: {datasets}")
    
    # 设置输出目录
    if args.output is None:
        args.output = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.output, exist_ok=True)
    
    # 分析每个数据集
    all_results = []
    for dataset in datasets:
        try:
            results = analyze_dataset(dataset, args.k)
            all_results.append(results)
        except Exception as e:
            print(f"\n错误: 数据集 {dataset} 分析失败: {e}")
            continue
    
    # 生成汇总
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)
    
    summary_table = generate_summary_table(all_results)
    print(summary_table)
    
    # 假设验证汇总
    print("\n## 假设验证汇总")
    print()
    h1_count = sum(1 for r in all_results if r["summary"]["h1_supported"])
    h2_count = sum(1 for r in all_results if r["summary"]["h2_supported"])
    h3_count = sum(1 for r in all_results if r["summary"]["h3_supported"])
    
    print(f"- **H1** (深层 Token 方差更低): {h1_count}/{len(all_results)} 支持")
    print(f"- **H2** (平滑度与区分力负相关): {h2_count}/{len(all_results)} 支持")
    print(f"- **H3** (异常检测需要高频信号): {h3_count}/{len(all_results)} 支持")
    
    # 直觉冲突回答
    print("\n## 直觉冲突回答")
    print()
    if h1_count > 0:
        print("**深层 Token 确实更平滑**：方差随层数下降")
    else:
        print("**深层 Token 不一定更平滑**：方差趋势不明显")
    
    if h2_count > 0:
        print("**平滑度与区分力负相关**：高方差层有更好的区分力")
    else:
        print("**平滑度与区分力关系不明确**：可能存在其他因素")
    
    if h3_count > 0:
        print("**异常检测需要高频信号**：最佳区分层在前几层")
    else:
        print("**异常检测不一定需要高频信号**：深层可能也有价值")
    
    # 保存结果
    output_json = os.path.join(args.output, "smoothness_analysis.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "k": args.k,
                "alpha": 0,
                "normalization": "D^{-1/2} A D^{-1/2}",
            },
            "results": all_results,
        }, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    print(f"\n✅ 结果已保存: {output_json}")
    
    # 保存汇总表格
    summary_md = os.path.join(args.output, "SUMMARY.md")
    with open(summary_md, 'w', encoding='utf-8') as f:
        f.write("# Token 平滑度与异常检测关系分析\n\n")
        f.write(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 假设\n\n")
        f.write("| 假设 | 说明 |\n")
        f.write("|------|------|\n")
        f.write("| H1 | 深层 Token 方差更低（更平滑） |\n")
        f.write("| H2 | 平滑度与区分力负相关 |\n")
        f.write("| H3 | 异常检测需要高频信号（高方差） |\n\n")
        f.write(summary_table)
        f.write("\n## 假设验证汇总\n\n")
        f.write(f"- **H1** (深层 Token 方差更低): {h1_count}/{len(all_results)} 支持\n")
        f.write(f"- **H2** (平滑度与区分力负相关): {h2_count}/{len(all_results)} 支持\n")
        f.write(f"- **H3** (异常检测需要高频信号): {h3_count}/{len(all_results)} 支持\n")
        f.write("\n## 直觉冲突回答\n\n")
        if h1_count > 0:
            f.write("**深层 Token 确实更平滑**：方差随层数下降\n\n")
        else:
            f.write("**深层 Token 不一定更平滑**：方差趋势不明显\n\n")
        if h2_count > 0:
            f.write("**平滑度与区分力负相关**：高方差层有更好的区分力\n\n")
        else:
            f.write("**平滑度与区分力关系不明确**：可能存在其他因素\n\n")
        if h3_count > 0:
            f.write("**异常检测需要高频信号**：最佳区分层在前几层\n\n")
        else:
            f.write("**异常检测不一定需要高频信号**：深层可能也有价值\n\n")
    print(f"✅ 汇总已保存: {summary_md}")


if __name__ == "__main__":
    main()