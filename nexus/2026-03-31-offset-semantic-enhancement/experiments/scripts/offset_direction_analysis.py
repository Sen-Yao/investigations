#!/usr/bin/env python3
"""
Offset 方向语义分析脚本

验证 H1 假设：
- H1a: 正常节点的 Offset 方向聚集
- H1b: 异常节点的 Offset 方向偏离正常方向

输出：
- outputs/stats_{dataset}.json - 统计数据
- plots/direction_tsne_{dataset}.png - t-SNE 可视化
- plots/similarity_boxplot_{dataset}.png - 相似度箱线图
"""

import os
import sys
import json
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 添加 VoxG 目录到路径
sys.path.insert(0, os.path.expanduser('~/VoxG'))


class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 序列化"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# 1. 数据加载模块
# ============================================================================

def load_dataset(dataset_name, data_dir='~/VoxG/dataset'):
    """
    加载数据集
    
    Args:
        dataset_name: 数据集名称 (如 'Photo', 'Amazon', 'Reddit')
        data_dir: 数据目录
    
    Returns:
        features: 特征矩阵 [N, D]
        labels: 标签向量 [N], 0=正常, 1=异常
        adj: 邻接矩阵 (scipy sparse)
    """
    data_path = os.path.expanduser(f'{data_dir}/{dataset_name}.mat')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件不存在: {data_path}")
    
    print(f"加载数据集: {data_path}")
    data = sio.loadmat(data_path)
    
    # 提取标签
    label_key = 'Label' if 'Label' in data else 'gnd'
    labels = np.squeeze(np.array(data[label_key]))
    
    # 提取特征
    attr_key = 'Attributes' if 'Attributes' in data else 'X'
    features = data[attr_key]
    if sp.issparse(features):
        features = features.tolil()
    
    # 提取邻接矩阵
    network_key = 'Network' if 'Network' in data else 'A'
    adj = sp.csr_matrix(data[network_key])
    
    # 转换特征为 dense numpy array
    if sp.issparse(features):
        features = features.toarray()
    features = np.array(features, dtype=np.float32)
    
    print(f"  节点数: {adj.shape[0]}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  异常节点: {labels.sum()} ({labels.mean()*100:.2f}%)")
    
    return features, labels, adj


def preprocess_features(features):
    """行归一化特征矩阵"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if sp.issparse(features):
        features = features.toarray()
    return features.astype(np.float32)


def normalize_adj(adj):
    """对称归一化邻接矩阵"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# ============================================================================
# 2. Offset 计算模块
# ============================================================================

def compute_hop_features(features, adj, k, alpha=0.1):
    """
    计算多跳特征
    
    使用 PPR 风格的聚合：
    X^{(k+1)} = (1-alpha) * A * X^{(k)} + alpha * X^{(0)}
    
    Args:
        features: 特征矩阵 [N, D]
        adj: 归一化邻接矩阵
        k: 跳数
        alpha: PPR 重启概率
    
    Returns:
        hop_features: 多跳特征列表 [X^{(0)}, X^{(1)}, ..., X^{(k)}]
    """
    features = torch.from_numpy(features).float()
    adj_tensor = torch.from_numpy(np.array(adj.todense())).float()
    
    hop_features = [features]  # X^{(0)}
    x_k = features
    
    for i in range(k):
        # PPR 风格聚合
        x_k = (1 - alpha) * torch.mm(adj_tensor, x_k) + alpha * features
        hop_features.append(x_k)
    
    return hop_features


def compute_offsets(features, adj, pp_k=6, alpha=0.1):
    """
    计算 Offset 向量
    
    Offset_k = Feature(Hop_k) - Feature(Hop_0)
    
    Args:
        features: 原始特征矩阵 [N, D]
        adj: 邻接矩阵
        pp_k: 最大跳数
        alpha: PPR 重启概率
    
    Returns:
        offsets: Offset 向量 [N, K, D]
    """
    print(f"计算多跳特征 (k={pp_k}, alpha={alpha})...")
    
    # 归一化邻接矩阵
    adj_norm = normalize_adj(adj)
    
    # 计算多跳特征
    hop_features = compute_hop_features(features, adj_norm, pp_k, alpha)
    
    # 计算 Offset
    N, D = features.shape
    offsets = np.zeros((N, pp_k, D), dtype=np.float32)
    
    hop0 = hop_features[0].numpy()  # X^{(0)}
    
    for k in range(1, pp_k + 1):
        hop_k = hop_features[k].numpy()
        offsets[:, k-1, :] = hop_k - hop0  # Offset_k = X^{(k)} - X^{(0)}
    
    print(f"  Offset 形状: {offsets.shape}")
    
    return offsets


# ============================================================================
# 3. 方向分析模块
# ============================================================================

def compute_directions(offsets, eps=1e-10):
    """
    计算方向向量（归一化），同时记录零向量信息
    
    direction = offset / ||offset||_2
    
    Args:
        offsets: Offset 向量 [N, K, D]
        eps: 判断零向量的阈值
    
    Returns:
        directions: 方向向量 [N, K, D]
        has_zero_vector: 每个节点是否有任何零向量 [N]
        zero_vectors: 每个节点每个跳是否有零向量 [N, K]
    """
    print("计算方向向量...")
    
    N, K, D = offsets.shape
    directions = np.zeros_like(offsets)
    zero_vectors = np.zeros((N, K), dtype=bool)  # 记录每个节点每个跳的零向量情况
    
    for k in range(K):
        offset_k = offsets[:, k, :]  # [N, D]
        
        # 计算 L2 范数
        norms = np.linalg.norm(offset_k, axis=1, keepdims=True)
        
        # 识别零向量
        zero_mask = (norms.flatten() < eps)
        zero_vectors[:, k] = zero_mask  # 记录
        
        # 避免除以零
        norms[zero_mask] = 1.0
        
        # 归一化
        directions[:, k, :] = offset_k / norms
        
        # 零向量保持为零
        directions[zero_mask, k, :] = 0.0
    
    # 计算每个节点是否有任何零向量（任意一跳为零）
    has_zero_vector = zero_vectors.any(axis=1)  # [N]
    
    print(f"  总零向量数量: {zero_vectors.sum()} / {N * K}")
    print(f"  有零向量的节点数: {has_zero_vector.sum()} / {N}")
    
    return directions, has_zero_vector, zero_vectors


def compute_direction_variance(directions, mask, k):
    """
    正确计算方向向量的方差
    
    方法：计算每个维度方差后平均
    
    Args:
        directions: 方向向量 [N, K, D]
        mask: 节点掩码
        k: 跳数索引
    
    Returns:
        方差均值
    """
    dirs = directions[mask, k, :]  # [n, D]
    if dirs.shape[0] == 0:
        return 0.0
    var_per_dim = np.var(dirs, axis=0)
    return float(var_per_dim.mean())


def compute_direction_center(directions, normal_mask):
    """
    计算正常节点的方向中心
    
    center = mean(directions[normal])
    center = normalize(center)
    
    Args:
        directions: 方向向量 [N, K, D]
        normal_mask: 正常节点掩码 [N]
    
    Returns:
        centers: 每个跳的方向中心 [K, D]
    """
    print("计算正常节点方向中心...")
    
    K = directions.shape[1]
    centers = np.zeros((K, directions.shape[2]), dtype=np.float32)
    
    for k in range(K):
        # 获取正常节点的方向向量
        normal_dirs = directions[normal_mask, k, :]  # [n_normal, D]
        
        # 计算平均方向
        center = normal_dirs.mean(axis=0)
        
        # 归一化
        norm = np.linalg.norm(center)
        if norm > 1e-10:
            center = center / norm
        
        centers[k] = center
    
    return centers


def compute_similarity(directions, centers):
    """
    计算与中心的余弦相似度
    
    similarity = cosine_sim(direction, center)
    
    Args:
        directions: 方向向量 [N, K, D]
        centers: 方向中心 [K, D]
    
    Returns:
        similarities: 相似度 [N, K]
    """
    print("计算与中心的余弦相似度...")
    
    N, K, D = directions.shape
    similarities = np.zeros((N, K), dtype=np.float32)
    
    for k in range(K):
        dir_k = directions[:, k, :]  # [N, D]
        center_k = centers[k]  # [D]
        
        # 余弦相似度
        similarities[:, k] = np.dot(dir_k, center_k)
    
    return similarities


def compute_deviation(directions, centers):
    """
    计算方向偏离度
    
    deviation = 1 - cosine_sim(direction, center)
    
    Args:
        directions: 方向向量 [N, K, D]
        centers: 方向中心 [K, D]
    
    Returns:
        deviations: 偏离度 [N, K]
    """
    similarities = compute_similarity(directions, centers)
    deviations = 1 - similarities
    return deviations


# ============================================================================
# 4. 统计检验模块
# ============================================================================

def mann_whitney_test(normal_sims, anomaly_sims):
    """
    Mann-Whitney U test
    
    检验 H1b: 正常节点相似度 > 异常节点相似度
    
    Args:
        normal_sims: 正常节点相似度
        anomaly_sims: 异常节点相似度
    
    Returns:
        statistic: U 统计量
        p_value: p 值
    
    注意:
        alternative='greater' 检验 H1: normal > anomaly
        如果 p < 0.05，则拒绝 H0，支持 H1（正常相似度显著更高）
    """
    statistic, p_value = mannwhitneyu(normal_sims, anomaly_sims, alternative='greater')
    return statistic, p_value


def spearman_correlation(deviations, anomaly_scores):
    """
    Spearman 相关分析
    
    检验方向偏离与异常的相关性
    
    Args:
        deviations: 偏离度 [N] 或 [N, K]
        anomaly_scores: 异常分数 [N]
    
    Returns:
        rho: Spearman 相关系数
        p_value: p 值
    """
    if len(deviations.shape) > 1:
        # 平均偏离度
        deviations = deviations.mean(axis=1)
    
    rho, p_value = spearmanr(deviations, anomaly_scores)
    return rho, p_value


# ============================================================================
# 5. 可视化模块
# ============================================================================

def visualize_tsne(directions, labels, save_path, perplexity=30, n_iter=1000):
    """
    t-SNE 降维可视化
    
    正常节点蓝色，异常节点红色
    
    Args:
        directions: 方向向量 [N, K, D]
        labels: 标签 [N], 0=正常, 1=异常
        save_path: 保存路径
        perplexity: t-SNE 困惑度
        n_iter: 迭代次数
    """
    print("生成 t-SNE 可视化...")
    
    # 展平为 [N, K*D]
    N, K, D = directions.shape
    X = directions.reshape(N, K * D)
    
    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    normal_mask = (labels == 0)
    anomaly_mask = (labels == 1)
    
    plt.scatter(X_embedded[normal_mask, 0], X_embedded[normal_mask, 1], 
                c='blue', alpha=0.5, label='Normal', s=10)
    plt.scatter(X_embedded[anomaly_mask, 0], X_embedded[anomaly_mask, 1], 
                c='red', alpha=0.5, label='Anomaly', s=10)
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('Offset Direction t-SNE Visualization')
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  保存至: {save_path}")


def visualize_similarity_boxplot(normal_sims, anomaly_sims, save_path, hop_labels=None):
    """
    相似度箱线图
    
    Args:
        normal_sims: 正常节点相似度 [n_normal, K]
        anomaly_sims: 异常节点相似度 [n_anomaly, K]
        save_path: 保存路径
        hop_labels: 跳数标签
    """
    print("生成相似度箱线图...")
    
    K = normal_sims.shape[1]
    if hop_labels is None:
        hop_labels = [f'Hop {i+1}' for i in range(K)]
    
    fig, axes = plt.subplots(1, K, figsize=(4*K, 6))
    if K == 1:
        axes = [axes]
    
    for k in range(K):
        data = [normal_sims[:, k], anomaly_sims[:, k]]
        bp = axes[k].boxplot(data, labels=['Normal', 'Anomaly'], patch_artist=True)
        
        # 设置颜色
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        axes[k].set_ylabel('Cosine Similarity')
        axes[k].set_title(hop_labels[k])
        axes[k].grid(True, alpha=0.3)
    
    plt.suptitle('Direction Similarity Distribution')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  保存至: {save_path}")


# ============================================================================
# 6. 主函数
# ============================================================================

def analyze_dataset(dataset_name, pp_k=6, alpha=0.1, base_dir='~/VoxG/nexus/investigations/2026-03-31-offset-semantic-enhancement/experiments'):
    """
    分析单个数据集
    
    Args:
        dataset_name: 数据集名称
        pp_k: 最大跳数
        alpha: PPR 重启概率
        base_dir: 输出目录
    """
    base_dir = os.path.expanduser(base_dir)
    output_dir = os.path.join(base_dir, 'outputs')
    plot_dir = os.path.join(base_dir, 'plots')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"分析数据集: {dataset_name}")
    print(f"{'='*60}\n")
    
    # 1. 加载数据集
    features, labels, adj = load_dataset(dataset_name)
    features = preprocess_features(features)
    
    # 2. 计算 Offset
    offsets = compute_offsets(features, adj, pp_k=pp_k, alpha=alpha)
    
    # 3. 计算方向向量（修复：同时返回零向量信息）
    directions, has_zero_vector, zero_vectors = compute_directions(offsets)
    
    # 4. 计算方向中心和相似度
    normal_mask = (labels == 0)
    centers = compute_direction_center(directions, normal_mask)
    similarities = compute_similarity(directions, centers)
    deviations = compute_deviation(directions, centers)
    
    # 5. 统计分析（修复：排除零向量节点，使用正确的方差计算）
    stats = {
        'dataset': dataset_name,
        'pp_k': pp_k,
        'alpha': alpha,
        'n_nodes': int(len(labels)),
        'n_normal': int(normal_mask.sum()),
        'n_anomaly': int((~normal_mask).sum()),
        'n_zero_vector_nodes': int(has_zero_vector.sum()),  # 新增：有零向量的节点数
        'hops': []
    }
    
    # 构造排除零向量后的掩码（修复 Bug 2）
    non_zero_mask = ~has_zero_vector
    normal_non_zero = normal_mask & non_zero_mask
    anomaly_non_zero = (~normal_mask) & non_zero_mask
    
    print(f"\n零向量统计:")
    print(f"  有零向量的节点: {has_zero_vector.sum()} / {len(labels)}")
    print(f"  用于统计的正常节点: {normal_non_zero.sum()} / {normal_mask.sum()}")
    print(f"  用于统计的异常节点: {anomaly_non_zero.sum()} / {(~normal_mask).sum()}")
    
    # 对每个跳进行分析
    normal_sims_all = []
    anomaly_sims_all = []
    
    for k in range(pp_k):
        hop_stats = {}
        
        # 零向量统计（新增）
        hop_stats['hop'] = k + 1
        hop_stats['zero_ratio_normal'] = float(zero_vectors[normal_mask, k].mean())
        hop_stats['zero_ratio_anomaly'] = float(zero_vectors[~normal_mask, k].mean())
        hop_stats['non_zero_count_normal'] = int(normal_non_zero.sum())
        hop_stats['non_zero_count_anomaly'] = int(anomaly_non_zero.sum())
        
        # 提取当前跳的相似度（修复：排除零向量节点）
        normal_sims = similarities[normal_non_zero, k]
        anomaly_sims = similarities[anomaly_non_zero, k]
        
        normal_sims_all.append(normal_sims)
        anomaly_sims_all.append(anomaly_sims)
        
        # 描述性统计
        hop_stats['normal_sim_mean'] = float(normal_sims.mean())
        hop_stats['normal_sim_std'] = float(normal_sims.std())
        hop_stats['anomaly_sim_mean'] = float(anomaly_sims.mean())
        hop_stats['anomaly_sim_std'] = float(anomaly_sims.std())
        
        # 方差比较 (H1a) - 修复 Bug 1：使用正确的方向向量方差计算
        normal_var = compute_direction_variance(directions, normal_non_zero, k)
        anomaly_var = compute_direction_variance(directions, anomaly_non_zero, k)
        
        hop_stats['normal_direction_var'] = float(normal_var)
        hop_stats['anomaly_direction_var'] = float(anomaly_var)
        hop_stats['var_ratio'] = float(anomaly_var / (normal_var + 1e-10))
        
        # Mann-Whitney U test (H1b) - 已修复 Bug 3：使用 alternative='greater'
        stat, p_val = mann_whitney_test(normal_sims, anomaly_sims)
        hop_stats['mann_whitney_stat'] = float(stat)
        hop_stats['mann_whitney_p'] = float(p_val)
        hop_stats['significant'] = bool(p_val < 0.05)
        
        stats['hops'].append(hop_stats)
        
        print(f"\nHop {k+1}:")
        print(f"  零向量比例 - 正常: {hop_stats['zero_ratio_normal']*100:.2f}%, 异常: {hop_stats['zero_ratio_anomaly']*100:.2f}%")
        print(f"  正常相似度: {normal_sims.mean():.4f} ± {normal_sims.std():.4f} (n={len(normal_sims)})")
        print(f"  异常相似度: {anomaly_sims.mean():.4f} ± {anomaly_sims.std():.4f} (n={len(anomaly_sims)})")
        print(f"  方向向量方差 - 正常: {normal_var:.6f}, 异常: {anomaly_var:.6f}")
        print(f"  方差比 (异常/正常): {hop_stats['var_ratio']:.4f}")
        print(f"  Mann-Whitney U (greater): stat={stat:.2f}, p={p_val:.4e}")
    
    # 6. Spearman 相关分析（修复：排除零向量节点）
    # 使用平均偏离度
    mean_deviations = deviations.mean(axis=1)
    
    # 排除零向量节点进行相关分析
    mean_deviations_non_zero = mean_deviations[non_zero_mask]
    labels_non_zero = labels[non_zero_mask].astype(float)
    
    rho, p_val = spearman_correlation(mean_deviations_non_zero, labels_non_zero)
    stats['spearman_rho'] = float(rho)
    stats['spearman_p'] = float(p_val)
    stats['spearman_significant'] = bool(p_val < 0.05)
    
    print(f"\nSpearman 相关分析 (排除零向量节点):")
    print(f"  偏离度-异常相关: ρ={rho:.4f}, p={p_val:.4e}")
    
    # 7. 可视化
    visualize_tsne(directions, labels, 
                   os.path.join(plot_dir, f'direction_tsne_{dataset_name}.png'))
    
    visualize_similarity_boxplot(np.array(normal_sims_all).T, 
                                  np.array(anomaly_sims_all).T,
                                  os.path.join(plot_dir, f'similarity_boxplot_{dataset_name}.png'))
    
    # 8. 保存统计结果
    stats_path = os.path.join(output_dir, f'stats_{dataset_name}.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    
    print(f"\n统计结果保存至: {stats_path}")
    
    return stats


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Offset 方向语义分析')
    parser.add_argument('--dataset', type=str, default='Photo', 
                        help='数据集名称')
    parser.add_argument('--pp_k', type=int, default=6, 
                        help='最大跳数')
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help='PPR 重启概率')
    parser.add_argument('--base_dir', type=str, 
                        default='~/VoxG/nexus/investigations/2026-03-31-offset-semantic-enhancement/experiments',
                        help='输出目录')
    
    args = parser.parse_args()
    
    stats = analyze_dataset(
        dataset_name=args.dataset,
        pp_k=args.pp_k,
        alpha=args.alpha,
        base_dir=args.base_dir
    )
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)
    
    # 输出 H1 假设验证结论
    print("\n## H1 假设验证")
    print("\n### H1a: 正常节点 Offset 方向聚集")
    for hop_stats in stats['hops']:
        var_ratio = hop_stats['var_ratio']
        conclusion = "✓ 支持" if var_ratio > 1 else "✗ 不支持"
        print(f"  Hop {hop_stats['hop']}: 方差比={var_ratio:.4f} {conclusion}")
    
    print("\n### H1b: 异常节点 Offset 方向偏离正常方向")
    for hop_stats in stats['hops']:
        sim_diff = hop_stats['normal_sim_mean'] - hop_stats['anomaly_sim_mean']
        significant = "显著" if hop_stats['significant'] else "不显著"
        print(f"  Hop {hop_stats['hop']}: 相似度差={sim_diff:.4f}, {significant}")
    
    print(f"\n### 相关性分析 (Spearman)")
    print(f"  ρ={stats['spearman_rho']:.4f}, p={stats['spearman_p']:.4e}")
    conclusion = "✓ 支持正相关" if stats['spearman_rho'] > 0.3 else "✗ 弱相关"
    print(f"  {conclusion}")


if __name__ == '__main__':
    main()