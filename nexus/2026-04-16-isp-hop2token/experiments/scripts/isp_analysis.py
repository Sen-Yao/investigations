#!/usr/bin/env python3
"""ISP (Individual Smoothing Pattern) 分析脚本

验证 ISP_like 指标在 hop2token 中的异常检测潜力

作者: Nexus
日期: 2026-04-16
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats
import os
import scipy.io as sio
import scipy.sparse as sp

def load_data(dataset='photo'):
    """加载数据集"""
    data = sio.loadmat(f'./dataset/{dataset}.mat')
    features = data.get('Attributes', data.get('X'))
    if sp.issparse(features):
        features = features.todense()
    adj = data.get('Network', data.get('A'))
    if sp.issparse(adj):
        adj = adj.todense()
    labels = data.get('Label', data.get('Y')).flatten()
    labels = labels - labels.min()
    return np.array(features), np.array(adj), labels

def normalize_adj(adj):
    """归一化邻接矩阵"""
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree.flatten()), 0)
    d_inv_sqrt = np.diag(d_inv_sqrt)
    return d_inv_sqrt @ adj @ d_inv_sqrt

def compute_hop_features(features, adj_norm, K=6):
    """计算多 hop 特征"""
    features_t = torch.FloatTensor(features)
    adj_norm_t = torch.FloatTensor(adj_norm)
    N, D = features.shape
    hop_features = torch.zeros((N, K+1, D))
    hop_features[:, 0] = features_t
    agg = features_t.clone()
    for k in range(1, K+1):
        agg = adj_norm_t @ agg
        hop_features[:, k] = agg
    return hop_features

def compute_isp_metrics(hop_features, labels, K=6):
    """计算 ISP 相关指标"""
    N = hop_features.shape[0]
    
    # 1. ISP_like: 传播总偏离程度
    isp_total = torch.norm(hop_features[:, K] - hop_features[:, 0], p=2, dim=1).numpy()
    
    # 2. ISP_per_hop: 每个 hop 的偏离程度
    isp_per_hop = torch.zeros(N, K)
    for k in range(K):
        isp_per_hop[:, k] = torch.norm(hop_features[:, k+1] - hop_features[:, k], p=2, dim=1)
    isp_per_hop = isp_per_hop.numpy()
    
    # 3. ISP_rate: 偏离速率变化
    isp_rate = np.zeros(N)
    for i in range(N):
        changes = isp_per_hop[i]
        isp_rate[i] = np.std(changes) / (np.mean(changes) + 1e-8)
    
    # 4. Delta_norm
    delta_norms = isp_per_hop.copy()
    
    return {
        'isp_total': isp_total,
        'isp_per_hop': isp_per_hop,
        'isp_rate': isp_rate,
        'delta_norms': delta_norms
    }

def analyze_distribution(metrics, labels, save_dir):
    """分析正常/异常节点的 ISP 分布"""
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    results = {}
    
    for metric_name, metric_values in metrics.items():
        if metric_name == 'isp_per_hop' or metric_name == 'delta_norms':
            K = metric_values.shape[1]
            for k in range(K):
                normal_vals = metric_values[normal_mask, k]
                anomaly_vals = metric_values[anomaly_mask, k]
                
                ks_stat, ks_pval = stats.ks_2samp(normal_vals, anomaly_vals)
                mw_stat, mw_pval = stats.mannwhitneyu(normal_vals, anomaly_vals)
                
                auc_like = 0.5 + abs(np.mean(anomaly_vals) - np.mean(normal_vals)) / (2 * (np.std(normal_vals) + np.std(anomaly_vals) + 1e-8))
                
                results[f'{metric_name}_hop{k}'] = {
                    'normal_mean': np.mean(normal_vals),
                    'normal_std': np.std(normal_vals),
                    'anomaly_mean': np.mean(anomaly_vals),
                    'anomaly_std': np.std(anomaly_vals),
                    'ks_stat': ks_stat,
                    'ks_pval': ks_pval,
                    'mw_pval': mw_pval,
                    'auc_like': auc_like
                }
        else:
            normal_vals = metric_values[normal_mask]
            anomaly_vals = metric_values[anomaly_mask]
            
            ks_stat, ks_pval = stats.ks_2samp(normal_vals, anomaly_vals)
            mw_stat, mw_pval = stats.mannwhitneyu(normal_vals, anomaly_vals)
            auc_like = 0.5 + abs(np.mean(anomaly_vals) - np.mean(normal_vals)) / (2 * (np.std(normal_vals) + np.std(anomaly_vals) + 1e-8))
            
            results[metric_name] = {
                'normal_mean': np.mean(normal_vals),
                'normal_std': np.std(normal_vals),
                'anomaly_mean': np.mean(anomaly_vals),
                'anomaly_std': np.std(anomaly_vals),
                'ks_stat': ks_stat,
                'ks_pval': ks_pval,
                'mw_pval': mw_pval,
                'auc_like': auc_like
            }
    
    return results

def visualize_distribution(metrics, labels, save_dir):
    """可视化分布"""
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ISP_total
    ax = axes[0, 0]
    sns.histplot(metrics['isp_total'][normal_mask], ax=ax, label='Normal', color='blue', alpha=0.5, stat='density')
    sns.histplot(metrics['isp_total'][anomaly_mask], ax=ax, label='Anomaly', color='red', alpha=0.5, stat='density')
    ax.set_title('ISP_total Distribution')
    ax.legend()
    
    # ISP_rate
    ax = axes[0, 1]
    sns.histplot(metrics['isp_rate'][normal_mask], ax=ax, label='Normal', color='blue', alpha=0.5, stat='density')
    sns.histplot(metrics['isp_rate'][anomaly_mask], ax=ax, label='Anomaly', color='red', alpha=0.5, stat='density')
    ax.set_title('ISP_rate Distribution')
    ax.legend()
    
    # Delta_norm per hop
    ax = axes[1, 0]
    K = metrics['delta_norms'].shape[1]
    
    normal_mean = np.mean(metrics['delta_norms'][normal_mask], axis=0)
    normal_std = np.std(metrics['delta_norms'][normal_mask], axis=0)
    anomaly_mean = np.mean(metrics['delta_norms'][anomaly_mask], axis=0)
    anomaly_std = np.std(metrics['delta_norms'][anomaly_mask], axis=0)
    
    ax.errorbar(range(K), anomaly_mean, yerr=anomaly_std, color='red', marker='o', label='Anomaly', capsize=3)
    ax.errorbar(range(K), normal_mean, yerr=normal_std, color='blue', marker='s', label='Normal', capsize=3)
    ax.set_xlabel('Hop')
    ax.set_ylabel('Delta Norm')
    ax.set_title('Delta Norm per Hop')
    ax.legend()
    
    # Attention weight potential
    ax = axes[1, 1]
    auc_per_hop = []
    for k in range(K):
        normal_vals = metrics['isp_per_hop'][normal_mask, k]
        anomaly_vals = metrics['isp_per_hop'][anomaly_mask, k]
        auc_like = 0.5 + abs(np.mean(anomaly_vals) - np.mean(normal_vals)) / (2 * (np.std(normal_vals) + np.std(anomaly_vals) + 1e-8))
        auc_per_hop.append(auc_like)
    ax.plot(range(K), auc_per_hop, marker='o', linewidth=2, color='green')
    ax.set_xlabel('Hop')
    ax.set_ylabel('AUC-like Score')
    ax.set_title('Attention Weight Potential per Hop')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Random')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'isp_distribution_analysis.pdf'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'isp_distribution_analysis.png'), dpi=300)
    plt.close()
    
    print(f'可视化保存至: {save_dir}/isp_distribution_analysis.pdf')

def main():
    print('='*60)
    print('ISP (Individual Smoothing Pattern) 分析')
    print('='*60)
    
    features, adj, labels = load_data('photo')
    print(f'节点数: {len(labels)}, 异常数: {sum(labels)}, 异常率: {sum(labels)/len(labels)*100:.2f}%')
    
    adj_norm = normalize_adj(adj)
    
    K = 6
    hop_features = compute_hop_features(features, adj_norm, K)
    print(f'Hop features shape: {hop_features.shape}')
    
    metrics = compute_isp_metrics(hop_features, labels, K)
    
    save_dir = '../outputs'
    os.makedirs(save_dir, exist_ok=True)
    
    results = analyze_distribution(metrics, labels, save_dir)
    
    print()
    print('='*60)
    print('统计特征对比')
    print('='*60)
    
    for metric_name, stats_dict in results.items():
        print(f"\n{metric_name}:")
        print(f"  Normal:  mean={stats_dict['normal_mean']:.4f}, std={stats_dict['normal_std']:.4f}")
        print(f"  Anomaly: mean={stats_dict['anomaly_mean']:.4f}, std={stats_dict['anomaly_std']:.4f}")
        print(f"  KS test: stat={stats_dict['ks_stat']:.4f}, p={stats_dict['ks_pval']:.4e}")
        print(f"  MW test: p={stats_dict['mw_pval']:.4e}")
        print(f"  AUC-like: {stats_dict['auc_like']:.4f}")
    
    visualize_distribution(metrics, labels, save_dir)
    
    print()
    print('='*60)
    print('注意力权重潜力分析')
    print('='*60)
    
    best_hop = 0
    best_auc = 0
    for k in range(K):
        key = f'isp_per_hop_hop{k}'
        if results[key]['auc_like'] > best_auc:
            best_auc = results[key]['auc_like']
            best_hop = k
    
    print(f"最有区分力的 Hop: {best_hop} (AUC-like={best_auc:.4f})")
    
    if best_auc > 0.7:
        print("✅ ISP_per_hop 有较好的区分能力，可作为注意力权重")
    elif best_auc > 0.55:
        print("⚠️ ISP_per_hop 有一定区分能力，但较弱")
    else:
        print("❌ ISP_per_hop 区分能力不足，不适合直接作为注意力权重")
    
    return results

if __name__ == '__main__':
    main()