#!/usr/bin/env python3
"""
Phase 2: Token Strategy Physical Meaning Analysis
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Tuple, List
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, pearsonr, ks_2samp, mannwhitneyu
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# Output directories
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / 'outputs'
PLOTS_DIR = SCRIPT_DIR.parent / 'plots'
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


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
    """Load dataset from .mat file"""
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
        raise FileNotFoundError(f"Dataset not found at: {possible_paths}")
    
    data = sio.loadmat(data_dir)
    
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
        raise KeyError(f"Features not found, keys: {list(data.keys())}")
    
    label_keys = ['y', 'label', 'Label', 'labels', 'Class', 'str_anomaly_label', 'attr_anomaly_label']
    labels = None
    for key in label_keys:
        if key in data:
            labels = data[key].flatten().astype(np.int32)
            break
    
    if labels is None:
        raise KeyError(f"Labels not found, keys: {list(data.keys())}")
    
    adj_keys = ['A', 'adj', 'Network', 'network', 'adjacency']
    adj = None
    for key in adj_keys:
        if key in data:
            adj = data[key]
            break
    
    if adj is None:
        raise KeyError(f"Adjacency not found, keys: {list(data.keys())}")
    
    if sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    else:
        adj = sp.csr_matrix(adj)
    
    labels = (labels != 0).astype(np.int32)
    
    print(f"Dataset: {dataset_name}")
    print(f"Nodes: {features.shape[0]}")
    print(f"Features: {features.shape[1]}")
    print(f"Normal: {(labels == 0).sum()}")
    print(f"Anomaly: {(labels == 1).sum()}")
    
    return features, labels, adj


def compute_hop_features(features: np.ndarray, adj: sp.spmatrix, 
                          k: int = 6) -> np.ndarray:
    """Compute k-hop features with bilateral normalization"""
    N, D = features.shape
    
    adj_dense = adj.toarray().astype(np.float32)
    degree = adj_dense.sum(axis=1)
    degree[degree == 0] = 1
    d_inv_sqrt = np.power(degree, -0.5)
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = d_mat_inv_sqrt @ adj_dense @ d_mat_inv_sqrt
    
    hop_features = np.zeros((N, k + 1, D), dtype=np.float32)
    hop_features[:, 0, :] = features
    
    X = features.copy()
    for t in range(k):
        X = adj_norm @ X
        hop_features[:, t + 1, :] = X
    
    return hop_features


def compute_token_strategies(hop_features: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute three token strategies"""
    hop_0 = hop_features[:, 0:1, :]
    
    hop_tokens = hop_features.copy()
    
    offset_tokens = hop_features.copy()
    offset_tokens[:, 1:, :] = hop_features[:, 1:, :] - hop_0
    
    delta_tokens = hop_features.copy()
    delta_tokens[:, 1:, :] = hop_features[:, 1:, :] - hop_features[:, :-1, :]
    
    return {
        "hop": hop_tokens,
        "offset": offset_tokens,
        "delta": delta_tokens
    }


def compute_graph_features(adj: sp.spmatrix) -> Dict[str, np.ndarray]:
    """Compute graph structural features"""
    import networkx as nx
    
    adj_dense = adj.toarray()
    G = nx.from_numpy_array(adj_dense)
    
    N = adj.shape[0]
    
    degree = adj_dense.sum(axis=1)
    
    clustering = np.array([nx.clustering(G, i) for i in range(N)])
    
    pagerank = np.array(list(nx.pagerank(G).values()))
    
    features = {
        'degree': degree,
        'clustering': clustering,
        'pagerank': pagerank
    }
    
    if N < 10000:
        betweenness = np.array(list(nx.betweenness_centrality(G).values()))
        features['betweenness'] = betweenness
    
    print(f"Graph features computed for {N} nodes")
    return features


def analyze_visualization(strategies: Dict, labels: np.ndarray, dataset_name: str):
    """Task 1: Visualization analysis with t-SNE"""
    results = {'dataset': dataset_name, 'timestamp': datetime.now().isoformat()}
    
    for name, features in strategies.items():
        print(f"\n=== {name.upper()} Strategy Visualization ===")
        
        n_samples = features.shape[0]
        if n_samples > 3000:
            np.random.seed(42)
            sample_idx = np.random.choice(n_samples, 3000, replace=False)
            feat_sample = features[sample_idx]
            label_sample = labels[sample_idx]
        else:
            feat_sample = features
            label_sample = labels
        
        # Flatten for t-SNE
        feat_flat = feat_sample.reshape(feat_sample.shape[0], -1)
        
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat_flat)
        
        print(f"  Running t-SNE on {feat_scaled.shape[0]} samples...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_result = tsne.fit_transform(feat_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        normal_mask = label_sample == 0
        anomaly_mask = label_sample == 1
        
        axes[0].scatter(tsne_result[normal_mask, 0], tsne_result[normal_mask, 1], 
                       c='blue', alpha=0.5, s=5, label='Normal')
        axes[0].scatter(tsne_result[anomaly_mask, 0], tsne_result[anomaly_mask, 1], 
                       c='red', alpha=0.7, s=10, label='Anomaly')
        axes[0].set_title(f'{name.upper()} - t-SNE (by Label)')
        axes[0].legend()
        
        xy = tsne_result.T
        kde = gaussian_kde(xy)
        density = kde(xy)
        
        scatter = axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                  c=density, cmap='viridis', alpha=0.5, s=5)
        plt.colorbar(scatter, ax=axes[1], label='Density')
        axes[1].set_title(f'{name.upper()} - t-SNE (by Density)')
        
        plt.tight_layout()
        plot_path = PLOTS_DIR / f'{dataset_name}_{name}_tsne.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved: {plot_path}")
        
        if len(np.unique(label_sample)) > 1:
            sil_score = silhouette_score(feat_scaled, label_sample)
            ch_score = calinski_harabasz_score(feat_scaled, label_sample)
            results[f'{name}_silhouette'] = float(sil_score)
            results[f'{name}_calinski_harabasz'] = float(ch_score)
            print(f"  Silhouette: {sil_score:.4f}")
            print(f"  Calinski-Harabasz: {ch_score:.4f}")
    
    return results


def analyze_graph_correlation(strategies: Dict, graph_features: Dict, dataset_name: str):
    """Task 2: Correlation analysis between tokens and graph structure"""
    results = {'dataset': dataset_name, 'timestamp': datetime.now().isoformat()}
    
    print("\n=== Graph Structure Correlation Analysis ===")
    
    for strategy_name, features in strategies.items():
        print(f"\n--- {strategy_name.upper()} ---")
        strategy_results = {}
        
        mean_feature = np.mean(features, axis=1)
        feat_norm = np.linalg.norm(mean_feature, axis=1)
        
        for gf_name, gf_values in graph_features.items():
            if gf_values is None:
                continue
            
            valid_mask = ~(np.isnan(gf_values) | np.isnan(feat_norm))
            if valid_mask.sum() < 10:
                continue
            
            sp_corr, sp_p = spearmanr(feat_norm[valid_mask], gf_values[valid_mask])
            pr_corr, pr_p = pearsonr(feat_norm[valid_mask], gf_values[valid_mask])
            
            strategy_results[gf_name] = {
                'spearman_r': float(sp_corr),
                'spearman_p': float(sp_p),
                'pearson_r': float(pr_corr),
                'pearson_p': float(pr_p)
            }
            
            print(f"  {gf_name}: Spearman r={sp_corr:.4f} (p={sp_p:.4e})")
        
        results[strategy_name] = strategy_results
    
    return results


def analyze_node_patterns(strategies: Dict, labels: np.ndarray, dataset_name: str):
    """Task 3: Compare token patterns between normal and anomaly nodes"""
    results = {'dataset': dataset_name, 'timestamp': datetime.now().isoformat()}
    
    print("\n=== Node Pattern Analysis ===")
    
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    for strategy_name, features in strategies.items():
        print(f"\n--- {strategy_name.upper()} ---")
        
        num_tokens = features.shape[1]
        
        normal_means = []
        anomaly_means = []
        normal_stds = []
        anomaly_stds = []
        
        for t in range(num_tokens):
            token_feat = features[:, t, :]
            normal_means.append(np.mean(token_feat[normal_mask]))
            anomaly_means.append(np.mean(token_feat[anomaly_mask]))
            normal_stds.append(np.std(token_feat[normal_mask]))
            anomaly_stds.append(np.std(token_feat[anomaly_mask]))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        x = np.arange(num_tokens)
        width = 0.35
        
        axes[0].bar(x - width/2, normal_means, width, label='Normal', alpha=0.7, yerr=normal_stds)
        axes[0].bar(x + width/2, anomaly_means, width, label='Anomaly', alpha=0.7, yerr=anomaly_stds)
        axes[0].set_xlabel('Token Index')
        axes[0].set_ylabel('Mean Feature Value')
        axes[0].set_title(f'{strategy_name.upper()} - Token-wise Distribution')
        axes[0].legend()
        axes[0].set_xticks(x)
        
        normal_norms = np.linalg.norm(features[normal_mask], axis=(1, 2))
        anomaly_norms = np.linalg.norm(features[anomaly_mask], axis=(1, 2))
        
        axes[1].hist(normal_norms, bins=50, alpha=0.7, label='Normal', density=True)
        axes[1].hist(anomaly_norms, bins=50, alpha=0.7, label='Anomaly', density=True)
        axes[1].set_xlabel('Feature Norm')
        axes[1].set_ylabel('Density')
        axes[1].set_title(f'{strategy_name.upper()} - Norm Distribution')
        axes[1].legend()
        
        plt.tight_layout()
        plot_path = PLOTS_DIR / f'{dataset_name}_{strategy_name}_node_patterns.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved: {plot_path}")
        
        ks_stat, ks_p = ks_2samp(normal_norms, anomaly_norms)
        mw_stat, mw_p = mannwhitneyu(normal_norms, anomaly_norms, alternative='two-sided')
        
        results[strategy_name] = {
            'normal_norm_mean': float(np.mean(normal_norms)),
            'normal_norm_std': float(np.std(normal_norms)),
            'anomaly_norm_mean': float(np.mean(anomaly_norms)),
            'anomaly_norm_std': float(np.std(anomaly_norms)),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_p),
            'mannwhitney_stat': float(mw_stat),
            'mannwhitney_pvalue': float(mw_p),
            'token_means_normal': [float(x) for x in normal_means],
            'token_means_anomaly': [float(x) for x in anomaly_means]
        }
        
        print(f"  Normal norm: {np.mean(normal_norms):.4f} +/- {np.std(normal_norms):.4f}")
        print(f"  Anomaly norm: {np.mean(anomaly_norms):.4f} +/- {np.std(anomaly_norms):.4f}")
        print(f"  KS test: stat={ks_stat:.4f}, p={ks_p:.4e}")
    
    return results


def analyze_convergence(strategies: Dict, dataset_name: str):
    """Task 4: Analyze convergence properties"""
    results = {'dataset': dataset_name, 'timestamp': datetime.now().isoformat()}
    
    print("\n=== Convergence Analysis ===")
    
    for strategy_name, features in strategies.items():
        print(f"\n--- {strategy_name.upper()} ---")
        num_nodes, num_tokens, dim = features.shape
        
        token_norms = np.linalg.norm(features, axis=2)
        
        mean_norms = np.mean(token_norms, axis=0)
        std_norms = np.std(token_norms, axis=0)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        x = np.arange(num_tokens)
        
        axes[0].errorbar(x, mean_norms, yerr=std_norms, fmt='o-', capsize=5)
        axes[0].set_xlabel('Token Index (Hop)')
        axes[0].set_ylabel('Feature Norm')
        axes[0].set_title(f'{strategy_name.upper()} - Token Norm vs Hop')
        axes[0].set_xticks(x)
        
        if strategy_name in ['delta', 'hop']:
            axes[1].semilogy(x, mean_norms, 'o-', label='Actual')
            
            valid = mean_norms > 1e-10
            if valid.sum() > 2:
                coeffs = np.polyfit(x[valid], np.log(mean_norms[valid] + 1e-10), 1)
                fitted = np.exp(coeffs[1] + coeffs[0] * x)
                axes[1].semilogy(x, fitted, '--', label=f'Fit: exp({coeffs[0]:.3f}*k)')
                results[f'{strategy_name}_decay_rate'] = float(-coeffs[0])
                print(f"  Decay rate: {-coeffs[0]:.4f}")
            
            axes[1].set_xlabel('Token Index (Hop)')
            axes[1].set_ylabel('Feature Norm (log scale)')
            axes[1].set_title(f'{strategy_name.upper()} - Convergence Rate')
            axes[1].legend()
        else:
            axes[1].bar(x, mean_norms, alpha=0.7, yerr=std_norms)
            axes[1].set_xlabel('Token Index (Hop)')
            axes[1].set_ylabel('Feature Norm')
            axes[1].set_title(f'{strategy_name.upper()} - Token Stability')
        
        plt.tight_layout()
        plot_path = PLOTS_DIR / f'{dataset_name}_{strategy_name}_convergence.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved: {plot_path}")
        
        results[f'{strategy_name}_mean_norms'] = mean_norms.tolist()
        results[f'{strategy_name}_std_norms'] = std_norms.tolist()
        
        if num_tokens > 1:
            variance_ratio = np.var(mean_norms) / (np.mean(mean_norms) ** 2 + 1e-10)
            results[f'{strategy_name}_variance_ratio'] = float(variance_ratio)
            print(f"  Variance ratio: {variance_ratio:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--k', type=int, default=6)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Phase 2: Physical Meaning Analysis for {args.dataset.upper()}")
    print(f"{'='*60}\n")
    
    features, labels, adj = load_dataset(args.dataset)
    
    print("\nComputing hop features...")
    hop_features = compute_hop_features(features, adj, k=args.k)
    print(f"  Hop features: {hop_features.shape}")
    
    print("\nComputing token strategies...")
    strategies = compute_token_strategies(hop_features)
    for name, feat in strategies.items():
        print(f"  {name}: {feat.shape}")
    
    print("\nComputing graph features...")
    graph_features = compute_graph_features(adj)
    
    all_results = {
        'dataset': args.dataset,
        'config': {'k': args.k},
        'timestamp': datetime.now().isoformat()
    }
    
    vis_results = analyze_visualization(strategies, labels, args.dataset)
    all_results['visualization'] = vis_results
    
    graph_results = analyze_graph_correlation(strategies, graph_features, args.dataset)
    all_results['graph_correlation'] = graph_results
    
    pattern_results = analyze_node_patterns(strategies, labels, args.dataset)
    all_results['node_patterns'] = pattern_results
    
    conv_results = analyze_convergence(strategies, args.dataset)
    all_results['convergence'] = conv_results
    
    output_path = OUTPUT_DIR / f'phase2_physical_{args.dataset}.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {output_path}")
    
    return all_results


if __name__ == '__main__':
    main()