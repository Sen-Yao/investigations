#!/usr/bin/env python3
"""
Phase 3: GT Attention Analysis and Validation
Tasks:
1. GT attention mechanism analysis
2. Parameter sensitivity analysis (k values, alpha values)
3. Mixed token strategy exploration
4. Simple MLP validation experiment
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Tuple, List
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr, ks_2samp
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
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
                          k: int = 6, alpha: float = 0.0, 
                          max_nodes: int = 20000) -> np.ndarray:
    """Compute k-hop features with bilateral normalization + PPR diffusion
    For large datasets, use only a subset to avoid memory issues
    """
    N, D = features.shape
    
    # For very large datasets, sample nodes
    if N > max_nodes:
        print(f"    Large dataset ({N} nodes), sampling {max_nodes} nodes for analysis...")
        np.random.seed(42)
        sample_idx = np.random.choice(N, max_nodes, replace=False)
        features = features[sample_idx]
        adj = adj[sample_idx, :][:, sample_idx]
        N = max_nodes
    
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
        # PPR-style diffusion: (1-alpha) * adj @ X + alpha * features
        X = (1 - alpha) * adj_norm @ X + alpha * features
        hop_features[:, t + 1, :] = X
    
    return hop_features


def compute_token_strategies(hop_features: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute three token strategies"""
    hop_0 = hop_features[:, 0:1, :]
    
    # Hop strategy: original multi-hop features
    hop_tokens = hop_features.copy()
    
    # Offset strategy: hop_k - hop_0
    offset_tokens = hop_features.copy()
    offset_tokens[:, 1:, :] = hop_features[:, 1:, :] - hop_0
    
    # Delta strategy: hop_{k+1} - hop_k
    delta_tokens = np.zeros_like(hop_features)
    delta_tokens[:, 0, :] = hop_features[:, 0, :]
    for t in range(hop_features.shape[1] - 1):
        delta_tokens[:, t + 1, :] = hop_features[:, t + 1, :] - hop_features[:, t, :]
    
    return {
        'hop': hop_tokens,
        'offset': offset_tokens,
        'delta': delta_tokens
    }


def simulate_gt_attention(tokens: np.ndarray, strategy_name: str) -> Dict:
    """Simulate GT attention behavior based on token characteristics"""
    N, num_tokens, D = tokens.shape
    
    # Compute token-wise statistics
    token_norms = np.linalg.norm(tokens, axis=2)  # [N, num_tokens]
    token_means = np.mean(token_norms, axis=0)  # [num_tokens]
    token_stds = np.std(token_norms, axis=0)  # [num_tokens]
    
    # Sample subset for attention analysis
    sample_size = min(500, N)
    sample_idx = np.random.choice(N, sample_size, replace=False)
    sample_tokens = tokens[sample_idx]
    
    # Compute pairwise cosine similarity for first token vs all others
    # This simulates "readout" attention pattern in VoxGFormer
    first_token = sample_tokens[:, 0, :]  # [sample, D]
    first_token_norm = np.linalg.norm(first_token, axis=1, keepdims=True) + 1e-8
    first_token_normalized = first_token / first_token_norm
    
    attention_scores = []
    for t in range(num_tokens):
        target_token = sample_tokens[:, t, :]
        target_token_norm = np.linalg.norm(target_token, axis=1, keepdims=True) + 1e-8
        target_token_normalized = target_token / target_token_norm
        
        # Cosine similarity
        sim = np.sum(first_token_normalized * target_token_normalized, axis=1)
        attention_scores.append(np.mean(sim))
    
    attention_distribution = np.array(attention_scores)
    
    # Normalize to simulate softmax
    attention_weights = np.exp(attention_distribution) / np.sum(np.exp(attention_distribution))
    
    return {
        'strategy': strategy_name,
        'num_tokens': num_tokens,
        'token_mean_norms': token_means.tolist(),
        'token_std_norms': token_stds.tolist(),
        'attention_scores': attention_distribution.tolist(),
        'attention_weights': attention_weights.tolist(),
        'token_0_attention_to_self': attention_weights[0],
        'deep_token_attention': np.mean(attention_weights[-3:]) if num_tokens >= 3 else 0
    }


def analyze_attention_patterns(features: np.ndarray, labels: np.ndarray, 
                                adj: sp.spmatrix, k: int = 6) -> Dict:
    """Analyze attention patterns for all strategies"""
    hop_features = compute_hop_features(features, adj, k)
    strategies = compute_token_strategies(hop_features)
    
    results = {}
    for name, tokens in strategies.items():
        print(f"  Analyzing attention for {name} strategy...")
        results[name] = simulate_gt_attention(tokens, name)
    
    # Cross-strategy comparison
    results['comparison'] = {
        'hop_attention_weight_0': results['hop']['attention_weights'][0],
        'offset_attention_weight_0': results['offset']['attention_weights'][0],
        'delta_attention_weight_0': results['delta']['attention_weights'][0],
        'hop_deep_attention': results['hop']['deep_token_attention'],
        'offset_deep_attention': results['offset']['deep_token_attention'],
        'delta_deep_attention': results['delta']['deep_token_attention']
    }
    
    return results


def parameter_sensitivity_analysis(features: np.ndarray, labels: np.ndarray,
                                   adj: sp.spmatrix, 
                                   k_values: List[int] = [3, 6, 9, 12],
                                   alpha_values: List[float] = [0.0, 0.1, 0.2]) -> Dict:
    """Analyze parameter sensitivity"""
    results = {
        'k_sensitivity': {},
        'alpha_sensitivity': {}
    }
    
    # K sensitivity (with fixed alpha=0)
    print("\n=== K Sensitivity Analysis ===")
    for k in k_values:
        print(f"  Testing k={k}...")
        hop_features = compute_hop_features(features, adj, k, alpha=0.0)
        strategies = compute_token_strategies(hop_features)
        
        k_result = {}
        for name, tokens in strategies.items():
            attention = simulate_gt_attention(tokens, name)
            k_result[name] = {
                'deep_attention': attention['deep_token_attention'],
                'attention_concentration': attention['attention_weights'][0],
                'token_norm_variance': np.var(attention['token_mean_norms'])
            }
        results['k_sensitivity'][k] = k_result
    
    # Alpha sensitivity (with fixed k=6)
    print("\n=== Alpha Sensitivity Analysis ===")
    for alpha in alpha_values:
        print(f"  Testing alpha={alpha}...")
        hop_features = compute_hop_features(features, adj, k=6, alpha=alpha)
        strategies = compute_token_strategies(hop_features)
        
        alpha_result = {}
        for name, tokens in strategies.items():
            attention = simulate_gt_attention(tokens, name)
            alpha_result[name] = {
                'deep_attention': attention['deep_token_attention'],
                'attention_concentration': attention['attention_weights'][0],
                'token_norm_variance': np.var(attention['token_mean_norms'])
            }
        results['alpha_sensitivity'][alpha] = alpha_result
    
    return results


def mixed_strategy_analysis(features: np.ndarray, labels: np.ndarray,
                            adj: sp.spmatrix, k: int = 6) -> Dict:
    """Analyze mixed token strategies"""
    print("\n=== Mixed Strategy Analysis ===")
    hop_features = compute_hop_features(features, adj, k)
    strategies = compute_token_strategies(hop_features)
    
    # Strategy 1: First 3 tokens Hop, rest Delta
    print("  Testing Hop(0-2) + Delta(3-6) mixed strategy...")
    mixed_tokens_1 = np.zeros_like(hop_features)
    mixed_tokens_1[:, :3, :] = strategies['hop'][:, :3, :]
    mixed_tokens_1[:, 3:, :] = strategies['delta'][:, 3:, :]
    
    # Strategy 2: First 3 tokens Hop, rest Offset
    print("  Testing Hop(0-2) + Offset(3-6) mixed strategy...")
    mixed_tokens_2 = np.zeros_like(hop_features)
    mixed_tokens_2[:, :3, :] = strategies['hop'][:, :3, :]
    mixed_tokens_2[:, 3:, :] = strategies['offset'][:, 3:, :]
    
    # Strategy 3: First half Offset, second half Delta
    print("  Testing Offset(0-3) + Delta(4-6) mixed strategy...")
    mixed_tokens_3 = np.zeros_like(hop_features)
    mixed_tokens_3[:, :4, :] = strategies['offset'][:, :4, :]
    mixed_tokens_3[:, 4:, :] = strategies['delta'][:, 4:, :]
    
    results = {
        'hop_delta_mix': simulate_gt_attention(mixed_tokens_1, 'hop_delta_mix'),
        'hop_offset_mix': simulate_gt_attention(mixed_tokens_2, 'hop_offset_mix'),
        'offset_delta_mix': simulate_gt_attention(mixed_tokens_3, 'offset_delta_mix')
    }
    
    # Information quality comparison
    results['comparison'] = {
        'hop_delta_deep_attention': results['hop_delta_mix']['deep_token_attention'],
        'hop_offset_deep_attention': results['hop_offset_mix']['deep_token_attention'],
        'offset_delta_deep_attention': results['offset_delta_mix']['deep_token_attention']
    }
    
    return results


def mlp_validation_experiment(features: np.ndarray, labels: np.ndarray,
                              adj: sp.spmatrix, k: int = 6,
                              test_ratio: float = 0.3,
                              max_samples: int = 10000) -> Dict:
    """Simple MLP validation experiment
    For large datasets, use stratified sampling to keep class balance
    """
    print("\n=== MLP Validation Experiment ===")
    
    N_orig = features.shape[0]
    
    # Sample if too large
    if N_orig > max_samples:
        print(f"    Sampling {max_samples} nodes for MLP (original: {N_orig})...")
        # Stratified sampling to maintain anomaly ratio
        anomaly_idx = np.where(labels == 1)[0]
        normal_idx = np.where(labels == 0)[0]
        
        anomaly_sample_size = min(len(anomaly_idx), int(max_samples * len(anomaly_idx) / N_orig))
        normal_sample_size = max_samples - anomaly_sample_size
        
        np.random.seed(42)
        sampled_anomaly = np.random.choice(anomaly_idx, anomaly_sample_size, replace=False)
        sampled_normal = np.random.choice(normal_idx, normal_sample_size, replace=False)
        
        sample_idx = np.concatenate([sampled_anomaly, sampled_normal])
        np.random.shuffle(sample_idx)
        
        features = features[sample_idx]
        labels = labels[sample_idx]
        adj = adj[sample_idx, :][:, sample_idx]
    
    hop_features = compute_hop_features(features, adj, k, max_nodes=max_samples)
    strategies = compute_token_strategies(hop_features)
    
    # Create feature representations for each strategy
    N, num_tokens, D = hop_features.shape
    
    results = {}
    
    for name, tokens in strategies.items():
        print(f"  Testing {name} strategy with MLP...")
        
        # Flatten tokens to [N, num_tokens * D]
        X_flat = tokens.reshape(N, -1)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flat)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, labels, test_size=test_ratio, stratify=labels, random_state=42
        )
        
        # Train simple MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=100,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        mlp.fit(X_train, y_train)
        
        # Evaluate
        y_pred = mlp.predict(X_test)
        y_prob = mlp.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Feature importance
        first_layer_weights = mlp.coefs_[0]
        token_importance = np.mean(np.abs(first_layer_weights), axis=1)
        
        token_importance_per_token = []
        for t in range(num_tokens):
            start_idx = t * D
            end_idx = (t + 1) * D
            token_importance_per_token.append(np.mean(token_importance[start_idx:end_idx]))
        
        results[name] = {
            'auc': auc,
            'accuracy': acc,
            'f1_score': f1,
            'num_features': X_flat.shape[1],
            'token_importance': token_importance_per_token,
            'deep_token_importance': np.mean(token_importance_per_token[-3:]) if num_tokens >= 3 else 0,
            'first_token_importance': token_importance_per_token[0]
        }
        
        print(f"    {name}: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
    
    return results


def generate_plots(results: Dict, dataset: str, plots_dir: Path):
    """Generate visualization plots"""
    # Plot 1: Attention distribution comparison
    plt.figure(figsize=(10, 6))
    
    strategies = ['hop', 'offset', 'delta']
    attention_data = {}
    for s in strategies:
        if s in results.get('attention', {}):
            attention_data[s] = results['attention'][s]['attention_weights']
    
    if attention_data:
        num_tokens = len(attention_data['hop'])
        x = range(num_tokens)
        
        for s, weights in attention_data.items():
            plt.plot(x, weights, marker='o', label=s)
        
        plt.xlabel('Token Index')
        plt.ylabel('Attention Weight')
        plt.title(f'{dataset}: Token Attention Distribution by Strategy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / f'{dataset}_attention_distribution.png', dpi=150)
        plt.close()
    
    # Plot 2: MLP performance comparison
    if 'mlp_validation' in results:
        plt.figure(figsize=(8, 5))
        
        mlp_results = results['mlp_validation']
        strategies = list(mlp_results.keys())
        aucs = [mlp_results[s]['auc'] for s in strategies]
        
        bars = plt.bar(strategies, aucs, color=['steelblue', 'darkorange', 'green'])
        plt.xlabel('Token Strategy')
        plt.ylabel('AUC Score')
        plt.title(f'{dataset}: MLP Classification Performance')
        plt.ylim(0, 1)
        
        for bar, auc in zip(bars, aucs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                     f'{auc:.3f}', ha='center', fontsize=10)
        
        plt.savefig(plots_dir / f'{dataset}_mlp_performance.png', dpi=150)
        plt.close()
    
    # Plot 3: K sensitivity
    if 'parameter_sensitivity' in results:
        k_sens = results['parameter_sensitivity']['k_sensitivity']
        
        plt.figure(figsize=(10, 6))
        k_values = list(k_sens.keys())
        strategies = ['hop', 'offset', 'delta']
        
        for s in strategies:
            deep_atts = [k_sens[k][s]['deep_attention'] for k in k_values]
            plt.plot(k_values, deep_atts, marker='o', label=f'{s}')
        
        plt.xlabel('K (number of hops)')
        plt.ylabel('Deep Token Attention')
        plt.title(f'{dataset}: K Parameter Sensitivity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / f'{dataset}_k_sensitivity.png', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Phase 3: GT Attention Analysis')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--k', type=int, default=6, help='Number of hops')
    parser.add_argument('--k_values', type=str, default='3,6,9,12', help='K values for sensitivity')
    parser.add_argument('--alpha_values', type=str, default='0,0.1,0.2', help='Alpha values for sensitivity')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--plots_dir', type=str, default=None, help='Plots directory')
    
    args = parser.parse_args()
    
    k_values = [int(x) for x in args.k_values.split(',')]
    alpha_values = [float(x) for x in args.alpha_values.split(',')]
    
    # Set output directories
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / 'outputs'
    
    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
    else:
        plots_dir = Path(__file__).parent.parent / 'plots'
    
    output_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Phase 3: GT Attention Analysis for {args.dataset}")
    print(f"{'='*60}\n")
    
    # Load dataset
    features, labels, adj = load_dataset(args.dataset)
    
    # Run analyses
    results = {
        'dataset': args.dataset,
        'config': {
            'k': args.k,
            'k_values': k_values,
            'alpha_values': alpha_values
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # 1. Attention analysis
    print("\n[1] Attention Pattern Analysis...")
    results['attention'] = analyze_attention_patterns(features, labels, adj, args.k)
    
    # 2. Parameter sensitivity
    print("\n[2] Parameter Sensitivity Analysis...")
    results['parameter_sensitivity'] = parameter_sensitivity_analysis(
        features, labels, adj, k_values, alpha_values
    )
    
    # 3. Mixed strategy analysis
    print("\n[3] Mixed Strategy Analysis...")
    results['mixed_strategy'] = mixed_strategy_analysis(features, labels, adj, args.k)
    
    # 4. MLP validation
    print("\n[4] MLP Validation Experiment...")
    results['mlp_validation'] = mlp_validation_experiment(features, labels, adj, args.k)
    
    # Generate plots
    print("\n[5] Generating plots...")
    generate_plots(results, args.dataset, plots_dir)
    
    # Save results
    output_file = output_dir / f'phase3_{args.dataset}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Phase 3 Analysis Summary for {args.dataset}")
    print(f"{'='*60}")
    
    # MLP comparison
    mlp = results['mlp_validation']
    print("\nMLP Performance:")
    for name in ['hop', 'offset', 'delta']:
        if name in mlp:
            print(f"  {name}: AUC={mlp[name]['auc']:.4f}, DeepImportance={mlp[name]['deep_token_importance']:.4f}")
    
    # Attention comparison
    att = results['attention']['comparison']
    print("\nAttention Patterns:")
    print(f"  Token 0 attention: Hop={att['hop_attention_weight_0']:.4f}, Offset={att['offset_attention_weight_0']:.4f}, Delta={att['delta_attention_weight_0']:.4f}")
    print(f"  Deep attention: Hop={att['hop_deep_attention']:.4f}, Offset={att['offset_deep_attention']:.4f}, Delta={att['delta_deep_attention']:.4f}")
    
    print(f"\n{'='*60}\n")
    
    return results


if __name__ == '__main__':
    main()