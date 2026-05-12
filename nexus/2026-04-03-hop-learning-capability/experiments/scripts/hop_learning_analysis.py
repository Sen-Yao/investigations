#!/usr/bin/env python3
"""
Hop Token Learning Capability Analysis

This script validates whether Transformer can automatically learn Delta from Hop.

Methods:
1. Linear Separability Test - Can Delta be linearly extracted from Hop?
2. Mutual Information Analysis - How much information does Hop contain about Delta?
3. Probing Task - Can a frozen Transformer extract Delta?

Author: Nexus
Date: 2026-04-03
"""

import sys
import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
import warnings
warnings.filterwarnings('ignore')


def load_dataset(dataset: str, data_path: str):
    """Load dataset from .mat file."""
    # Support both case variations
    dataset_map = {
        "photo": ["photo.mat", "Photo.mat"],
        "amazon": ["amazon.mat", "Amazon.mat"],
        "elliptic": ["elliptic.mat", "Elliptic.mat"],
        "tolokers": ["tolokers.mat", "Tolokers.mat"],
    }
    
    possible_names = dataset_map.get(dataset.lower(), [f"{dataset}.mat", f"{dataset.capitalize()}.mat"])
    
    data_file = None
    for name in possible_names:
        candidate = os.path.join(data_path, name)
        if os.path.exists(candidate):
            data_file = candidate
            break
    
    if data_file is None:
        raise FileNotFoundError(f"Dataset not found at {data_path}. Tried: {possible_names}")
    
    data = sio.loadmat(data_file)
    
    # Extract features
    features = data.get("Attributes", data.get("X", data.get("features", data.get("node_feat"))))
    if features is None:
        raise KeyError(f"Features not found")
    
    # Extract labels
    labels = data.get("Label", data.get("Y", data.get("labels", data.get("node_label"))))
    if labels is None:
        raise KeyError(f"Labels not found")
    
    # Extract adjacency
    adj = data.get("Network", data.get("A", data.get("adj", data.get("edge_index"))))
    if adj is None:
        raise KeyError(f"Adjacency not found")
    
    # Convert sparse to dense
    if sp.issparse(features):
        features = features.toarray()
    if sp.issparse(adj):
        adj = adj.toarray()
    
    # Handle edge_index format
    if adj.shape[0] == 2 and adj.shape[1] > 2:
        N = features.shape[0]
        adj_dense = np.zeros((N, N), dtype=np.float32)
        edges = adj.astype(np.int64)
        for i in range(edges.shape[1]):
            src, dst = edges[0, i], edges[1, i]
            if src < N and dst < N:
                adj_dense[src, dst] = 1
                adj_dense[dst, src] = 1
        adj = adj_dense
    
    labels = labels.flatten()
    if labels.min() > 0:
        labels = labels - labels.min()
    
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)
    adj = adj.astype(np.float32)
    
    return features, labels, adj


def compute_hop_features(features, adj, K=6, alpha=0.0):
    """Compute multi-hop aggregated features with PPR-style."""
    N, D = features.shape
    
    # Symmetric normalization: D^{-0.5} A D^{-0.5}
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
    d_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt
    
    hop_features = np.zeros((N, K + 1, D), dtype=np.float32)
    hop_features[:, 0, :] = features
    
    agg = features.copy()
    for k in range(1, K + 1):
        # Pure neighbor aggregation (alpha=0)
        agg = adj_norm @ agg
        hop_features[:, k, :] = agg
    
    return hop_features


def compute_delta_tokens(hop_features):
    """Compute Delta tokens: delta_k = hop_k - hop_{k-1}."""
    delta_tokens = hop_features[:, 1:, :] - hop_features[:, :-1, :]
    return delta_tokens


# ============================================================
# Method 1: Linear Separability Test
# ============================================================

def linear_separability_test(hop_features, delta_tokens, sample_ratio=0.5):
    """
    Test if Delta can be linearly extracted from Hop.
    
    For each hop k, we test if:
        delta_k = W @ hop_k + b
    can be learned with high R².
    """
    N, K_plus_1, D = hop_features.shape
    K = K_plus_1 - 1
    
    results = {
        'per_hop_r2': [],
        'per_hop_mse': [],
        'overall_r2': None,
        'overall_mse': None,
    }
    
    # Sample nodes for efficiency
    n_samples = int(N * sample_ratio)
    indices = np.random.choice(N, n_samples, replace=False)
    
    all_hops = []
    all_deltas = []
    
    for k in range(K):
        hop_k = hop_features[indices, k, :]  # Input: hop_k
        delta_k = delta_tokens[indices, k, :]  # Target: delta_k
        
        # Train linear regression
        model = Ridge(alpha=1.0)
        model.fit(hop_k, delta_k)
        
        # Predict
        delta_pred = model.predict(hop_k)
        
        # Compute metrics
        r2 = r2_score(delta_k.flatten(), delta_pred.flatten())
        mse = mean_squared_error(delta_k.flatten(), delta_pred.flatten())
        
        results['per_hop_r2'].append(r2)
        results['per_hop_mse'].append(mse)
        
        all_hops.append(hop_k)
        all_deltas.append(delta_k)
    
    # Overall test: predict all deltas from all hops
    all_hops = np.concatenate(all_hops, axis=0)  # (N*K, D)
    all_deltas = np.concatenate(all_deltas, axis=0)  # (N*K, D)
    
    model = Ridge(alpha=1.0)
    model.fit(all_hops, all_deltas)
    delta_pred = model.predict(all_hops)
    
    results['overall_r2'] = r2_score(all_deltas.flatten(), delta_pred.flatten())
    results['overall_mse'] = mean_squared_error(all_deltas.flatten(), delta_pred.flatten())
    
    return results


# ============================================================
# Method 2: Mutual Information Analysis
# ============================================================

def estimate_entropy_continuous(data, n_bins=50):
    """Estimate entropy for continuous data using histogram."""
    N, D = data.shape
    
    # Use only first few dimensions for efficiency
    D_use = min(D, 10)
    data_use = data[:, :D_use]
    
    # Discretize
    data_discrete = np.zeros_like(data_use, dtype=np.int32)
    for d in range(D_use):
        data_discrete[:, d] = np.digitize(data_use[:, d], 
                                          bins=np.linspace(data_use[:, d].min(), 
                                                          data_use[:, d].max(), 
                                                          n_bins))
    
    # Count joint frequency
    joint = {}
    for i in range(N):
        key = tuple(data_discrete[i])
        joint[key] = joint.get(key, 0) + 1
    
    # Compute entropy
    counts = np.array(list(joint.values()))
    probs = counts / N
    H = -np.sum(probs * np.log(probs + 1e-10))
    
    return H


def mutual_information_analysis(hop_features, delta_tokens, n_samples=1000):
    """
    Compute mutual information between Hop and Delta.
    
    I(Hop; Delta) = H(Delta) - H(Delta | Hop)
    
    Approximation: I(X; Y) ≈ H(X) + H(Y) - H(X, Y)
    """
    N, K_plus_1, D = hop_features.shape
    K = K_plus_1 - 1
    
    # Sample for efficiency
    indices = np.random.choice(N, min(N, n_samples), replace=False)
    
    results = {
        'per_hop_mi': [],
        'H_delta': [],
        'H_hop': [],
    }
    
    for k in range(K):
        hop_k = hop_features[indices, k, :]
        delta_k = delta_tokens[indices, k, :]
        
        # Estimate entropies
        H_hop = estimate_entropy_continuous(hop_k)
        H_delta = estimate_entropy_continuous(delta_k)
        
        # Joint entropy (concatenate)
        joint = np.concatenate([hop_k, delta_k], axis=1)
        H_joint = estimate_entropy_continuous(joint)
        
        # Mutual information
        MI = H_hop + H_delta - H_joint
        
        results['per_hop_mi'].append(max(0, MI))  # MI should be non-negative
        results['H_delta'].append(H_delta)
        results['H_hop'].append(H_hop)
    
    return results


# ============================================================
# Method 3: Probing Task
# ============================================================

class SimpleProbe(nn.Module):
    """Linear probe for Delta prediction."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


def probing_task(hop_features, delta_tokens, test_ratio=0.2, epochs=100, lr=0.01):
    """
    Test if a linear probe can extract Delta from Hop.
    
    This tests whether the representation is linearly extractable.
    """
    N, K_plus_1, D = hop_features.shape
    K = K_plus_1 - 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {
        'per_hop_r2': [],
        'per_hop_mse': [],
        'train_loss_history': [],
    }
    
    for k in range(K):
        hop_k = hop_features[:, k, :]
        delta_k = delta_tokens[:, k, :]
        
        # Split train/test
        n_train = int(N * (1 - test_ratio))
        indices = np.random.permutation(N)
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        
        X_train = torch.FloatTensor(hop_k[train_idx]).to(device)
        y_train = torch.FloatTensor(delta_k[train_idx]).to(device)
        X_test = torch.FloatTensor(hop_k[test_idx]).to(device)
        y_test = torch.FloatTensor(delta_k[test_idx]).to(device)
        
        # Create probe
        probe = SimpleProbe(D, D).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Train
        probe.train()
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = probe(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        results['train_loss_history'].append(losses)
        
        # Evaluate
        probe.eval()
        with torch.no_grad():
            pred_test = probe(X_test).cpu().numpy()
            y_test_np = y_test.cpu().numpy()
            
            r2 = r2_score(y_test_np.flatten(), pred_test.flatten())
            mse = mean_squared_error(y_test_np.flatten(), pred_test.flatten())
        
        results['per_hop_r2'].append(r2)
        results['per_hop_mse'].append(mse)
    
    return results


# ============================================================
# Main Analysis
# ============================================================

def run_analysis(dataset_name, data_path, K=6, seed=42):
    """Run all three analysis methods on a dataset."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load data
    features, labels, adj = load_dataset(dataset_name, data_path)
    N, D = features.shape
    print(f"Nodes: {N}, Features: {D}")
    
    # Compute Hop and Delta
    hop_features = compute_hop_features(features, adj, K=K)
    delta_tokens = compute_delta_tokens(hop_features)
    print(f"Hop shape: {hop_features.shape}")
    print(f"Delta shape: {delta_tokens.shape}")
    
    # ========================
    # Method 1: Linear Separability
    # ========================
    print(f"\n--- Method 1: Linear Separability Test ---")
    linear_results = linear_separability_test(hop_features, delta_tokens)
    
    print(f"Per-hop R²: {linear_results['per_hop_r2']}")
    print(f"Overall R²: {linear_results['overall_r2']:.4f}")
    
    # ========================
    # Method 2: Mutual Information
    # ========================
    print(f"\n--- Method 2: Mutual Information Analysis ---")
    mi_results = mutual_information_analysis(hop_features, delta_tokens)
    
    print(f"Per-hop MI: {mi_results['per_hop_mi']}")
    print(f"Avg MI: {np.mean(mi_results['per_hop_mi']):.4f}")
    
    # ========================
    # Method 3: Probing Task
    # ========================
    print(f"\n--- Method 3: Probing Task ---")
    probe_results = probing_task(hop_features, delta_tokens)
    
    print(f"Per-hop R²: {probe_results['per_hop_r2']}")
    print(f"Avg R²: {np.mean(probe_results['per_hop_r2']):.4f}")
    
    # ========================
    # Summary
    # ========================
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Linear Separability R²: {linear_results['overall_r2']:.4f}")
    print(f"Mutual Information:     {np.mean(mi_results['per_hop_mi']):.4f}")
    print(f"Probing Task R²:        {np.mean(probe_results['per_hop_r2']):.4f}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print(f"INTERPRETATION")
    print(f"{'='*60}")
    
    if linear_results['overall_r2'] > 0.9:
        print("✅ Delta is LINEARLY EXTRACTABLE from Hop")
        print("   → Transformer CAN learn Delta without explicit construction")
    elif linear_results['overall_r2'] > 0.5:
        print("⚠️  Delta is PARTIALLY linearly extractable")
        print("   → Some benefit from explicit Delta construction")
    else:
        print("❌ Delta is NOT linearly extractable")
        print("   → Explicit Delta construction is VALUABLE")
    
    return {
        'dataset': dataset_name,
        'linear_r2': linear_results['overall_r2'],
        'mi': np.mean(mi_results['per_hop_mi']),
        'probe_r2': np.mean(probe_results['per_hop_r2']),
    }


def main():
    """Main entry point."""
    # Try multiple possible data paths
    possible_paths = [
        "/root/gpufree-data/linziyao/GGADFormer/dataset",
        os.path.expanduser("~/VoxG/data"),
        os.path.expanduser("~/VoxG/dataset"),
    ]
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    if data_path is None:
        raise FileNotFoundError("Could not find data directory")
    
    datasets = ['photo', 'elliptic', 'tolokers']
    
    all_results = []
    for dataset in datasets:
        try:
            result = run_analysis(dataset, data_path)
            all_results.append(result)
        except Exception as e:
            print(f"Error on {dataset}: {e}")
    
    # Final comparison
    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON ACROSS DATASETS")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} {'Linear R²':<15} {'MI':<15} {'Probe R²':<15}")
    print(f"{'-'*60}")
    for r in all_results:
        print(f"{r['dataset']:<15} {r['linear_r2']:<15.4f} {r['mi']:<15.4f} {r['probe_r2']:<15.4f}")
    
    print(f"\n{'='*80}")
    print(f"CONCLUSION")
    print(f"{'='*80}")
    avg_r2 = np.mean([r['linear_r2'] for r in all_results])
    if avg_r2 > 0.8:
        print("✅ OVERALL: Delta IS linearly extractable from Hop")
        print("   → Transformer CAN learn Delta automatically")
        print("   → Recommendation: Use original Hop, no need for explicit Delta")
    else:
        print("❌ OVERALL: Delta is NOT consistently linearly extractable")
        print("   → Explicit Delta construction has VALUE")
        print("   → Recommendation: Continue exploring explicit feature design")


if __name__ == "__main__":
    main()