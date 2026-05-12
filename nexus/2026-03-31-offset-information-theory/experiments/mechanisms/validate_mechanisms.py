#!/usr/bin/env python3
"""
Phase 4: Mechanism Validation Experiment

Quick validation (10 epochs) of three GT injection mechanisms:
1. Convergence-Aware Attention (CAA) - for Delta tokens
2. Stability Attention Bias (SAB) - for Offset tokens
3. Dual-Stream Architecture (DSA) - Offset + Delta fusion

Dataset: Photo (D=745, high-dimensional)

Author: Nexus
Date: 2026-03-31
"""

import sys
import os
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import scipy.io as sio
import scipy.sparse as sp

# Add mechanisms path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convergence_aware_attention import create_convergence_aware_gt
from stability_attention_bias import create_stability_aware_gt
from dual_stream_architecture import create_dual_stream_gt


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_photo_dataset(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Photo dataset from VoxG data directory.
    
    Args:
        data_path: Path to VoxG data directory
    
    Returns:
        features: [N, D] node features
        labels: [N] binary labels (0: normal, 1: anomaly)
        adj: [N, N] adjacency matrix
    """
    # Photo dataset path
    photo_path = os.path.join(data_path, "Photo.mat")
    
    if not os.path.exists(photo_path):
        raise FileNotFoundError(f"Photo dataset not found at {photo_path}")
    
    # Load .mat file
    data = sio.loadmat(photo_path)
    
    # Extract features, labels, and adjacency
    features = data.get("Attributes", data.get("X", data.get("features")))
    labels = data.get("Label", data.get("Y", data.get("labels")))
    adj = data.get("Network", data.get("A", data.get("adj")))
    
    # Convert to numpy arrays
    if sp.issparse(features):
        features = features.toarray()
    if sp.issparse(adj):
        adj = adj.toarray()
    
    # Flatten labels
    labels = labels.flatten()
    
    # Ensure binary labels
    if labels.min() > 0:
        labels = labels - labels.min()
    
    # Convert to proper types
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)
    adj = adj.astype(np.float32)
    
    print(f"Photo dataset loaded:")
    print(f"  - Features: {features.shape}")
    print(f"  - Labels: {labels.shape} (normal={labels.sum()}, anomaly={len(labels)-labels.sum()})")
    print(f"  - Adjacency: {adj.shape}")
    
    return features, labels, adj


def compute_hop_features(
    features: np.ndarray,
    adj: np.ndarray,
    K: int = 6,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute multi-hop aggregated features.
    
    Args:
        features: [N, D] node features
        adj: [N, N] adjacency matrix
        K: Number of hops
        normalize: Whether to normalize adjacency
    
    Returns:
        hop_features: [N, K+1, D] multi-hop features
    """
    N, D = features.shape
    
    # Normalize adjacency
    if normalize:
        degree = adj.sum(axis=1)
        degree_inv = np.where(degree > 0, 1.0 / degree, 0)
        adj_norm = adj * degree_inv[:, np.newaxis]
    else:
        adj_norm = adj
    
    # Initialize hop features
    hop_features = np.zeros((N, K + 1, D), dtype=np.float32)
    hop_features[:, 0, :] = features  # Hop 0 = original features
    
    # Compute multi-hop aggregations
    for k in range(1, K + 1):
        hop_features[:, k, :] = adj_norm @ hop_features[:, k - 1, :]
    
    return hop_features


def compute_offset_delta_tokens(
    hop_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Offset and Delta token features from Hop tokens.
    
    Args:
        hop_features: [N, K+1, D] multi-hop features
    
    Returns:
        offset_tokens: [N, K+1, D] Offset tokens (hop_t - hop_0)
        delta_tokens: [N, K, D] Delta tokens (hop_t - hop_{t-1})
    """
    N, K_plus_1, D = hop_features.shape
    
    # Offset: hop_t - hop_0
    offset_tokens = hop_features - hop_features[:, 0:1, :]
    
    # Delta: hop_t - hop_{t-1} (t >= 1)
    delta_tokens = hop_features[:, 1:, :] - hop_features[:, :-1, :]
    
    return offset_tokens, delta_tokens


def prepare_data_for_mechanisms(
    hop_features: np.ndarray,
    offset_tokens: np.ndarray,
    delta_tokens: np.ndarray,
    labels: np.ndarray,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Prepare data splits for all three mechanisms.
    
    Args:
        hop_features: [N, K+1, D]
        offset_tokens: [N, K+1, D]
        delta_tokens: [N, K, D]
        labels: [N]
        test_ratio: Test set ratio
        val_ratio: Validation set ratio
        seed: Random seed
    
    Returns:
        data_dict: Dictionary containing train/val/test splits for each mechanism
    """
    N = len(labels)
    K_plus_1 = hop_features.shape[1]
    K = delta_tokens.shape[1]
    
    # First split: train+val vs test
    indices = np.arange(N)
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=seed, stratify=labels
    )
    
    # Second split: train vs val
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio/(1-test_ratio), 
        random_state=seed, stratify=labels[train_val_idx]
    )
    
    # Prepare data for each mechanism
    data_dict = {
        "hop": {
            "train": (hop_features[train_idx], labels[train_idx]),
            "val": (hop_features[val_idx], labels[val_idx]),
            "test": (hop_features[test_idx], labels[test_idx])
        },
        "offset": {
            "train": (offset_tokens[train_idx], labels[train_idx]),
            "val": (offset_tokens[val_idx], labels[val_idx]),
            "test": (offset_tokens[test_idx], labels[test_idx])
        },
        "delta": {
            "train": (delta_tokens[train_idx], labels[train_idx]),
            "val": (delta_tokens[val_idx], labels[val_idx]),
            "test": (delta_tokens[test_idx], labels[test_idx])
        },
        "dual_stream": {
            # For dual-stream: Offset(0-3) + Delta(0-4)
            "train": (offset_tokens[train_idx, :4, :], delta_tokens[train_idx, :4, :], labels[train_idx]),
            "val": (offset_tokens[val_idx, :4, :], delta_tokens[val_idx, :4, :], labels[val_idx]),
            "test": (offset_tokens[test_idx, :4, :], delta_tokens[test_idx, :4, :], labels[test_idx])
        },
        "split_info": {
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "train_anomaly_ratio": labels[train_idx].sum() / len(train_idx),
            "test_anomaly_ratio": labels[test_idx].sum() / len(test_idx)
        }
    }
    
    return data_dict


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: Tuple,
    mechanism_type: str,
    device: torch.device,
    batch_size: int = 64
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        optimizer: Optimizer
        data: Training data tuple
        mechanism_type: Type of mechanism ("hop", "offset", "delta", "dual_stream")
        device: Device to use
        batch_size: Batch size
    
    Returns:
        avg_loss: Average loss for epoch
    """
    if mechanism_type == "dual_stream":
        X1, X2, y = data
        dataset = TensorDataset(
            torch.FloatTensor(X1), 
            torch.FloatTensor(X2), 
            torch.FloatTensor(y)
        )
    else:
        X, y = data
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    criterion = nn.BCEWithLogitsLoss()
    
    for batch in loader:
        optimizer.zero_grad()
        
        if mechanism_type == "dual_stream":
            x1, x2, targets = batch
            x1, x2, targets = x1.to(device), x2.to(device), targets.to(device)
            outputs, _ = model(x1, x2)
        else:
            x, targets = batch
            x, targets = x.to(device), targets.to(device)
            outputs, _ = model(x)
        
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_model(
    model: nn.Module,
    data: Tuple,
    mechanism_type: str,
    device: torch.device,
    batch_size: int = 64
) -> Dict:
    """
    Evaluate model performance.
    
    Args:
        model: Model to evaluate
        data: Evaluation data tuple
        mechanism_type: Type of mechanism
        device: Device to use
        batch_size: Batch size
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    if mechanism_type == "dual_stream":
        X1, X2, y = data
        dataset = TensorDataset(
            torch.FloatTensor(X1), 
            torch.FloatTensor(X2), 
            torch.FloatTensor(y)
        )
    else:
        X, y = data
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_attentions = []
    
    with torch.no_grad():
        for batch in loader:
            if mechanism_type == "dual_stream":
                x1, x2, labels = batch
                x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
                outputs, attentions = model(x1, x2)
            else:
                x, labels = batch
                x, labels = x.to(device), labels.to(device)
                outputs, attentions = model(x)
            
            preds = torch.sigmoid(outputs.squeeze())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if attentions is not None:
                all_attentions.append(attentions)
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    auc = roc_auc_score(all_labels, all_preds)
    pred_labels = (all_preds > 0.5).astype(int)
    
    f1 = f1_score(all_labels, pred_labels)
    precision = precision_score(all_labels, pred_labels, zero_division=0)
    recall = recall_score(all_labels, pred_labels, zero_division=0)
    
    # Compute attention statistics
    attention_stats = compute_attention_statistics(all_attentions, mechanism_type)
    
    return {
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "attention_stats": attention_stats
    }


def compute_attention_statistics(
    attentions: List,
    mechanism_type: str
) -> Dict:
    """
    Compute attention weight statistics.
    
    Args:
        attentions: List of attention tensors
        mechanism_type: Type of mechanism
    
    Returns:
        stats: Dictionary of attention statistics
    """
    if not attentions:
        return {}
    
    stats = {}
    
    if mechanism_type == "dual_stream":
        # For dual-stream, get final layer attention
        if len(attentions) > 0:
            last_attn = attentions[-1]
            if isinstance(last_attn, dict) and "gate" in last_attn:
                gate = last_attn["gate"]
                if isinstance(gate, torch.Tensor):
                    stats["mean_gate"] = gate.mean().item()
                    stats["std_gate"] = gate.std().item()
    else:
        # For single-stream mechanisms, get mean attention to tokens
        if len(attentions) > 0:
            last_attn = attentions[-1]
            if isinstance(last_attn, torch.Tensor) and last_attn.dim() == 4:
                # [B, H, T, T] -> mean over batch and heads
                mean_attn = last_attn.mean(dim=(0, 1)).cpu().numpy()
                stats["token0_attention"] = mean_attn[:, 0].mean()
                
                # Deep token attention (tokens 4-6)
                if mean_attn.shape[0] > 4:
                    deep_attn = mean_attn[:, 4:].mean()
                    stats["deep_attention"] = deep_attn
    
    return stats


# ============================================================================
# Experiment Runner
# ============================================================================

def run_validation_experiment(
    data_path: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    ffn_dim: int = 256,
    lr: float = 0.001,
    K: int = 6,
    seed: int = 42,
    device: int = 0
) -> Dict:
    """
    Run validation experiment for all three mechanisms.
    
    Args:
        data_path: Path to VoxG data
        output_dir: Output directory for results
        epochs: Number of training epochs
        batch_size: Batch size
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        ffn_dim: FFN dimension
        lr: Learning rate
        K: Number of hops
        seed: Random seed
        device: GPU device ID
    
    Returns:
        results: Dictionary of results for all mechanisms
    """
    print("=" * 60)
    print("Phase 4: Mechanism Validation Experiment")
    print("=" * 60)
    print(f"Dataset: Photo")
    print(f"Epochs: {epochs}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Device: cuda:{device}")
    print("=" * 60)
    
    # Set device
    torch_device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch_device}")
    
    # Load data
    print("\n[1] Loading Photo dataset...")
    features, labels, adj = load_photo_dataset(data_path)
    
    # Compute token features
    print("\n[2] Computing Hop, Offset, and Delta tokens...")
    hop_features = compute_hop_features(features, adj, K=K)
    offset_tokens, delta_tokens = compute_offset_delta_tokens(hop_features)
    
    print(f"  - Hop tokens: {hop_features.shape}")
    print(f"  - Offset tokens: {offset_tokens.shape}")
    print(f"  - Delta tokens: {delta_tokens.shape}")
    
    # Prepare data splits
    print("\n[3] Preparing data splits...")
    data_dict = prepare_data_for_mechanisms(
        hop_features, offset_tokens, delta_tokens, labels,
        test_ratio=0.2, val_ratio=0.1, seed=seed
    )
    
    print(f"  - Train: {data_dict['split_info']['train_size']} samples")
    print(f"  - Val: {data_dict['split_info']['val_size']} samples")
    print(f"  - Test: {data_dict['split_info']['test_size']} samples")
    
    # Get input dimension
    input_dim = features.shape[1]
    num_tokens = K + 1
    
    results = {}
    
    # ============================================================================
    # Mechanism A: Convergence-Aware Attention (for Delta tokens)
    # ============================================================================
    print("\n" + "=" * 60)
    print("[A] Testing Convergence-Aware Attention (CAA)")
    print("=" * 60)
    
    # Create model for Delta tokens
    caa_model = create_convergence_aware_gt(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_tokens=K,  # Delta has K tokens
        dropout_rate=0.1,
        attention_dropout_rate=0.1
    ).to(torch_device)
    
    print(f"Model created: {sum(p.numel() for p in caa_model.parameters())} parameters")
    
    optimizer = torch.optim.Adam(caa_model.parameters(), lr=lr)
    
    # Training loop
    caa_results = {"train_losses": [], "val_metrics": []}
    
    print("\nTraining CAA model...")
    for epoch in range(epochs):
        train_loss = train_epoch(
            caa_model, optimizer, data_dict["delta"]["train"],
            "delta", torch_device, batch_size
        )
        caa_results["train_losses"].append(train_loss)
        
        val_metrics = evaluate_model(
            caa_model, data_dict["delta"]["val"],
            "delta", torch_device, batch_size
        )
        caa_results["val_metrics"].append(val_metrics)
        
        print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Val AUC={val_metrics['auc']:.4f}")
    
    # Final evaluation
    test_metrics = evaluate_model(
        caa_model, data_dict["delta"]["test"],
        "delta", torch_device, batch_size
    )
    caa_results["test_metrics"] = test_metrics
    
    print(f"\nCAA Test Results:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    
    results["CAA"] = caa_results
    
    # ============================================================================
    # Mechanism B: Stability Attention Bias (for Offset tokens)
    # ============================================================================
    print("\n" + "=" * 60)
    print("[B] Testing Stability Attention Bias (SAB)")
    print("=" * 60)
    
    # Create model for Offset tokens
    sab_model = create_stability_aware_gt(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_tokens=num_tokens,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        token0_penalty=0.3
    ).to(torch_device)
    
    print(f"Model created: {sum(p.numel() for p in sab_model.parameters())} parameters")
    
    optimizer = torch.optim.Adam(sab_model.parameters(), lr=lr)
    
    # Training loop
    sab_results = {"train_losses": [], "val_metrics": []}
    
    print("\nTraining SAB model...")
    for epoch in range(epochs):
        train_loss = train_epoch(
            sab_model, optimizer, data_dict["offset"]["train"],
            "offset", torch_device, batch_size
        )
        sab_results["train_losses"].append(train_loss)
        
        val_metrics = evaluate_model(
            sab_model, data_dict["offset"]["val"],
            "offset", torch_device, batch_size
        )
        sab_results["val_metrics"].append(val_metrics)
        
        print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Val AUC={val_metrics['auc']:.4f}")
    
    # Final evaluation
    test_metrics = evaluate_model(
        sab_model, data_dict["offset"]["test"],
        "offset", torch_device, batch_size
    )
    sab_results["test_metrics"] = test_metrics
    
    print(f"\nSAB Test Results:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    
    # Check Token 0 attention
    if "token0_attention" in test_metrics["attention_stats"]:
        token0_attn = test_metrics["attention_stats"]["token0_attention"]
        print(f"  Token 0 attention: {token0_attn:.4f} (should be lower than baseline ~0.48)")
    
    results["SAB"] = sab_results
    
    # ============================================================================
    # Mechanism C: Dual-Stream Architecture
    # ============================================================================
    print("\n" + "=" * 60)
    print("[C] Testing Dual-Stream Architecture (DSA)")
    print("=" * 60)
    
    # Create dual-stream model
    dsa_model = create_dual_stream_gt(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_tokens=num_tokens,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        use_cross_attention=True,
        token_split=(4, 4)  # Offset(0-3) + Delta(0-3)
    ).to(torch_device)
    
    print(f"Model created: {sum(p.numel() for p in dsa_model.parameters())} parameters")
    
    optimizer = torch.optim.Adam(dsa_model.parameters(), lr=lr)
    
    # Training loop
    dsa_results = {"train_losses": [], "val_metrics": []}
    
    print("\nTraining DSA model...")
    for epoch in range(epochs):
        train_loss = train_epoch(
            dsa_model, optimizer, data_dict["dual_stream"]["train"],
            "dual_stream", torch_device, batch_size
        )
        dsa_results["train_losses"].append(train_loss)
        
        val_metrics = evaluate_model(
            dsa_model, data_dict["dual_stream"]["val"],
            "dual_stream", torch_device, batch_size
        )
        dsa_results["val_metrics"].append(val_metrics)
        
        print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Val AUC={val_metrics['auc']:.4f}")
    
    # Final evaluation
    test_metrics = evaluate_model(
        dsa_model, data_dict["dual_stream"]["test"],
        "dual_stream", torch_device, batch_size
    )
    dsa_results["test_metrics"] = test_metrics
    
    print(f"\nDSA Test Results:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    
    # Check gate values
    if "mean_gate" in test_metrics["attention_stats"]:
        gate = test_metrics["attention_stats"]["mean_gate"]
        print(f"  Gate value (stable weight): {gate:.4f}")
    
    results["DSA"] = dsa_results
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 60)
    print("Validation Experiment Summary")
    print("=" * 60)
    
    summary_table = []
    for mechanism, res in results.items():
        test_auc = res["test_metrics"]["auc"]
        test_f1 = res["test_metrics"]["f1"]
        summary_table.append({
            "Mechanism": mechanism,
            "Test AUC": f"{test_auc:.4f}",
            "Test F1": f"{test_f1:.4f}",
            "Best Val AUC": f"{max(m['auc'] for m in res['val_metrics']):.4f}"
        })
    
    # Print table
    print(f"\n{'Mechanism':<15} {'Test AUC':<12} {'Test F1':<12} {'Best Val AUC':<12}")
    print("-" * 51)
    for row in summary_table:
        print(f"{row['Mechanism']:<15} {row['Test AUC']:<12} {row['Test F1']:<12} {row['Best Val AUC']:<12}")
    
    # Save results
    results["summary"] = summary_table
    results["config"] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ffn_dim": ffn_dim,
        "lr": lr,
        "K": K,
        "seed": seed,
        "device": device,
        "input_dim": input_dim,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to file
    output_file = os.path.join(output_dir, "phase4_validation_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Mechanism Validation")
    
    parser.add_argument("--data_path", type=str, 
                        default="/root/gpufree-data/linziyao/VoxG/data",
                        help="Path to VoxG data directory")
    parser.add_argument("--output_dir", type=str,
                        default="/root/gpufree-data/linziyao/VoxG/nexus/investigations/2026-03-31-offset-information-theory/experiments/outputs",
                        help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of encoder layers")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--ffn_dim", type=int, default=256,
                        help="FFN dimension")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--K", type=int, default=6,
                        help="Number of hops")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device ID")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    results = run_validation_experiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        lr=args.lr,
        K=args.K,
        seed=args.seed,
        device=args.device
    )