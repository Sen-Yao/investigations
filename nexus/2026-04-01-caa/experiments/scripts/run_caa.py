#!/usr/bin/env python3
"""
CAA (Convergence-Aware Attention) Sweep Script

For hyperparameter tuning of CAA mechanism on Photo dataset.

Author: Nexus
Date: 2026-04-01
Updated: 2026-04-01 - Added AP (AUPRC) metric
"""

import sys
import os
import argparse
import random
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb

# Add mechanisms path
sys.path.insert(0, os.path.expanduser("~/VoxG/nexus/investigations/2026-03-31-offset-information-theory/experiments/mechanisms"))
from convergence_aware_attention import create_convergence_aware_gt


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_photo_dataset(data_path: str):
    """Load Photo dataset."""
    photo_path = os.path.join(data_path, "Photo.mat")
    
    if not os.path.exists(photo_path):
        raise FileNotFoundError(f"Photo dataset not found at {photo_path}")
    
    data = sio.loadmat(photo_path)
    
    features = data.get("Attributes", data.get("X", data.get("features")))
    labels = data.get("Label", data.get("Y", data.get("labels")))
    adj = data.get("Network", data.get("A", data.get("adj")))
    
    if sp.issparse(features):
        features = features.toarray()
    if sp.issparse(adj):
        adj = adj.toarray()
    
    labels = labels.flatten()
    if labels.min() > 0:
        labels = labels - labels.min()
    
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)
    adj = adj.astype(np.float32)
    
    return features, labels, adj


def compute_hop_features(features, adj, K=6):
    """Compute multi-hop aggregated features."""
    N, D = features.shape
    
    # Normalize adjacency
    degree = adj.sum(axis=1)
    degree_inv = np.where(degree > 0, 1.0 / degree, 0)
    adj_norm = adj * degree_inv[:, np.newaxis]
    
    hop_features = np.zeros((N, K + 1, D), dtype=np.float32)
    hop_features[:, 0, :] = features
    
    for k in range(1, K + 1):
        hop_features[:, k, :] = adj_norm @ hop_features[:, k - 1, :]
    
    return hop_features


def compute_delta_tokens(hop_features):
    """Compute Delta tokens (hop_t - hop_{t-1})."""
    delta_tokens = hop_features[:, 1:, :] - hop_features[:, :-1, :]
    return delta_tokens


def train_epoch(model, optimizer, data, device, batch_size=64):
    """Train one epoch."""
    X, y = data
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    for x, targets in loader:
        x, targets = x.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = model(x)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, data, device, batch_size=64):
    """Evaluate model, returns AUC and AP (AUPRC)."""
    X, y = data
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            outputs, _ = model(x)
            preds = torch.sigmoid(outputs.squeeze())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    return auc, ap


def main():
    parser = argparse.ArgumentParser(description="CAA Sweep")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="photo")
    parser.add_argument("--data_path", type=str, 
                        default=os.path.expanduser("~/VoxG/dataset"))
    
    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--ffn_dim", type=int, default=256)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--K", type=int, default=6)
    
    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize WandB
    wandb.init(project="voxg-caa", name=f"caa_{args.dataset}_h{args.hidden_dim}_lr{args.lr}")
    
    # Log config
    config = {
        "dataset": args.dataset,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "ffn_dim": args.ffn_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "K": args.K,
        "seed": args.seed
    }
    wandb.config.update(config)
    
    # Device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    features, labels, adj = load_photo_dataset(args.data_path)
    print(f"Dataset: {features.shape[0]} nodes, {features.shape[1]} features")
    
    # Compute Delta tokens
    hop_features = compute_hop_features(features, adj, K=args.K)
    delta_tokens = compute_delta_tokens(hop_features)
    
    input_dim = features.shape[1]
    num_tokens = args.K  # Delta tokens: K hops
    
    # Split data (5% labeled for semi-supervised)
    N = len(labels)
    train_val_idx, test_idx = train_test_split(
        np.arange(N), test_size=0.8, random_state=args.seed, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.5, random_state=args.seed, stratify=labels[train_val_idx]
    )
    
    train_data = (delta_tokens[train_idx], labels[train_idx])
    val_data = (delta_tokens[val_idx], labels[val_idx])
    test_data = (delta_tokens[test_idx], labels[test_idx])
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create CAA model
    model = create_convergence_aware_gt(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        num_tokens=num_tokens,
        dropout_rate=0.1,
        attention_dropout_rate=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_auc = 0.0
    best_test_auc = 0.0
    best_val_ap = 0.0
    best_test_ap = 0.0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, optimizer, train_data, device, args.batch_size)
        val_auc, val_ap = evaluate(model, val_data, device, args.batch_size)
        test_auc, test_ap = evaluate(model, test_data, device, args.batch_size)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auc": val_auc,
            "val_ap": val_ap,
            "test_auc": test_auc,
            "test_ap": test_ap
        })
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc = test_auc
            best_val_ap = val_ap
            best_test_ap = test_ap
            wandb.log({
                "AUC.max": best_test_auc,
                "AP.max": best_test_ap
            })
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, Val AUC={val_auc:.4f} AP={val_ap:.4f}, Test AUC={test_auc:.4f} AP={test_ap:.4f}")
    
    # Final log
    wandb.log({
        "final_val_auc": best_val_auc,
        "final_val_ap": best_val_ap,
        "final_test_auc": best_test_auc,
        "final_test_ap": best_test_ap
    })
    
    print(f"\nFinal Results:")
    print(f"  Best Val AUC: {best_val_auc:.4f}, AP: {best_val_ap:.4f}")
    print(f"  Best Test AUC: {best_test_auc:.4f}, AP: {best_test_ap:.4f}")
    
    wandb.finish()


if __name__ == "__main__":
    main()