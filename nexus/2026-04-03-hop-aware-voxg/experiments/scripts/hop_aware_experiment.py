#!/usr/bin/env python3
"""
Hop-Aware VoxGFormer Experiment Script

简化版实验脚本，用于快速验证 Hop-Aware Attention 效果

遵循：
1. train_rate=0.05（与 VecGAD 一致）
2. 5-seed 验证
3. WandB logging
4. AUC/AP 评估
"""

import sys
import os
import argparse
import random
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb
import math

# ===== Hop-Aware Multi-Head Attention =====

class HopAwareMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, max_hop=7):
        super().__init__()
        
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        self.scale = 1 / math.sqrt(self.att_size)
        self.max_hop = max_hop
        
        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * self.att_size, hidden_size)
        
        # Hop 相对距离偏置
        self.hop_bias = nn.Parameter(torch.zeros(2 * max_hop - 1))
        
    def forward(self, q, k, v, hop_indices=None):
        batch_size = q.size(0)
        seq_len = q.size(1)
        
        d_k = self.att_size
        d_v = self.att_size
        
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2).transpose(2, 3)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v).transpose(1, 2)
        
        q = q * self.scale
        x = torch.matmul(q, k)
        
        # Hop-Aware Bias
        if hop_indices is not None:
            hop_bias_matrix = self._build_hop_bias_matrix(hop_indices, seq_len)
            x = x + hop_bias_matrix.unsqueeze(0).unsqueeze(0)
        
        attention_weights = torch.softmax(x, dim=3)
        x = self.att_dropout(attention_weights)
        x = x.matmul(v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * d_v)
        
        return self.output_layer(x), attention_weights
    
    def _build_hop_bias_matrix(self, hop_indices, seq_len):
        hop_i = hop_indices.unsqueeze(1)
        hop_j = hop_indices.unsqueeze(0)
        hop_diff = hop_j - hop_i
        bias_idx = hop_diff + (self.max_hop - 1)
        bias_idx = bias_idx.clamp(0, 2 * self.max_hop - 2)
        return self.hop_bias[bias_idx]


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)
    
    def forward(self, x):
        return self.layer2(self.gelu(self.layer1(x)))


class HopAwareEncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, max_hop=7):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = HopAwareMultiHeadAttention(hidden_size, attention_dropout_rate, num_heads, max_hop)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, hop_indices=None):
        y = self.self_attention_norm(x)
        y, attn = self.self_attention(y, y, y, hop_indices)
        y = self.self_attention_dropout(y)
        x = x + y
        
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        
        return x, attn


# ===== Hop-Aware VoxGFormer Model =====

class HopAwareVoxGFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=2, 
                 ffn_dim=256, dropout=0.1, attention_dropout=0.1, max_hop=7):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            HopAwareEncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout, num_heads, max_hop)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 伪异常生成参数
        self.noise_mean = 0.02
        self.noise_var = 0.01
        
    def forward(self, hop_tokens, hop_indices=None, normal_idx=None, train_flag=True):
        """
        Args:
            hop_tokens: [N, pp_k+1, d] - hop features
            hop_indices: [pp_k+1] - hop indices (0, 1, 2, ..., pp_k)
            normal_idx: indices of normal nodes for training
            train_flag: whether in training mode
        """
        N, num_hops, d = hop_tokens.shape
        device = hop_tokens.device
        
        if hop_indices is None:
            hop_indices = torch.arange(num_hops, device=device)
        
        # Project input
        x = self.input_proj(hop_tokens)  # [N, pp_k+1, hidden_dim]
        
        # Pass through layers
        for layer in self.layers:
            x, _ = layer(x, hop_indices)
        
        x = self.final_ln(x)
        
        # Pooling: use 0-hop attention scores (简单平均)
        node_repr = x[:, 0, :]  # [N, hidden_dim] - 使用 hop_0 作为节点表示
        
        # Classification
        logits = self.classifier(node_repr).squeeze(-1)  # [N]
        
        # Generate pseudo anomalies if training
        if train_flag and normal_idx is not None:
            # Sample normal nodes
            num_pseudo = int(len(normal_idx) * 0.2)
            pseudo_idx = normal_idx[:num_pseudo]
            
            # Add noise
            pseudo_repr = node_repr[pseudo_idx]
            noise = torch.randn_like(pseudo_repr) * self.noise_var + self.noise_mean
            pseudo_repr_noised = pseudo_repr + noise
            
            # Combine for BCE loss
            combined_repr = torch.cat([node_repr[normal_idx], pseudo_repr_noised])
            combined_logits = self.classifier(combined_repr).squeeze(-1)
            
            return logits, combined_logits, normal_idx, pseudo_idx
        
        return logits


# ===== Data Loading =====

def load_dataset(dataset_name, data_path):
    """Load dataset"""
    dataset_map = {
        "photo": "Photo.mat",
        "amazon": "Amazon.mat",
        "elliptic": "Elliptic.mat",
        "tolokers": "Tolokers.mat",
        "tfinance": "T-Finance.mat"
    }
    
    mat_file = dataset_map.get(dataset_name.lower(), f"{dataset_name}.mat")
    data_file = os.path.join(data_path, mat_file)
    
    data = sio.loadmat(data_file)
    
    features = data.get("Attributes", data.get("X", data.get("features")))
    labels = data.get("Label", data.get("Y", data.get("labels"))).flatten()
    adj = data.get("Network", data.get("A", data.get("adj")))
    
    if sp.issparse(features):
        features = features.toarray()
    if sp.issparse(adj):
        adj = adj.toarray()
    
    # Normalize labels
    if labels.min() > 0:
        labels = labels - labels.min()
    
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)
    adj = adj.astype(np.float32)
    
    return features, labels, adj


def compute_hop_features(features, adj, K=6):
    """Compute hop features"""
    N, D = features.shape
    
    # Row-normalize adjacency
    degree = adj.sum(axis=1)
    degree_inv = np.where(degree > 0, 1.0 / degree, 0)
    adj_norm = adj * degree_inv[:, np.newaxis]
    
    hop_features = np.zeros((N, K + 1, D), dtype=np.float32)
    hop_features[:, 0, :] = features
    
    for k in range(1, K + 1):
        hop_features[:, k, :] = adj_norm @ hop_features[:, k - 1, :]
    
    return hop_features


def split_data(labels, train_rate=0.05, val_rate=0.1, seed=42):
    """Split data with semi-supervised setting"""
    np.random.seed(seed)
    
    N = len(labels)
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    
    # Train: only normal nodes
    n_train = int(len(normal_idx) * train_rate)
    train_idx = np.random.choice(normal_idx, n_train, replace=False)
    
    # Val: from remaining nodes
    remaining_normal = np.setdiff1d(normal_idx, train_idx)
    n_val = int(N * val_rate)
    val_normal = np.random.choice(remaining_normal, n_val // 2, replace=False)
    val_anomaly = np.random.choice(anomaly_idx, n_val // 2, replace=False)
    val_idx = np.concatenate([val_normal, val_anomaly])
    
    # Test: remaining
    test_idx = np.setdiff1d(np.arange(N), np.concatenate([train_idx, val_idx]))
    
    # Verify
    assert np.sum(labels[train_idx]) == 0, "Train contains anomalies!"
    
    return train_idx, val_idx, test_idx


# ===== Training =====

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, optimizer, hop_features, labels, train_idx, device, batch_size=256):
    """Train one epoch"""
    model.train()
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Create batches
    N = len(train_idx)
    total_loss = 0.0
    
    for i in range(0, N, batch_size):
        batch_idx = train_idx[i:i+batch_size]
        
        batch_tokens = torch.FloatTensor(hop_features[batch_idx]).to(device)
        batch_labels = torch.FloatTensor(labels[batch_idx]).to(device)
        
        optimizer.zero_grad()
        
        logits, combined_logits, normal_local_idx, pseudo_idx = model(
            batch_tokens, 
            normal_idx=torch.arange(len(batch_idx)), 
            train_flag=True
        )
        
        # BCE loss: normal=0, pseudo=1
        bce_labels = torch.zeros(len(batch_idx) + len(pseudo_idx)).to(device)
        bce_labels[len(batch_idx):] = 1.0
        
        loss = criterion(combined_logits, bce_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / (N // batch_size + 1)


def evaluate(model, hop_features, labels, eval_idx, device, batch_size=256):
    """Evaluate AUC and AP"""
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, len(eval_idx), batch_size):
            batch_idx = eval_idx[i:i+batch_size]
            
            batch_tokens = torch.FloatTensor(hop_features[batch_idx]).to(device)
            
            logits = model(batch_tokens, train_flag=False)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels[batch_idx])
    
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    
    return auc, ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--data_path', type=str, default='/root/gpufree-data/linziyao/VoxG/dataset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pp_k', type=int, default=6)
    parser.add_argument('--max_hop', type=int, default=7)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_hop_bias', type=bool, default=True)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading {args.dataset}...")
    features, labels, adj = load_dataset(args.dataset, args.data_path)
    
    print(f"  Nodes: {features.shape[0]}, Features: {features.shape[1]}, Anomalies: {labels.sum()} ({labels.mean()*100:.2f}%)")
    
    # Compute hop features
    print("Computing hop features...")
    hop_features = compute_hop_features(features, adj, K=args.pp_k)
    print(f"  Hop features shape: {hop_features.shape}")
    
    # Split data
    train_idx, val_idx, test_idx = split_data(labels, train_rate=args.train_rate, seed=args.seed)
    print(f"  Train: {len(train_idx)} (normal), Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Model
    model = HopAwareVoxGFormer(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_hop=args.max_hop
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # WandB
    wandb.init(
        project="hop-aware-voxg",
        entity="HCCS",
        config=args,
        name=f"{args.dataset}-seed{args.seed}"
    )
    
    # Training
    print("Training...")
    best_val_auc = 0
    best_test_auc = 0
    best_test_ap = 0
    
    for epoch in range(args.num_epochs):
        loss = train_epoch(model, optimizer, hop_features, labels, train_idx, device, args.batch_size)
        
        val_auc, val_ap = evaluate(model, hop_features, labels, val_idx, device)
        test_auc, test_ap = evaluate(model, hop_features, labels, test_idx, device)
        
        wandb.log({
            "epoch": epoch,
            "loss": loss,
            "val_auc": val_auc,
            "val_ap": val_ap,
            "test_auc": test_auc,
            "test_ap": test_ap
        })
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc = test_auc
            best_test_ap = test_ap
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss:.4f}, Val_AUC={val_auc:.4f}, Test_AUC={test_auc:.4f}")
    
    print(f"\nFinal: Test_AUC={best_test_auc:.4f}, Test_AP={best_test_ap:.4f}")
    
    # Analyze hop_bias
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            if hasattr(layer.self_attention, 'hop_bias'):
                hop_bias = layer.self_attention.hop_bias.data.cpu().numpy()
                print(f"\nLayer {i} hop_bias: {hop_bias}")
                
                wandb.log({f"hop_bias_layer_{i}": hop_bias.tolist()})
    
    wandb.finish()
    
    return best_test_auc, best_test_ap


if __name__ == "__main__":
    main()