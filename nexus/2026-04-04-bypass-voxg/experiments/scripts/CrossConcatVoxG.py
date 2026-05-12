"""
CrossConcatVoxG: Cross-Attention 增强的 Concat 架构

核心创新：
1. Hop 和 Delta 先做 Cross-Attention 交互
2. 然后 Concat 融合
3. 保留无损信息的同时增强交互

作者: Nexus
日期: 2026-04-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossAttention(nn.Module):
    """
    Cross-Attention: Hop 和 Delta 互相查询
    
    让 Hop 关注 Delta 的关键变化
    让 Delta 关注 Hop 的上下文
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_q, dim)
            key_value: (batch, seq_kv, dim)
        
        Returns:
            output: (batch, seq_q, dim)
        """
        B, N_q, D = query.shape
        N_kv = key_value.shape[1]
        
        # 投影
        Q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 输出
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)
        
        return out


class CrossConcatVoxG(nn.Module):
    """
    Cross-Attention 增强的 Concat 架构
    
    流程：
    1. Hop 和 Delta 分别投影
    2. Cross-Attention 交互
    3. Concat 融合
    4. Transformer 编码
    5. 重构 + 分类
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_hops: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        
        # 输入投影
        self.hop_proj = nn.Linear(input_dim, hidden_dim)
        self.delta_proj = nn.Linear(input_dim, hidden_dim)
        
        # Cross-Attention
        self.hop_to_delta_attn = CrossAttention(hidden_dim, num_heads, dropout)
        self.delta_to_hop_attn = CrossAttention(hidden_dim, num_heads, dropout)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 输入 dropout（去噪）
        self.input_dropout = nn.Dropout(0.2)
    
    def forward(
        self,
        hop_features: torch.Tensor,
        return_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hop_features: (batch, num_hops, input_dim)
        
        Returns:
            logits: (batch,)
            loss: Optional[Tensor]
        """
        batch_size = hop_features.shape[0]
        
        # 去噪
        if self.training:
            hop_input = self.input_dropout(hop_features)
        else:
            hop_input = hop_features
        
        # 计算 Delta
        deltas = torch.zeros_like(hop_input)
        deltas[:, 1:] = hop_input[:, 1:] - hop_input[:, :-1]
        
        # 投影
        hop_emb = self.hop_proj(hop_input)
        delta_emb = self.delta_proj(deltas)
        
        # Cross-Attention
        hop_enhanced = self.hop_to_delta_attn(hop_emb, delta_emb)
        delta_enhanced = self.delta_to_hop_attn(delta_emb, hop_emb)
        
        # 残差连接
        hop_emb = hop_emb + hop_enhanced
        delta_emb = delta_emb + delta_enhanced
        
        # Concat
        concat = torch.cat([hop_emb, delta_emb], dim=-1)
        
        # Fusion
        fused = self.fusion(concat)
        
        # Transformer
        trans_out = self.transformer(fused)
        
        # Pool
        pooled = trans_out.mean(dim=1)
        
        # 分类
        logits = self.classifier(pooled).squeeze(-1)
        
        if return_loss:
            # 重构（重构干净输入）
            recon = self.decoder(pooled)
            target = hop_features[:, 0]  # 重构原始输入
            loss = F.mse_loss(recon, target)
            return logits, loss
        
        return logits, None
    
    def get_model_info(self) -> dict:
        n_params = sum(p.numel() for p in self.parameters())
        return {
            'model': 'CrossConcatVoxG',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'n_params': n_params,
            'n_params_M': n_params / 1e6
        }


def create_cross_concat_voxg(input_dim: int, config: dict = None) -> CrossConcatVoxG:
    default_config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'num_heads': 4,
        'num_hops': 7,
        'dropout': 0.1
    }
    if config:
        default_config.update(config)
    return CrossConcatVoxG(input_dim, **default_config)


if __name__ == "__main__":
    print("Testing CrossConcatVoxG")
    model = create_cross_concat_voxg(745)
    info = model.get_model_info()
    print(f"Model: {info}")
    
    x = torch.randn(4, 7, 745)
    logits, _ = model(x, return_loss=False)
    print(f"Input: {x.shape}")
    print(f"Output: {logits.shape}")
    
    logits, loss = model(x, return_loss=True)
    print(f"Loss: {loss.item():.4f}")
    print("✅ Test passed")