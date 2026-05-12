"""
Hop-Aware VoxGFormer

在 VoxGFormer 的 Transformer 中集成 Hop-Aware Attention

核心修改：
1. MultiHeadAttention 添加 hop_bias 支持
2. TransformerEncoder 构建 hop_bias_matrix 并传入
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HopAwareMultiHeadAttention(nn.Module):
    """
    Hop-Aware Multi-Head Attention
    
    在标准注意力中注入 hop_bias（相对距离偏置）
    """
    
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
        
        # Hop 相对距离偏置（可学习）
        # B[k] 表示 hop 差值为 k 时的偏置
        self.hop_bias = nn.Parameter(torch.zeros(2 * max_hop - 1))
        
    def forward(self, q, k, v, attn_bias=None, hop_indices=None):
        """
        Args:
            q, k, v: [batch_size, seq_len, hidden_size]
            attn_bias: 可选的外部偏置
            hop_indices: [seq_len] 每个 token 的 hop 索引（0, 1, 2, ..., pp_k）
        """
        orig_q_size = q.size()
        batch_size = q.size(0)
        seq_len = q.size(1)
        
        d_k = self.att_size
        d_v = self.att_size
        
        # Linear projections
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        
        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
        
        # Scaled Dot-Product Attention
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        
        # 外部偏置（如果有）
        if attn_bias is not None:
            x = x + attn_bias
        
        # Hop-Aware Bias
        if hop_indices is not None:
            hop_bias_matrix = self._build_hop_bias_matrix(hop_indices, seq_len)
            x = x + hop_bias_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        attention_weights = torch.softmax(x, dim=3)
        x = self.att_dropout(attention_weights)
        
        x = x.matmul(v)  # [b, h, q_len, attn]
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)
        
        x = self.output_layer(x)
        
        assert x.size() == orig_q_size
        return x, attention_weights
    
    def _build_hop_bias_matrix(self, hop_indices, seq_len):
        """
        构建 hop_bias 矩阵
        
        hop_indices: [seq_len] - 每个 token 的 hop 索引
        返回: [seq_len, seq_len] - hop_bias_matrix[i,j] = B[hop[j] - hop[i]]
        """
        hop_i = hop_indices.unsqueeze(1)  # [seq_len, 1]
        hop_j = hop_indices.unsqueeze(0)  # [1, seq_len]
        hop_diff = hop_j - hop_i  # [seq_len, seq_len]
        
        # 映射到索引
        bias_idx = hop_diff + (self.max_hop - 1)
        bias_idx = bias_idx.clamp(0, 2 * self.max_hop - 2)
        
        return self.hop_bias[bias_idx]


class HopAwareEncoderLayer(nn.Module):
    """
    Hop-Aware Encoder Layer
    
    使用 HopAwareMultiHeadAttention
    """
    
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, max_hop=7):
        super().__init__()
        
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = HopAwareMultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads, max_hop)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, attn_bias=None, hop_indices=None):
        y = self.self_attention_norm(x)
        y, attention_weights = self.self_attention(y, y, y, attn_bias, hop_indices)
        y = self.self_attention_dropout(y)
        x = x + y
        
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        
        return x, attention_weights


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


# ===== 使用说明 =====
#
# 1. 在 GGADFormer.__init__ 中替换 EncoderLayer 为 HopAwareEncoderLayer
# 2. 在 TransformerEncoder 中构建 hop_indices 并传入
# 3. hop_indices = torch.arange(args.pp_k + 1)  # [0, 1, 2, ..., pp_k]
#
# 示例修改：
#
# class GGADFormer(nn.Module):
#     def __init__(self, ...):
#         # 替换 EncoderLayer
#         encoders = [HopAwareEncoderLayer(...) for _ in range(args.GT_num_layers)]
#         
#     def TransformerEncoder(self, tokens):
#         hop_indices = torch.arange(tokens.size(1), device=tokens.device)
#         for i, l in enumerate(self.layers):
#             emb, attn = self.layers[i](emb, hop_indices=hop_indices)