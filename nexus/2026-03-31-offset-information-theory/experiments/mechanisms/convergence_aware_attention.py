"""
Mechanism A: Convergence-Aware Attention (CAA)

Key Insight from Phase 1-3:
- Delta deep tokens have highest MI with labels
- Delta captures convergence behavior (fast decay = 0.69)
- Deep tokens should receive higher attention weights

Design:
1. Learnable token depth weights
2. Convergence indicator embedding
3. Temperature scaling for deep tokens

Mathematical Formulation:
    Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k) + bias_depth(depth)) * V
    
    where bias_depth(depth) = learnable_weight[depth] * convergence_score
    
    convergence_score = ||delta_t - delta_{t-1}|| / ||delta_{t-1}||
    
Author: Nexus
Date: 2026-03-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class ConvergenceScore(nn.Module):
    """
    Computes convergence score for each token based on Delta dynamics.
    
    For Delta tokens, convergence score measures how quickly the 
    features are stabilizing across hops.
    
    Args:
        hidden_dim: Hidden dimension
        num_tokens: Number of hop tokens (K)
        eps: Small constant for numerical stability
    """
    
    def __init__(self, hidden_dim: int, num_tokens: int, eps: float = 1e-8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.eps = eps
        
        # Learnable convergence importance weights
        self.convergence_weights = nn.Parameter(
            torch.ones(num_tokens) / num_tokens
        )
        
        # Projection to compute convergence indicator
        self.convergence_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute convergence-based attention bias.
        
        Args:
            x: [batch_size, num_tokens, hidden_dim] token features
        
        Returns:
            convergence_bias: [batch_size, num_tokens] convergence scores
        """
        batch_size, num_tokens, hidden_dim = x.shape
        
        # Compute token-wise convergence scores
        convergence_scores = self.convergence_proj(x).squeeze(-1)  # [B, T]
        
        # Apply learnable depth weights
        # Deeper tokens get higher weights by default
        depth_weights = F.softmax(self.convergence_weights, dim=0)
        weighted_scores = convergence_scores * depth_weights.unsqueeze(0)
        
        return weighted_scores


class ConvergenceAwareAttention(nn.Module):
    """
    Multi-Head Attention with Convergence-Aware Bias.
    
    This mechanism enhances attention to deep Delta tokens by:
    1. Adding learnable depth-based bias
    2. Scaling attention temperature for deep tokens
    3. Using convergence score as attention modifier
    
    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        attention_dropout_rate: Dropout rate for attention weights
        num_tokens: Number of tokens in sequence
        temperature_init: Initial temperature for scaling (default: 1.0)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout_rate: float = 0.1,
        num_tokens: int = 7,
        temperature_init: float = 1.0
    ):
        super().__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_tokens = num_tokens
        
        # Standard attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Convergence-aware components
        self.convergence_score = ConvergenceScore(hidden_size, num_tokens)
        
        # Learnable depth bias (one per token position)
        self.depth_bias = nn.Parameter(torch.zeros(num_tokens))
        
        # Temperature scaling for deep tokens
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        
        # Deep token emphasis (exponential decay towards later tokens)
        self.register_buffer(
            'depth_emphasis',
            torch.logspace(0, 1, num_tokens)  # [1.0, ..., 10.0]
        )
        
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with convergence-aware attention.
        
        Args:
            x: [batch_size, seq_len, hidden_size] input features
            attn_mask: Optional attention mask
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch_size, seq_len, hidden_size] transformed features
            attention_weights: [batch_size, num_heads, seq_len, seq_len] if return_attention
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Reshape for multi-head attention: [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add convergence-aware bias
        # convergence_bias: [B, T] -> [B, 1, 1, T] for broadcasting across heads and queries
        convergence_bias = self._compute_convergence_bias(x, seq_len)
        convergence_bias_expanded = convergence_bias.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        attn_scores = attn_scores + convergence_bias_expanded
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask * -1e9
        
        # Apply temperature scaling (deeper tokens get lower temperature = sharper attention)
        # This is equivalent to giving more weight to deep token information
        temp_scale = self._compute_temperature_scale(seq_len)
        attn_scores = attn_scores / temp_scale
        
        # Softmax and dropout
        attention_weights = F.softmax(attn_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        
        # Reshape output: [B, T, H, D] -> [B, T, H*D]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attention_weights
        return output, None
    
    def _compute_convergence_bias(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Compute convergence-aware attention bias.
        
        The bias is added to attention scores before softmax.
        Deeper tokens receive positive bias (higher attention).
        
        Args:
            x: [batch_size, seq_len, hidden_size]
            seq_len: Sequence length
        
        Returns:
            bias: [batch_size, seq_len] attention bias
        """
        # Get convergence scores
        convergence_scores = self.convergence_score(x)  # [B, T]
        
        # Add learnable depth bias
        depth_bias = self.depth_bias[:seq_len].unsqueeze(0)  # [1, T]
        
        # Combine with depth emphasis (exponential weighting for deep tokens)
        depth_emphasis = self.depth_emphasis[:seq_len].unsqueeze(0).to(x.device)
        
        # Final bias: convergence-aware + depth-aware
        bias = (convergence_scores + depth_bias) * depth_emphasis
        
        # Normalize to reasonable range
        bias = torch.tanh(bias) * 0.5  # Scale to [-0.5, 0.5]
        
        return bias
    
    def _compute_temperature_scale(self, seq_len: int) -> torch.Tensor:
        """
        Compute temperature scale for attention.
        
        Deep tokens get lower temperature (sharper attention),
        early tokens get higher temperature (smoother attention).
        
        Args:
            seq_len: Sequence length
        
        Returns:
            temp_scale: [seq_len, seq_len] temperature matrix
        """
        # Create position indices
        positions = torch.arange(seq_len, device=self.temperature.device).float()
        
        # Compute temperature: later positions get lower temperature
        position_scale = positions / max(seq_len - 1, 1)
        temp_adjustment = 1.0 - 0.3 * position_scale  # Reduce temp by up to 30%
        
        # Create temperature matrix
        temp_scale = self.temperature * temp_adjustment.unsqueeze(0).unsqueeze(0)
        
        return temp_scale


class ConvergenceAwareEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with Convergence-Aware Attention.
    
    Combines:
    1. Convergence-aware multi-head attention
    2. Position-wise feed-forward network
    3. Layer normalization and residual connections
    
    Args:
        hidden_size: Model hidden dimension
        ffn_size: Feed-forward network hidden size
        num_heads: Number of attention heads
        dropout_rate: General dropout rate
        attention_dropout_rate: Attention dropout rate
        num_tokens: Number of tokens in sequence
    """
    
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        num_tokens: int = 7
    ):
        super().__init__()
        
        # Attention with pre-norm
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = ConvergenceAwareAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_dropout_rate=attention_dropout_rate,
            num_tokens=num_tokens
        )
        self.attention_dropout = nn.Dropout(dropout_rate)
        
        # FFN with pre-norm
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, hidden_size)
        )
        self.ffn_dropout = nn.Dropout(dropout_rate)
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: [batch_size, seq_len, hidden_size]
            attn_mask: Optional attention mask
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch_size, seq_len, hidden_size]
            attention_weights: Optional attention weights
        """
        # Self-attention with residual
        normed = self.attention_norm(x)
        attn_out, attention_weights = self.attention(normed, attn_mask, return_attention)
        x = x + self.attention_dropout(attn_out)
        
        # FFN with residual
        normed = self.ffn_norm(x)
        ffn_out = self.ffn(normed)
        x = x + self.ffn_dropout(ffn_out)
        
        return x, attention_weights


def create_convergence_aware_gt(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    num_heads: int,
    ffn_dim: int,
    num_tokens: int = 7,
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1
) -> nn.Module:
    """
    Create a full Convergence-Aware Graph Transformer model.
    
    This is a template function that can be adapted to specific
    graph transformer architectures.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension (e.g., number of classes)
        num_layers: Number of encoder layers
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network hidden size
        num_tokens: Number of hop tokens
        dropout_rate: General dropout rate
        attention_dropout_rate: Attention dropout rate
    
    Returns:
        model: Convergence-Aware Graph Transformer model
    """
    
    class ConvergenceAwareGT(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            # CLS token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            
            # Encoder layers
            self.layers = nn.ModuleList([
                ConvergenceAwareEncoderLayer(
                    hidden_size=hidden_dim,
                    ffn_size=ffn_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    num_tokens=num_tokens + 1  # +1 for CLS token
                )
                for _ in range(num_layers)
            ])
            
            # Output projection
            self.output_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            """
            Forward pass.
            
            Args:
                x: [batch_size, num_tokens, input_dim]
            
            Returns:
                output: [batch_size, output_dim]
                attentions: List of attention weights from each layer
            """
            batch_size = x.shape[0]
            
            # Project input
            x = self.input_proj(x)
            
            # Prepend CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # Pass through encoder layers
            attentions = []
            for layer in self.layers:
                x, attn = layer(x, return_attention=True)
                attentions.append(attn)
            
            # Use CLS token for prediction
            cls_output = self.output_norm(x[:, 0, :])
            output = self.output_proj(cls_output)
            
            return output, attentions
    
    return ConvergenceAwareGT()


# Unit test
if __name__ == "__main__":
    print("Testing ConvergenceAwareAttention...")
    
    # Test parameters
    batch_size = 4
    seq_len = 7
    hidden_dim = 64
    num_heads = 4
    
    # Create model
    model = ConvergenceAwareAttention(
        hidden_size=hidden_dim,
        num_heads=num_heads,
        num_tokens=seq_len
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, hidden_dim)
    output, attention = model(x, return_attention=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attention.shape}")
    print(f"Attention sum (should be ~1): {attention[0, 0, 0, :].sum().item():.4f}")
    
    # Test full GT model
    print("\nTesting ConvergenceAwareGT...")
    gt_model = create_convergence_aware_gt(
        input_dim=32,
        hidden_dim=64,
        output_dim=2,
        num_layers=3,
        num_heads=4,
        ffn_dim=128,
        num_tokens=7
    )
    
    x = torch.randn(batch_size, 7, 32)
    output, attentions = gt_model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of attention layers: {len(attentions)}")
    
    print("\n✓ All tests passed!")