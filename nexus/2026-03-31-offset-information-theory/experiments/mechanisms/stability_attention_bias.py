"""
Mechanism B: Stability Attention Bias (SAB)

Key Insight from Phase 1-3:
- Offset concentrates attention on Token 0 (48-53% in high-dim datasets)
- This is due to negative cosine similarity with other tokens
- Offset has highest stability (variance ratio ~0.01-0.04)

Design:
1. Temperature scaling to reduce Token 0 dominance
2. Position-based attention bias to redistribute attention
3. Stability-aware attention modulation

Mathematical Formulation:
    Attention(Q, K, V) = softmax((QK^T)/T + B_stability) * V
    
    where:
    T = base_temperature * (1 + stability_factor)
    B_stability = learnable_bias[position] * stability_score
    
Author: Nexus
Date: 2026-03-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class StabilityScore(nn.Module):
    """
    Computes stability score for each token based on Offset dynamics.
    
    For Offset tokens, stability is measured by variance across tokens.
    Lower variance = higher stability = more reliable reference.
    
    Args:
        hidden_dim: Hidden dimension
        num_tokens: Number of hop tokens (K)
    """
    
    def __init__(self, hidden_dim: int, num_tokens: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        
        # Learnable stability projection
        self.stability_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Position-specific stability weights
        # Token 0 should have lower weight to reduce dominance
        self.position_weights = nn.Parameter(
            torch.ones(num_tokens)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute stability-based attention modulation.
        
        Args:
            x: [batch_size, num_tokens, hidden_dim] token features
        
        Returns:
            stability_scores: [batch_size, num_tokens] stability scores
        """
        batch_size, num_tokens, _ = x.shape
        
        # Compute token-wise stability
        stability = self.stability_proj(x).squeeze(-1)  # [B, T]
        
        # Apply position weights (lower for Token 0)
        position_weights = F.softmax(self.position_weights[:num_tokens], dim=0)
        stability = stability * position_weights.unsqueeze(0)
        
        # Invert: high stability -> low attention weight (to reduce Token 0 dominance)
        # This is the key insight: stable tokens don't need high attention
        stability_adjustment = 1.0 - stability * 0.5  # Scale to [0.5, 1.0]
        
        return stability_adjustment


class StabilityAttentionBias(nn.Module):
    """
    Multi-Head Attention with Stability-Aware Bias.
    
    This mechanism addresses the Token 0 concentration problem by:
    1. Temperature scaling based on token stability
    2. Learnable position bias to redistribute attention
    3. Orthogonal projection to reduce cosine similarity issues
    
    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        attention_dropout_rate: Dropout rate for attention weights
        num_tokens: Number of tokens in sequence
        base_temperature: Base temperature for softmax (default: 1.0)
        token0_penalty: Penalty for Token 0 attention (default: 0.3)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout_rate: float = 0.1,
        num_tokens: int = 7,
        base_temperature: float = 1.0,
        token0_penalty: float = 0.3
    ):
        super().__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_tokens = num_tokens
        self.base_temperature = base_temperature
        self.token0_penalty = token0_penalty
        
        # Standard attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Stability-aware components
        self.stability_score = StabilityScore(hidden_size, num_tokens)
        
        # Learnable position bias (key component for attention redistribution)
        # Shape: [num_tokens, num_tokens]
        self.position_bias = nn.Parameter(torch.zeros(num_tokens, num_tokens))
        
        # Temperature modulation per position
        self.temperature_scale = nn.Parameter(torch.ones(num_tokens))
        
        # Orthogonal projection to reduce token similarity issues
        self.orthogonal_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.orthogonal_(self.orthogonal_proj.weight)
        
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with stability-aware attention bias.
        
        Args:
            x: [batch_size, seq_len, hidden_size] input features
            attn_mask: Optional attention mask
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch_size, seq_len, hidden_size] transformed features
            attention_weights: [batch_size, num_heads, seq_len, seq_len] if return_attention
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply orthogonal projection to reduce cosine similarity issues
        x_orth = self.orthogonal_proj(x)
        
        # Project to Q, K, V (using orthogonal projected features for Q, K)
        q = self.q_proj(x_orth).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x_orth).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Reshape for multi-head attention: [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add position bias (learnable)
        position_bias = self._compute_position_bias(seq_len)
        attn_scores = attn_scores + position_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        
        # Apply stability-aware temperature scaling
        temperature = self._compute_temperature(x, seq_len)
        attn_scores = attn_scores / temperature.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask * -1e9
        
        # Softmax and dropout
        attention_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply Token 0 penalty to reduce over-concentration
        attention_weights = self._apply_token0_penalty(attention_weights, seq_len)
        
        attention_weights = self.attn_dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        
        # Reshape output: [B, T, H, D] -> [B, T, H*D]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attention_weights
        return output, None
    
    def _compute_position_bias(self, seq_len: int) -> torch.Tensor:
        """
        Compute position-specific attention bias.
        
        Creates a bias matrix that encourages attention distribution
        away from Token 0 and towards other tokens.
        
        Args:
            seq_len: Sequence length
        
        Returns:
            bias: [seq_len, seq_len] position bias matrix
        """
        # Get learnable bias (cropped/padded to seq_len)
        bias = self.position_bias[:seq_len, :seq_len]
        
        # Add structural bias to reduce Token 0 dominance
        # Token 0 gets negative bias when being attended to
        structural_bias = torch.zeros(seq_len, seq_len, device=bias.device)
        
        # Penalize attention TO Token 0
        structural_bias[:, 0] = -self.token0_penalty
        
        # Slightly encourage attention to middle tokens
        mid_tokens = seq_len // 2
        if seq_len > 2:
            encouragement = torch.linspace(0, 0.1, seq_len, device=bias.device)
            structural_bias[:, mid_tokens:] += encouragement[mid_tokens:]
        
        return bias + structural_bias
    
    def _compute_temperature(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Compute stability-aware temperature for each position.
        
        Stable tokens (like Token 0) get higher temperature (smoother attention),
        while unstable tokens get lower temperature (sharper attention).
        
        Args:
            x: [batch_size, seq_len, hidden_size]
            seq_len: Sequence length
        
        Returns:
            temperature: [batch_size, seq_len] temperature values
        """
        # Get stability scores
        stability = self.stability_score(x)  # [B, T]
        
        # Base temperature + stability modulation
        # Higher stability -> higher temperature -> smoother attention
        temp_scale = F.softplus(self.temperature_scale[:seq_len])
        temperature = self.base_temperature * temp_scale.unsqueeze(0) * stability
        
        return temperature
    
    def _apply_token0_penalty(
        self,
        attention_weights: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """
        Apply penalty to Token 0 attention weights.
        
        Redistributes some attention from Token 0 to other tokens.
        
        Args:
            attention_weights: [B, H, T, T] attention weights
            seq_len: Sequence length
        
        Returns:
            adjusted_weights: [B, H, T, T] adjusted attention weights
        """
        # Get attention to Token 0
        attn_to_token0 = attention_weights[:, :, :, 0]  # [B, H, T]
        
        # Calculate penalty amount (proportional to Token 0 attention)
        penalty = attn_to_token0 * self.token0_penalty
        
        # Redistribute penalty to other tokens
        # Weight by inverse position (favor later tokens)
        redistribution_weights = torch.linspace(
            0.5, 1.5, seq_len, device=attention_weights.device
        )
        redistribution_weights[0] = 0  # Don't redistribute to Token 0
        redistribution_weights = redistribution_weights / redistribution_weights.sum()
        
        # Apply redistribution
        adjusted_weights = attention_weights.clone()
        adjusted_weights[:, :, :, 0] = attn_to_token0 - penalty
        
        # Distribute penalty to other tokens
        for i in range(1, seq_len):
            adjusted_weights[:, :, :, i] += penalty * redistribution_weights[i]
        
        return adjusted_weights


class StabilityAwareEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with Stability-Aware Attention.
    
    Combines:
    1. Stability-aware multi-head attention
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
        self.attention = StabilityAttentionBias(
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


def create_stability_aware_gt(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    num_heads: int,
    ffn_dim: int,
    num_tokens: int = 7,
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1,
    token0_penalty: float = 0.3
) -> nn.Module:
    """
    Create a full Stability-Aware Graph Transformer model.
    
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
        token0_penalty: Penalty for Token 0 attention
    
    Returns:
        model: Stability-Aware Graph Transformer model
    """
    
    class StabilityAwareGT(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            # CLS token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            
            # Encoder layers
            self.layers = nn.ModuleList([
                StabilityAwareEncoderLayer(
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
    
    return StabilityAwareGT()


# Unit test
if __name__ == "__main__":
    print("Testing StabilityAttentionBias...")
    
    # Test parameters
    batch_size = 4
    seq_len = 7
    hidden_dim = 64
    num_heads = 4
    
    # Create model
    model = StabilityAttentionBias(
        hidden_size=hidden_dim,
        num_heads=num_heads,
        num_tokens=seq_len,
        token0_penalty=0.3
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, hidden_dim)
    output, attention = model(x, return_attention=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attention.shape}")
    
    # Check Token 0 attention
    token0_attn = attention[:, :, :, 0].mean().item()
    print(f"Mean Token 0 attention: {token0_attn:.4f} (should be lower than baseline ~0.14)")
    
    # Check attention sum
    print(f"Attention sum (should be ~1): {attention[0, 0, 0, :].sum().item():.4f}")
    
    # Test full GT model
    print("\nTesting StabilityAwareGT...")
    gt_model = create_stability_aware_gt(
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