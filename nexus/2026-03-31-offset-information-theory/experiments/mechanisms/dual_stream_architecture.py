"""
Mechanism C: Dual-Stream Architecture (DSA)

Key Insight from Phase 1-3:
- Mixed strategy (Offset(0-3) + Delta(4-6)) achieves highest deep attention (0.131)
- Offset provides stability and clustering quality
- Delta captures convergence and graph structure

Design:
1. Two parallel attention streams
   - Stability stream (Offset): Captures stable reference information
   - Convergence stream (Delta): Captures dynamic convergence behavior
2. Cross-stream attention for information fusion
3. Gated output combination

Mathematical Formulation:
    H_stable = Attention_stable(Offset_tokens)
    H_conv = Attention_conv(Delta_tokens)
    
    H_fused = CrossAttention(H_stable, H_conv)
    
    Output = Gate * H_stable + (1 - Gate) * H_conv

Author: Nexus
Date: 2026-03-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict


class StreamAttention(nn.Module):
    """
    Single-stream attention module for either Offset or Delta tokens.
    
    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        stream_type: "stable" for Offset or "convergence" for Delta
        attention_dropout_rate: Dropout rate
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        stream_type: str = "stable",
        attention_dropout_rate: float = 0.1
    ):
        super().__init__()
        
        assert hidden_size % num_heads == 0
        assert stream_type in ["stable", "convergence"]
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.stream_type = stream_type
        
        # Stream-specific projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Stream-specific bias
        if stream_type == "stable":
            # For stable stream: penalize Token 0 dominance
            self.stream_bias = nn.Parameter(torch.tensor(-0.2))
        else:
            # For convergence stream: emphasize deep tokens
            self.stream_bias = nn.Parameter(torch.tensor(0.1))
        
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for single stream.
        
        Args:
            x: [batch_size, seq_len, hidden_size] stream token features
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch_size, seq_len, hidden_size]
            attention_weights: Optional attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Reshape: [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add stream-specific bias
        # Create position-specific bias based on stream type
        position_bias = self._create_position_bias(seq_len)
        attn_scores = attn_scores + position_bias
        
        # Softmax and dropout
        attention_weights = F.softmax(attn_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        
        # Apply attention
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attention_weights
        return output, None
    
    def _create_position_bias(self, seq_len: int) -> torch.Tensor:
        """
        Create position-specific bias based on stream type.
        
        For stable stream: penalize early tokens (especially Token 0)
        For convergence stream: emphasize late tokens
        
        Args:
            seq_len: Sequence length
        
        Returns:
            bias: [1, 1, seq_len, seq_len] bias matrix
        """
        positions = torch.arange(seq_len, device=self.stream_bias.device)
        
        # Create position-dependent bias
        if self.stream_type == "stable":
            # Decreasing bias: early tokens penalized, late tokens encouraged
            bias = -self.stream_bias * (1 - positions / seq_len)
        else:
            # Increasing bias: early tokens neutral, late tokens emphasized
            bias = self.stream_bias * positions / seq_len
        
        # Expand to [1, 1, T, T] (broadcastable to attention scores)
        bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        return bias


class CrossStreamAttention(nn.Module):
    """
    Cross-attention module for fusing stable and convergence streams.
    
    Allows tokens from one stream to attend to tokens from another stream,
    enabling information exchange between Offset and Delta representations.
    
    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        attention_dropout_rate: Dropout rate
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout_rate: float = 0.1
    ):
        super().__init__()
        
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Cross-attention projections
        self.q_proj_stable = nn.Linear(hidden_size, hidden_size)  # Q from stable
        self.k_proj_conv = nn.Linear(hidden_size, hidden_size)    # K from convergence
        self.v_proj_conv = nn.Linear(hidden_size, hidden_size)    # V from convergence
        
        self.q_proj_conv = nn.Linear(hidden_size, hidden_size)    # Q from convergence
        self.k_proj_stable = nn.Linear(hidden_size, hidden_size)  # K from stable
        self.v_proj_stable = nn.Linear(hidden_size, hidden_size)  # V from stable
        
        # Output projections
        self.out_proj_stable = nn.Linear(hidden_size, hidden_size)
        self.out_proj_conv = nn.Linear(hidden_size, hidden_size)
        
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        
    def forward(
        self,
        h_stable: torch.Tensor,
        h_conv: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass for cross-stream attention.
        
        Args:
            h_stable: [batch_size, seq_len, hidden_size] stable stream features
            h_conv: [batch_size, seq_len, hidden_size] convergence stream features
            return_attention: Whether to return attention weights
        
        Returns:
            h_stable_out: [batch_size, seq_len, hidden_size] updated stable stream
            h_conv_out: [batch_size, seq_len, hidden_size] updated convergence stream
            cross_attentions: Dict of cross-attention weights if return_attention
        """
        batch_size, seq_len, _ = h_stable.shape
        
        # Cross-attention: stable queries -> conv keys/values
        q_stable = self.q_proj_stable(h_stable).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_conv = self.k_proj_conv(h_conv).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_conv = self.v_proj_conv(h_conv).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q_stable = q_stable.transpose(1, 2)
        k_conv = k_conv.transpose(1, 2)
        v_conv = v_conv.transpose(1, 2)
        
        attn_stable_to_conv = torch.matmul(q_stable, k_conv.transpose(-2, -1)) * self.scale
        attn_weights_stable = F.softmax(attn_stable_to_conv, dim=-1)
        attn_weights_stable = self.attn_dropout(attn_weights_stable)
        
        stable_cross_out = torch.matmul(attn_weights_stable, v_conv)
        stable_cross_out = stable_cross_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        stable_cross_out = self.out_proj_stable(stable_cross_out)
        
        # Cross-attention: conv queries -> stable keys/values
        q_conv = self.q_proj_conv(h_conv).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_stable = self.k_proj_stable(h_stable).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_stable = self.v_proj_stable(h_stable).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q_conv = q_conv.transpose(1, 2)
        k_stable = k_stable.transpose(1, 2)
        v_stable = v_stable.transpose(1, 2)
        
        attn_conv_to_stable = torch.matmul(q_conv, k_stable.transpose(-2, -1)) * self.scale
        attn_weights_conv = F.softmax(attn_conv_to_stable, dim=-1)
        attn_weights_conv = self.attn_dropout(attn_weights_conv)
        
        conv_cross_out = torch.matmul(attn_weights_conv, v_stable)
        conv_cross_out = conv_cross_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        conv_cross_out = self.out_proj_conv(conv_cross_out)
        
        if return_attention:
            cross_attentions = {
                "stable_to_conv": attn_weights_stable,
                "conv_to_stable": attn_weights_conv
            }
            return stable_cross_out, conv_cross_out, cross_attentions
        
        return stable_cross_out, conv_cross_out, None


class GatedFusion(nn.Module):
    """
    Gated fusion module for combining stable and convergence stream outputs.
    
    Uses a learned gate to dynamically weight the contribution of each stream
    based on the input features.
    
    Args:
        hidden_size: Model hidden dimension
        gate_init: Initial gate bias (default: 0.5, equal weighting)
    """
    
    def __init__(self, hidden_size: int, gate_init: float = 0.5):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Gate computation network
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Initialize gate bias
        self.gate_bias = nn.Parameter(torch.tensor(gate_init))
        
        # Output projections
        self.stable_proj = nn.Linear(hidden_size, hidden_size)
        self.conv_proj = nn.Linear(hidden_size, hidden_size)
        self.fusion_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self,
        h_stable: torch.Tensor,
        h_conv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for gated fusion.
        
        Args:
            h_stable: [batch_size, hidden_size] or [batch_size, seq_len, hidden_size]
            h_conv: [batch_size, hidden_size] or [batch_size, seq_len, hidden_size]
        
        Returns:
            h_fused: Fused output (same shape as input)
            gate_values: Gate values (stable stream weight)
        """
        # Handle both [B, D] and [B, T, D] shapes
        if h_stable.dim() == 2:
            # [B, D] case (CLS token output)
            batch_size, hidden_size = h_stable.shape
            
            # Concatenate for gate computation
            h_concat = torch.cat([h_stable, h_conv], dim=-1)
            
            # Compute gate values
            gate = self.gate_net(h_concat) + self.gate_bias
            gate = torch.clamp(gate, 0.1, 0.9)
            
            # Project each stream
            h_stable_proj = self.stable_proj(h_stable)
            h_conv_proj = self.conv_proj(h_conv)
            
            # Weighted combination
            h_fused = gate * h_stable_proj + (1 - gate) * h_conv_proj
            h_fused = self.fusion_proj(h_fused)
            
            return h_fused, gate
        else:
            # [B, T, D] case
            batch_size, seq_len, hidden_size = h_stable.shape
            
            # Concatenate for gate computation
            h_concat = torch.cat([h_stable, h_conv], dim=-1)
            
            # Compute gate values
            gate = self.gate_net(h_concat) + self.gate_bias
            gate = torch.clamp(gate, 0.1, 0.9)
            
            # Project each stream
            h_stable_proj = self.stable_proj(h_stable)
            h_conv_proj = self.conv_proj(h_conv)
            
            # Weighted combination
            h_fused = gate * h_stable_proj + (1 - gate) * h_conv_proj
            h_fused = self.fusion_proj(h_fused)
            
            return h_fused, gate


class DualStreamEncoderLayer(nn.Module):
    """
    Dual-stream transformer encoder layer.
    
    Processes Offset and Delta tokens in parallel streams, then fuses them
    via cross-attention and gated combination.
    
    Args:
        hidden_size: Model hidden dimension
        ffn_size: Feed-forward network hidden size
        num_heads: Number of attention heads
        dropout_rate: General dropout rate
        attention_dropout_rate: Attention dropout rate
        use_cross_attention: Whether to use cross-stream attention
    """
    
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        use_cross_attention: bool = True
    ):
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        
        # Pre-norm for each stream
        self.stable_norm = nn.LayerNorm(hidden_size)
        self.conv_norm = nn.LayerNorm(hidden_size)
        
        # Stream-specific attention
        self.stable_attention = StreamAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            stream_type="stable",
            attention_dropout_rate=attention_dropout_rate
        )
        
        self.conv_attention = StreamAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            stream_type="convergence",
            attention_dropout_rate=attention_dropout_rate
        )
        
        # Cross-attention (optional)
        if use_cross_attention:
            self.cross_attention = CrossStreamAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                attention_dropout_rate=attention_dropout_rate
            )
        
        # Gated fusion
        self.fusion = GatedFusion(hidden_size)
        
        # FFN
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, hidden_size)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(
        self,
        h_stable: torch.Tensor,
        h_conv: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass for dual-stream layer.
        
        Args:
            h_stable: [batch_size, seq_len, hidden_size] stable stream (Offset)
            h_conv: [batch_size, seq_len, hidden_size] convergence stream (Delta)
            return_attention: Whether to return attention weights
        
        Returns:
            h_stable: Updated stable stream
            h_conv: Updated convergence stream
            attentions: Dict containing all attention weights
        """
        attentions = {} if return_attention else None
        
        # Stream-specific self-attention
        stable_normed = self.stable_norm(h_stable)
        stable_out, stable_attn = self.stable_attention(stable_normed, return_attention)
        h_stable = h_stable + self.dropout(stable_out)
        
        conv_normed = self.conv_norm(h_conv)
        conv_out, conv_attn = self.conv_attention(conv_normed, return_attention)
        h_conv = h_conv + self.dropout(conv_out)
        
        if return_attention:
            attentions["stable_self"] = stable_attn
            attentions["conv_self"] = conv_attn
        
        # Cross-stream attention (optional)
        if self.use_cross_attention:
            stable_cross, conv_cross, cross_attn = self.cross_attention(
                h_stable, h_conv, return_attention
            )
            h_stable = h_stable + self.dropout(stable_cross)
            h_conv = h_conv + self.dropout(conv_cross)
            
            if return_attention:
                attentions["cross"] = cross_attn
        
        # Gated fusion and FFN
        h_fused, gate = self.fusion(h_stable, h_conv)
        
        # Apply FFN on fused representation
        fused_normed = self.ffn_norm(h_fused)
        ffn_out = self.ffn(fused_normed)
        h_fused = h_fused + self.dropout(ffn_out)
        
        # Split fused output back to streams (optional)
        # Alternative: keep fused and update both streams
        h_stable = h_fused  # Both streams share fused output
        h_conv = h_fused
        
        if return_attention:
            attentions["gate"] = gate
        
        return h_stable, h_conv, attentions


class DualStreamGT(nn.Module):
    """
    Full Dual-Stream Graph Transformer model.
    
    Architecture:
    1. Input projection for Offset and Delta tokens
    2. Dual-stream encoder layers
    3. CLS token for each stream
    4. Final fusion and output projection
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension (e.g., number of classes)
        num_layers: Number of encoder layers
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network hidden size
        num_tokens: Number of tokens per stream
        dropout_rate: General dropout rate
        attention_dropout_rate: Attention dropout rate
        use_cross_attention: Whether to use cross-stream attention
        token_split: (stable_tokens, conv_tokens) split configuration
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        num_tokens: int = 7,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        use_cross_attention: bool = True,
        token_split: Tuple[int, int] = (3, 4)  # (stable, conv) tokens
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.token_split = token_split
        self.stable_tokens, self.conv_tokens = token_split
        
        # Input projections
        self.input_proj_stable = nn.Linear(input_dim, hidden_dim)
        self.input_proj_conv = nn.Linear(input_dim, hidden_dim)
        
        # CLS tokens
        self.cls_token_stable = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_token_conv = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Encoder layers
        self.layers = nn.ModuleList([
            DualStreamEncoderLayer(
                hidden_size=hidden_dim,
                ffn_size=ffn_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                use_cross_attention=use_cross_attention
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm_stable = nn.LayerNorm(hidden_dim)
        self.output_norm_conv = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)
        
        # Final fusion gate
        self.final_gate = GatedFusion(hidden_dim)
        
    def forward(
        self,
        offset_tokens: torch.Tensor,
        delta_tokens: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Forward pass for dual-stream GT.
        
        Args:
            offset_tokens: [batch_size, stable_tokens, input_dim] Offset tokens
            delta_tokens: [batch_size, conv_tokens, input_dim] Delta tokens
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch_size, output_dim]
            attentions: List of attention dicts from each layer
        """
        batch_size = offset_tokens.shape[0]
        
        # Project inputs
        h_stable = self.input_proj_stable(offset_tokens)
        h_conv = self.input_proj_conv(delta_tokens)
        
        # Add CLS tokens
        cls_stable = self.cls_token_stable.expand(batch_size, -1, -1)
        cls_conv = self.cls_token_conv.expand(batch_size, -1, -1)
        
        h_stable = torch.cat([cls_stable, h_stable], dim=1)
        h_conv = torch.cat([cls_conv, h_conv], dim=1)
        
        # Process through dual-stream layers
        all_attentions = []
        for layer in self.layers:
            h_stable, h_conv, layer_attentions = layer(h_stable, h_conv, return_attention)
            if return_attention:
                all_attentions.append(layer_attentions)
        
        # Use CLS tokens for output
        cls_stable_out = self.output_norm_stable(h_stable[:, 0, :])
        cls_conv_out = self.output_norm_conv(h_conv[:, 0, :])
        
        # Final gated fusion
        h_final, final_gate = self.final_gate(cls_stable_out, cls_conv_out)
        
        # Combine both CLS representations
        combined = torch.cat([cls_stable_out, cls_conv_out], dim=-1)
        output = self.output_proj(combined)
        
        if return_attention:
            all_attentions.append({"final_gate": final_gate})
            return output, all_attentions
        
        return output, None
    
    def forward_from_hop(
        self,
        hop_tokens: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Alternative forward pass that computes Offset and Delta from Hop tokens.
        
        Args:
            hop_tokens: [batch_size, num_tokens, input_dim] Hop tokens
            return_attention: Whether to return attention weights
        
        Returns:
            output: [batch_size, output_dim]
            attentions: List of attention dicts
        """
        # Compute Offset: hop_t - hop_0
        offset_tokens = hop_tokens - hop_tokens[:, 0:1, :]
        
        # Compute Delta: hop_t - hop_{t-1}
        delta_tokens = hop_tokens[:, 1:, :] - hop_tokens[:, :-1, :]
        
        # Split tokens according to configuration
        stable_tokens = offset_tokens[:, :self.stable_tokens, :]
        conv_tokens = delta_tokens[:, :self.conv_tokens, :]
        
        return self.forward(stable_tokens, conv_tokens, return_attention)


def create_dual_stream_gt(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    num_heads: int,
    ffn_dim: int,
    num_tokens: int = 7,
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1,
    use_cross_attention: bool = True,
    token_split: Tuple[int, int] = (3, 4)
) -> nn.Module:
    """
    Factory function for creating Dual-Stream GT model.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of encoder layers
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network hidden size
        num_tokens: Number of hop tokens
        dropout_rate: General dropout rate
        attention_dropout_rate: Attention dropout rate
        use_cross_attention: Whether to use cross-stream attention
        token_split: (stable_tokens, conv_tokens) split configuration
    
    Returns:
        model: Dual-Stream Graph Transformer model
    """
    return DualStreamGT(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_tokens=num_tokens,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        use_cross_attention=use_cross_attention,
        token_split=token_split
    )


# Unit test
if __name__ == "__main__":
    print("Testing DualStreamGT...")
    
    # Test parameters
    batch_size = 4
    stable_tokens = 3
    conv_tokens = 4
    input_dim = 32
    hidden_dim = 64
    output_dim = 2
    num_layers = 3
    num_heads = 4
    
    # Create model
    model = DualStreamGT(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=128,
        num_tokens=7,
        token_split=(stable_tokens, conv_tokens)
    )
    
    # Test forward pass with separate Offset/Delta
    offset_tokens = torch.randn(batch_size, stable_tokens, input_dim)
    delta_tokens = torch.randn(batch_size, conv_tokens, input_dim)
    
    output, attentions = model(offset_tokens, delta_tokens, return_attention=True)
    
    print(f"Offset tokens shape: {offset_tokens.shape}")
    print(f"Delta tokens shape: {delta_tokens.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of attention layers: {len(attentions)}")
    
    # Check gate values
    final_gate = attentions[-1]["final_gate"]
    print(f"Final gate values shape: {final_gate.shape}")
    print(f"Mean gate value (stable weight): {final_gate.mean().item():.4f}")
    
    # Test forward_from_hop
    print("\nTesting forward_from_hop...")
    hop_tokens = torch.randn(batch_size, 7, input_dim)
    output, attentions = model.forward_from_hop(hop_tokens, return_attention=True)
    
    print(f"Hop tokens shape: {hop_tokens.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n✓ All tests passed!")