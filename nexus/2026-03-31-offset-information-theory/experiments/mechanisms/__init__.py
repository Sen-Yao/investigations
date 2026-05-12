"""
GT Injection Mechanisms for VoxGFormer

Based on Phase 1-3 discoveries:
- Offset concentrates attention on Token 0 (48-53% in high-dim)
- Delta distributes attention across deep tokens
- Mixed strategies (Offset+Delta) achieve best deep attention

Three novel mechanisms:
1. Convergence-Aware Attention (for Delta)
2. Stability Attention Bias (for Offset)
3. Dual-Stream Architecture (Offset + Delta fusion)

Author: Nexus
Date: 2026-03-31
"""

from .convergence_aware_attention import (
    ConvergenceAwareAttention,
    ConvergenceAwareEncoderLayer,
    create_convergence_aware_gt
)

from .stability_attention_bias import (
    StabilityAttentionBias,
    StabilityAwareEncoderLayer,
    create_stability_aware_gt
)

from .dual_stream_architecture import (
    DualStreamGT,
    CrossStreamAttention,
    create_dual_stream_gt
)

__all__ = [
    # Mechanism A: Convergence-Aware
    "ConvergenceAwareAttention",
    "ConvergenceAwareEncoderLayer",
    "create_convergence_aware_gt",
    
    # Mechanism B: Stability-Aware
    "StabilityAttentionBias",
    "StabilityAwareEncoderLayer",
    "create_stability_aware_gt",
    
    # Mechanism C: Dual-Stream
    "DualStreamGT",
    "CrossStreamAttention",
    "create_dual_stream_gt",
]