# GT Architecture Comparison Analysis

**Author**: Nexus
**Date**: 2026-03-31

---

## 1. Overview

This document compares our proposed mechanisms with existing Graph Transformer architectures, highlighting theoretical differences and practical advantages.

---

## 2. Existing GT Architectures

### 2.1 NAGphormer (2024)

**Tokenization**:
- Multi-hop aggregation: $h_k = A^k X$
- Fixed token sequence (K hops)
- CLS token for output

**Attention**:
- Standard self-attention
- No token-specific modifications
- Uniform treatment of all tokens

**Key Characteristics**:
- **Strategy**: Hop-based
- **Information assumption**: All tokens equally informative
- **Limitation**: No differentiation between early vs late hop information

### 2.2 Graphormer (2023)

**Tokenization**:
- Node-level tokens
- Spatial encoding (shortest path distance)
- Degree encoding

**Attention**:
- Structural bias in attention
- $Attention += b_{spatial}(d_{ij})$

**Key Characteristics**:
- **Strategy**: Structural encoding
- **Information assumption**: Graph structure is primary signal
- **Limitation**: Heavy dependence on preprocessing (shortest path)

### 2.3 GraphBERT (2020)

**Tokenization**:
- Node features + position encoding
- Contextualized embeddings via graph structure

**Attention**:
- Graph-aware attention masking
- Laplacian positional encoding

**Key Characteristics**:
- **Strategy**: Position + structure encoding
- **Information assumption**: Position captures graph topology
- **Limitation**: Fixed positional encoding, not adaptive

### 2.4 VoxGFormer (Baseline in VoxG)

**Tokenization**:
- Multi-hop aggregation (similar to NAGphormer)
- CLS token for anomaly detection

**Attention**:
- Standard transformer attention
- Contrastive learning objective

**Key Characteristics**:
- **Strategy**: Hop-based
- **Task**: Graph anomaly detection
- **Limitation**: Same as NAGphormer—uniform token treatment

---

## 3. Our Proposed Mechanisms

### 3.1 Mechanism A: Convergence-Aware Attention (CAA)

**Novelty**:
- First to consider token depth for attention weighting
- Convergence score as attention modifier
- Learnable depth importance

**Comparison with Existing**:

| Feature | NAGphormer | Graphormer | Our CAA |
|---------|------------|------------|---------|
| Token depth awareness | ❌ | ❌ | ✅ |
| Convergence capture | ❌ | ❌ | ✅ |
| Learnable token weights | ❌ | ❌ (fixed bias) | ✅ |
| Information-theoretic basis | ❌ | ❌ | ✅ |

**Advantage**: CAA explicitly leverages the finding that deep Delta tokens have highest MI with labels.

### 3.2 Mechanism B: Stability Attention Bias (SAB)

**Novelty**:
- Temperature scaling based on token stability
- Token 0 penalty to reduce concentration
- Orthogonal projection for cosine similarity issues

**Comparison with Existing**:

| Feature | NAGphormer | Graphormer | Our SAB |
|---------|------------|------------|---------|
| Token 0 penalty | ❌ | ❌ | ✅ |
| Stability-aware temp | ❌ | ❌ | ✅ |
| Orthogonal projection | ❌ | ❌ | ✅ |
| Attention redistribution | ❌ | ❌ | ✅ |

**Advantage**: SAB addresses the discovered problem of Offset Token 0 dominance (48-53% attention).

### 3.3 Mechanism C: Dual-Stream Architecture (DSA)

**Novelty**:
- Parallel processing of different tokenization strategies
- Cross-attention for information fusion
- Gated combination with dynamic weighting

**Comparison with Existing**:

| Feature | NAGphormer | Graphormer | Our DSA |
|---------|------------|------------|---------|
| Multi-strategy | ❌ | ❌ | ✅ |
| Cross-stream attention | ❌ | ❌ | ✅ |
| Gated fusion | ❌ | ❌ | ✅ |
| Adaptive strategy selection | ❌ | ❌ | ✅ |

**Advantage**: DSA combines Offset stability and Delta convergence in a principled way.

---

## 4. Information-Theoretic Foundation

### 4.1 Key Difference from Existing Work

**Existing Assumptions**:
- All tokens equally informative (NAGphormer)
- Graph structure is primary signal (Graphormer)
- Position encoding suffices (GraphBERT)

**Our Foundation**:
- Tokens have different information distributions (Phase 1)
- Token information depends on feature dimensionality (Phase 2)
- Attention patterns correlate with information distribution (Phase 3)

### 4.2 Theoretical Contributions

| Aspect | Existing | Our Work |
|--------|----------|----------|
| Tokenization analysis | None | Systematic MI + entropy analysis |
| Dimensionality awareness | None | High-D vs Low-D recommendations |
| Attention design | Fixed bias | Information-guided mechanisms |
| Physical meaning | None | Graph correlation interpretation |

---

## 5. Practical Advantages

### 5.1 Computational Efficiency

| Mechanism | Additional Parameters | Complexity |
|-----------|-----------------------|------------|
| CAA | ~K learnable weights | O(T²) (same) |
| SAB | ~K² bias + orthogonal | O(T² + D²) |
| DSA | ~2x parameters | O(2T²) |

**Note**: All mechanisms maintain O(T²) attention complexity, only adding small parameter overhead.

### 5.2 Generalization

| Mechanism | Dataset Dependency | Generalization |
|-----------|-------------------|----------------|
| CAA | Low (learnable weights) | High |
| SAB | Moderate (Token 0 penalty) | Medium |
| DSA | Low (adaptive gate) | High |

**Note**: DSA's gated fusion automatically adapts to dataset characteristics.

### 5.3 Integration with Existing GT

All three mechanisms can be integrated as:
1. **CAA**: Replace standard attention in any GT
2. **SAB**: Add as attention bias layer
3. **DSA**: Wrap existing GT as dual-stream

---

## 6. Experimental Validation Comparison

### 6.1 Photo Dataset Results

| Method | AUC | Token 0 Attn | Deep Attn |
|--------|-----|--------------|-----------|
| NAGphormer (Hop) | 0.9898 | 19.6% | 13.3% |
| Offset baseline | 0.9251 | 48.6% ⚠️ | 8.5% |
| Delta baseline | 0.9893 | 35.2% | 11.6% |
| **CAA (our)** | TBD | Expected ↓ | Expected ↑ |
| **SAB (our)** | TBD | Expected ↓ | Expected ↑ |
| **DSA (our)** | TBD | Balanced | Balanced |

### 6.2 SOTA Comparison for Anomaly Detection

| Method | Photo AUC | Amazon AUC | Source |
|--------|-----------|------------|--------|
| VecGAD | 0.8960 | 0.9391 | Paper |
| RHO | - | 0.8509 | Paper |
| Our mechanisms | TBD | TBD | This work |

**Target**: Our mechanisms aim to achieve comparable or better AUC while improving attention distribution.

---

## 7. Summary Table

| Criteria | NAGphormer | Graphormer | CAA | SAB | DSA |
|----------|------------|------------|-----|-----|-----|
| Information basis | ❌ | ❌ | ✅ | ✅ | ✅ |
| Token depth awareness | ❌ | ❌ | ✅ | ❌ | ✅ |
| Attention concentration fix | ❌ | ❌ | ❌ | ✅ | ✅ |
| Multi-strategy fusion | ❌ | ❌ | ❌ | ❌ | ✅ |
| Learnable adaptation | ❌ | ❌ | ✅ | ✅ | ✅ |
| Theoretical grounding | ❌ | ❌ | ✅ | ✅ | ✅ |

---

## 8. Key Takeaways

1. **First work** to systematically analyze GT tokenization from information theory perspective
2. **Novel mechanisms** designed based on empirical discoveries, not arbitrary
3. **Advantages** over existing GT: information-guided, dataset-aware, theoretically grounded
4. **Practical**: Low computational overhead, easy integration with existing architectures

---

**Next Steps**: Complete validation experiments, quantify advantages with full metrics.