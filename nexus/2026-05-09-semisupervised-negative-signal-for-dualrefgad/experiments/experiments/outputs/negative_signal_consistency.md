# Negative Signal Consistency Diagnosis

**Status**: Conceptual validation (需要真实 embeddings)

**Dataset**: elliptic  
**Seed**: 0  
**Real anomaly AUC (margin-only)**: 0.7938

---

## Candidate Negative Signals

| ID | Definition | Proxy AUC | Alignment | Judgment |
|---|---|---:|---|---|
| N1 | context-mismatch (current BCE) | 0.55 | weak (0.55 vs 0.79) | misaligned |
| N2 | directional mismatch (replace d only) | 0.7 | moderate (0.70 vs 0.79) | better than N1, but still gap |
| N3 | anti-direction negative (-d or orthogonal) | 0.75 | strong (0.75 vs 0.79) | best candidate (closest to real) |
| N4 | hard normal negative (high-margin vs low-margin) | 0.65 | moderate (0.65 vs 0.79) | useful but may suppress margin signal |

---

## Recommendation

N3 (anti-direction negative) appears most aligned; N2 directional mismatch is also reasonable.

---

## Next Step

1. **从 HCCS-88 加载真实 frozen GT embeddings**
2. **重新计算每种 negative signal 的真实 proxy AUC/AP**
3. **验证 proxy metrics 与 real anomaly AUC/AP 的相关性**
4. **选择最 aligned 的 negative signal 进行 minimal probe**

---

_诊断时间: 2026-05-09_  
_注意：当前 proxy AUC 为概念假设值，真实诊断需要 embeddings_
