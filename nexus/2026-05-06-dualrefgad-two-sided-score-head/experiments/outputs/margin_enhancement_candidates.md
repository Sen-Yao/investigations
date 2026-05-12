# Margin Enhancement Candidates Discussion

**Date**: 2026-05-07 17:26
**Context**: After dual_margin_two_score breakthrough (AUC 0.7429)
**Goal**: Discuss stronger margin forms before implementation

---

## Current Effective Form



Only captures: target projection along R_n -> R_a direction

---

## Candidate Enhancements

### 1. Squared Margin (Priority 1 - Recommended)



- Remove sign, only look at direction consistency strength
- ∈ [0, 1], still lambda-free
- Simplest: only change one line of code

**Pros**:
- Removes sign dependency
- Anomaly nodes may not strictly shift toward R_a, but direction consistency still matters
- Simplest implementation

**Cons**:
- Loses "toward R_n or R_a" direction info

---

### 2. Distance-Weighted Margin (Priority 2)



Or:



- Weight margin by how far target is from R_n
- Intuitively: farther deviation should have stronger margin signal

**Pros**:
- Distance information added
- Matches intuition: farther deviation = stronger signal

**Cons**:
- Scale issues, may need normalization
- Ratio unstable when ||ra_mean - rn_mean|| small

---

### 3. Learnable Margin Scale (Priority 3)



where alpha is head parameter, initialized to 1.

- Auto-adapt to dataset-specific margin importance

**Pros**:
- Automatic scale adaptation
- Still theory-aligned

**Cons**:
- Introduces one parameter
- But initialization to 1 may keep stable

---

### 4. Two-Sided Margin (Priority 4)



- Fine-grained separation of normal-side and deviation-side geometry

**Pros**:
- More detailed geometry reading

**Cons**:
- Increased complexity, may overfit

---

### 5. Margin + Cosine Similarity (Priority 5)



**Pros**:
- Both relative direction and absolute similarity

**Cons**:
- cos_sim may overlap with learned s_n/s_a

---

### 6. Margin as Separate Loss (Priority 6 - Not Recommended)



Or unsupervised:



**Pros**:
- Independent margin supervision

**Cons**:
- Introduces new loss, complexity
- Violates "only change head form, not loss" principle
- May need validation for threshold

---

## Recommendation

Based on:
- Simplicity first
- Theory interpretability first
- No-validation constraint first

Recommended order:

| Priority | Form | Reason |
|---|---|---|
| **1** | Squared margin | Simplest, one-line change, remove sign |
| **2** | Distance-weighted | Add distance info, intuitive |
| **3** | Learnable scale | Auto-adapt, init to 1 stable |
| **4** | Two-sided margin | Fine-grained but may overfit |
| **5** | Margin + cos_sim | May overlap with s_n/s_a |
| **6** | Margin as loss | Too complex, violates simplicity |

---

## Next Step

Try squared margin first on Elliptic:



---

_This discussion preserved before implementation._
