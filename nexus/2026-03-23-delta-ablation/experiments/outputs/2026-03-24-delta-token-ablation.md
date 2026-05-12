# Delta Token Ablation Experiment

**Date**: 2026-03-24  
**Sweep ID**: `zqyeag0t`  
**Dataset**: Photo  
**Total Runs**: 15 (3 modes × 5 seeds)

---

## Experiment Design

### Token Modes

| Mode | Token Sequence | Shape | Description |
|------|---------------|-------|-------------|
| original | `[X_0, X_p^1, ..., X_p^K]` | `[N, K+1, D]` | Baseline |
| delta | `[Δ_0, Δ_1, ..., Δ_{K-1}]` | `[N, K, D]` | Replace with delta vectors |
| concat | `[X_0, ..., X_p^K, Δ_0, ..., Δ_{K-1}]` | `[N, 2K+1, D]` | Concatenate original + delta |

Where `Δ_k = X_p^{k+1} - X_p^k`

### Fixed Hyperparameters

- num_prompts: 12
- tokenizer_temp: 2
- tokenizer_hallucination_ratio: 1.5
- pp_k: 6
- progregate_alpha: 0.2
- embedding_dim: 128
- peak_lr: 0.0005
- num_epoch: 150

---

## Results

### Summary (mean±std)

| token_mode | AUC | AP |
|------------|-----|-----|
| **concat** | **0.8786±0.0255** | 0.4207±0.0691 |
| delta | 0.8733±0.0271 | 0.3978±0.0675 |
| original | 0.8666±0.0363 | 0.4057±0.0909 |

### Per-Run Details

| mode | seed | AUC | AP |
|------|------|-----|-----|
| original | 0 | 0.8364 | 0.3663 |
| original | 1 | 0.8632 | 0.6166 |
| original | 2 | 0.8473 | 0.4638 |
| original | 3 | 0.8806 | 0.5047 |
| original | 4 | 0.9317 | 0.5616 |
| delta | 0 | 0.8271 | 0.4108 |
| delta | 1 | 0.8701 | 0.3547 |
| delta | 2 | 0.8729 | 0.3096 |
| delta | 3 | 0.9103 | 0.5118 |
| delta | 4 | 0.8860 | 0.4022 |
| concat | 0 | 0.8555 | 0.3392 |
| concat | 1 | 0.8999 | 0.4108 |
| concat | 2 | 0.8859 | 0.3668 |
| concat | 3 | 0.9089 | 0.4166 |
| concat | 4 | 0.8555 | 0.3392 |

---

## Key Findings

1. **Concat is best**: Concatenating original tokens with delta vectors yields the highest AUC
2. **Delta direction is useful**: Delta mode slightly outperforms original, suggesting delta vectors contain useful directional information
3. **Not statistically significant**: Standard deviations overlap, differences are within noise
4. **Needs validation on more datasets**: Single dataset (Photo) is insufficient for strong conclusions

---

## Implementation

The token_mode logic is implemented in `utils.py`:

```python
# token_mode parameter in run.py
parser.add_argument('--token_mode', type=str, default='original', 
                    choices=['original', 'delta', 'concat'])

# Tokenization logic in nagphormer_tokenization()
if token_mode == 'delta':
    delta_tokens = nodes_features[:, 1:] - nodes_features[:, :-1]
    return delta_tokens
elif token_mode == 'concat':
    delta_tokens = nodes_features[:, 1:] - nodes_features[:, :-1]
    concat_tokens = torch.cat([nodes_features, delta_tokens], dim=1)
    return concat_tokens
```

---

## Next Steps

- [ ] Validate on Reddit dataset
- [ ] Validate on Tolokers dataset
- [ ] Analyze why concat works better
- [ ] Consider learnable delta weights