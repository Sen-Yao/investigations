# Hypothesis

## H1: Global G_a should be query-independent

`G_a` should be a global node-level anomaly-reference eligibility score:

```text
G_a ∈ R^N
```

It should be computed once for the whole graph and reused for all query nodes. Query-specific relevance should be handled by `L_a(u|v)`.

## H2: Normal-calibrated deviation is a better G_a than raw q*c

Because the setting allows access to a small normal-only training set, `G_a` can legally use 5% normal labels to estimate the normal manifold. A node that deviates from this normal manifold is a stronger anomaly-reference candidate than one selected only by structural anomaly support `q*c`.

## H3: Additive G-L combination is preferable as the main formulation

Although multiplication can be interpreted as prior × likelihood, current `G` and `L` are not calibrated probabilities. Additive combination is more consistent with graph classification/anomaly detection scoring conventions and performs similarly or better in key probe metrics.

Main selection formula:

```text
S_a(u|v) = G_a(u) + L_a(u|v)
```

## H4: Full-model validation is now required

Reference purity probe has shown strong signal, but it does not guarantee final AUC/AP improvement. The next hypothesis is that higher-quality anomaly-reference tokens will be used by the Transformer and improve downstream classification.
