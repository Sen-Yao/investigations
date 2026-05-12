# Hypotheses

## H1: Structured readout is better aligned with DualRefGAD than single pooled scalar readout

A VecGAD-style head pools GT-interacted tokens into one embedding and feeds it to a scalar MLP. This is expressive but does not preserve the target / normal-side / deviation-side structure.

Hypothesis:

```text
A structured readout that keeps target, normal-side, and deviation-side representations separate before scoring will better use fixed dual-reference evidence.
```

## H2: Score decomposition can expose and improve `R_a` utilization

The score should not be an opaque scalar only. It should include interpretable sub-scores:

```text
s_n(v): normal compatibility
s_a(v): deviation support
s(v): final anomaly score from s_n, s_a, and interaction features
```

Hypothesis:

```text
Explicitly modeling normal compatibility and deviation support improves or clarifies anomaly scoring compared with a single scalar MLP.
```

## H3: `R_a` should be modeled as deviation-side evidence, not anomaly pseudo-label

`R_a` answers whether a non-normal-like deviation has a similar reference. It should not be used as direct anomaly supervision.

Hypothesis:

```text
A head that treats `R_a` as deviation-side evidence will be more stable and more scientifically defensible than a head that implicitly treats `R_a` as anomaly positive samples.
```

## H4: Fine-grained negative scenes should follow head semantics

The previous context replacement objective is useful as a probe but too coarse as a final training strategy.

Hypothesis:

```text
Negative scenes that separately perturb normal-side compatibility, deviation-side support, or their interaction will provide cleaner supervision than replacing the whole reference context.
```

## H5: GT training should be delayed until the head semantics are validated

Before training GT layers, the structured head should be tested with frozen GT embeddings.

Hypothesis:

```text
If a structured head improves over scalar MLP under frozen GT, then training projection / last GT layer becomes better motivated.
```
