# Initial Design: Two-Sided Score Head

## Baseline

```text
GT tokens -> pooled embedding z -> MLP(z) -> scalar anomaly score
```

## Structured readout candidate

```text
GT tokens -> z_v, z_n, z_a
r_n = relation(z_v, z_n)
r_a = relation(z_v, z_a)
r_inter = interaction(z_v, z_n, z_a)
s_n = normal_compatibility(r_n)
s_a = deviation_support(r_a)
s = final_score([s_n, s_a, r_inter])
```

## First experimental matrix

| variant | GT | readout | head | negative scene |
|---|---|---|---|---|
| scalar_mlp_baseline | frozen | pooled | scalar MLP | current context replacement |
| structured_readout | frozen | z_v/z_n/z_a | scalar final | current context replacement |
| decomposition_head | frozen | z_v/z_n/z_a | s_n + s_a + final | current context replacement |
| decomposition_split_mismatch | frozen | z_v/z_n/z_a | s_n + s_a + final | separate R_n / R_a mismatch |

The first phase should validate the head with frozen GT before training GT layers.
