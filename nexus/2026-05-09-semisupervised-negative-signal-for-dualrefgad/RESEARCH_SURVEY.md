# Semi-supervised Negative Signal Mechanism Audit

> Investigation: `2026-05-09-semisupervised-negative-signal-for-dualrefgad`  
> Date: 2026-05-09  
> Status: draft audit after proxy-metric critique

## 0. Why this audit replaces the proxy-first route

The earlier N1/N2/N3/N4 proxy metrics should **not** be treated as primary scientific evidence. They used a human-designed negative-pair rule and evaluated it with a human-designed margin score. Good proxy AUC/AP only shows compatibility between the rule and the score geometry, not validity for real anomaly detection.

Therefore the next step is not to optimize proxy AUC. The right question is:

> In a normal-only semi-supervised GAD protocol, what assumptions justify a negative or contrastive training signal, and can those assumptions be translated into DualRefGAD without using true anomaly labels?

This audit compares GGAD, VecGAD, and RHO from that perspective.

---

## 1. GGAD: pseudo anomalies from anomaly priors

### Sources checked

- Official repository: `mala-lab/GGAD` README and code.
- Paper abstract / README statement: GGAD generates pseudo anomaly nodes, called outlier nodes, to provide negative node samples for a discriminative one-class classifier.
- Local code inspection:
  - `/tmp/GGAD/utils.py`
  - `/tmp/GGAD/model.py`
  - `/tmp/GGAD/run.py`

### How GGAD constructs negative signal

GGAD starts with labeled normal nodes only. It splits normal training nodes into:

- `normal_label_idx`: normal nodes used as positive / normal samples.
- `abnormal_label_idx`: a subset of normal nodes selected to generate pseudo outliers.

In official code, pseudo-outlier indices are selected from normal labeled nodes:

```python
all_normal_label_idx = [i for i in idx_train if ano_labels[i] == 0]
normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * rate)]
if dataset in ['Amazon']:
    abnormal_label_idx = normal_label_idx[: int(len(normal_label_idx) * 0.05)]
else:
    abnormal_label_idx = normal_label_idx[: int(len(normal_label_idx) * 0.15)]
```

The model then transforms these selected normal embeddings into pseudo outlier embeddings using neighborhood aggregation and perturbation. In `model.py`, the outlier embedding is produced by aggregating neighbors of the selected nodes and applying a linear layer:

```python
neigh_adj = adj[0, sample_abnormal_idx, :]
emb_con = torch.mm(neigh_adj, emb[0, :, :])
emb_con = self.act(self.fc4(emb_con))
emb_combine = torch.cat((emb[:, normal_idx, :], torch.unsqueeze(emb_con, 0)), 1)
```

Training uses at least three signals:

1. **BCE discrimination** between real normal nodes and generated outliers.
2. **Local affinity margin** that encourages normal affinity to exceed pseudo-outlier affinity.
3. **Reconstruction / proximity term** between pseudo outlier and perturbed source embedding.

From `run.py`:

```python
lbl = torch.unsqueeze(torch.cat(
    (torch.zeros(len(normal_label_idx)), torch.ones(len(emb_con)))),
    1).unsqueeze(0)
loss_bce = b_xent(logits, lbl)

confidence_margin = 0.7
loss_margin = (confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)

diff_attribute = torch.pow(emb_con - emb_abnormal, 2)
loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))

loss = loss_margin + loss_bce + loss_rec
```

### Theoretical assumption

GGAD's negative signal is justified by an **anomaly-prior generation assumption**:

> Even without real anomaly labels, one can generate useful pseudo anomalies from normal nodes if the generated nodes mimic known structural/feature properties of anomalies.

The README states two priors:

- **asymmetric local affinity**
- **egocentric closeness**

So GGAD is not merely “random mismatch”. It tries to synthesize negative samples that resemble expected anomalies in both structure and feature representation.

### Relevance to DualRefGAD

GGAD supports the idea that DualRefGAD may use pseudo anomalies, but only if the pseudo anomaly is grounded in a normal-manifold departure assumption. A simple relation mismatch such as N1/N2 is weaker because it lacks an explicit anomaly prior.

Transferable components:

| GGAD component | Transferability to DualRefGAD | Comment |
|---|---:|---|
| Pseudo outliers from normal nodes | High | Allowed under normal-only protocol if generated from normal nodes only. |
| Local affinity margin | Medium | Could map to reference affinity / relation consistency. |
| BCE normal vs pseudo-outlier | Medium | Works only if pseudo-outlier construction is theoretically justified. |
| Structural priors | High | Need DualRef-specific priors, not arbitrary pair mismatch. |

---

## 2. VecGAD: pseudo anomalies from structured reconstruction residuals

### Sources checked

- Local repository clone: `Sen-Yao/GGADFormer`.
- Local documentation: `/tmp/GGADFormer/docs/VecGAD.md`.
- Code:
  - `/tmp/GGADFormer/model.py`
  - `/tmp/GGADFormer/SGT.py`
  - `/tmp/GGADFormer/run.py`
  - `/tmp/GGADFormer/utils.py`

### How VecGAD constructs negative signal

VecGAD uses only a small labeled-normal set. It generates pseudo anomalies from normal nodes, but the key distinction from GGAD is that the perturbation direction is intended to be guided by **structured reconstruction error**, not by raw random noise or arbitrary mismatch.

The theory in `docs/VecGAD.md` defines token reconstruction error:

```text
e_i^(tok) = D_tok(h_i) - t_i
```

and lifts it into embedding space:

```text
e_i^(emb) = P(e_i^(tok))
```

Then pseudo anomaly embeddings are generated as:

```text
h_tilde_i = h_i + beta * e_i^(emb)
```

The code path also uses normal nodes for generation:

```python
normal_for_generation_emb = emb[:, normal_for_generation_idx, :]
noise = torch.randn(normal_for_generation_emb.size(), device=self.device) * args.var + args.mean
noised_normal_for_generation_emb = normal_for_generation_emb + noise
neigh_adj = adj[0, normal_for_generation_idx, :]
outlier_emb = ...
emb_combine = torch.cat((emb[:, normal_for_train_idx, :], torch.unsqueeze(outlier_emb, 0)), 1)
```

Training uses generated outliers as negative samples with BCE:

```python
lbl = torch.unsqueeze(torch.cat(
    (torch.zeros(len(normal_for_train_idx)), torch.ones(len(outlier_emb)))),
    1).unsqueeze(0)
loss_bce = b_xent(logits, lbl)
```

Additional terms include reconstruction loss, central/ring alignment, contrastive loss, and optional margin loss depending on configuration.

### Theoretical assumption

VecGAD's negative signal is justified by a **normal-manifold residual assumption**:

> A model trained on normal nodes learns a normality manifold. The structured reconstruction residual identifies the direction in which a normal node is least compatible with the learned normal pattern. Moving a normal embedding along that residual direction creates a boundary-aware hard negative.

This is stronger than an arbitrary proxy because the negative direction comes from model failure under the normality assumption.

### Relevance to DualRefGAD

VecGAD is the most relevant source for DualRefGAD. DualRefGAD already has reference-derived directions:

- `u_i = h_i - r_n(i)` target deviation from normal reference.
- `d_i = r_a(i) - r_n(i)` deviation reference direction.

But the important lesson is not “use any deviation direction”. The lesson is:

> The negative direction should be derived from an interpretable normality residual, ideally one that identifies how the node fails to match normal references.

Transferable components:

| VecGAD component | Transferability to DualRefGAD | Comment |
|---|---:|---|
| Structured residual as negative direction | Very high | DualRefGAD can define residual relative to normal references. |
| Boundary-aware hard negative | Very high | Avoid trivial far-away negatives. |
| BCE against pseudo anomalies | Medium-high | Valid if pseudo anomalies are residual-grounded. |
| Center/ring/CPA constraint | Medium | Useful to prevent pseudo anomalies from drifting too far. |
| Pure relation mismatch | Low | Not the main VecGAD principle. |

---

## 3. RHO: no pseudo anomaly; normality alignment and hypersphere scoring

### Sources checked

- Official repository: `mala-lab/RHO` README and code.
- Local knowledge: `knowledge/Nexus/alvis-audit-rho.md`.
- Code:
  - `/tmp/RHO/model.py`
  - `/tmp/RHO/train.py`
  - `/tmp/RHO/utils.py`

### How RHO constructs negative signal

RHO does **not** generate pseudo anomaly nodes. Its training signal consists of:

1. **One-class hypersphere objective** on labeled normal nodes.
2. **Graph Normality Alignment (GNA)** via InfoNCE-style consistency between global and local views.

From `train.py`, RHO computes centers and minimizes normal-node distances to them:

```python
outputs_global, outputs_local, nce_loss = model(adj, features)
dist_global = torch.sum((outputs_global[idx_train] - center_global) ** 2, dim=1)
dist_local = torch.sum((outputs_local[idx_train] - center_local) ** 2, dim=1)
dist = 0.5 * dist_global + 0.5 * dist_local
loss = torch.mean(dist) + args.alpha * nce_loss
```

At test time, anomaly score is distance from the learned normal centers:

```python
scores = ((torch.sum((outputs_global[idx] - center_global) ** 2, dim=1)) +
          (torch.sum((outputs_local[idx] - center_local) ** 2, dim=1))) / 2
```

GNA is implemented as InfoNCE between global and local views. The positive pair is the same node across views; negatives are other nodes in the batch/full graph:

```python
pos_mask = torch.eye(z1.shape[0])
neg_mask = 1 - pos_mask
loss_0 = self.infonce(z1, z2, pos_mask, neg_mask, temperature)
loss_1 = self.infonce(z2, z1, pos_mask, neg_mask, temperature)
```

The `infonce` denominator uses both cross-view and anchor-anchor similarities over negative masks:

```python
sim_anchor = self.similarity(anchor, anchor) / tau
exp_sim_anchor = torch.exp(sim_anchor) * neg_mask
sim = self.similarity(anchor, sample) / tau
exp_sim = torch.exp(sim) * neg_mask
log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True)) - torch.log(exp_sim_anchor.sum(dim=1, keepdim=True))
loss = log_prob * pos_mask
```

### Theoretical assumption

RHO's negative signal is not pseudo anomaly supervision. It is a **normality consistency assumption**:

> Normal nodes may have heterogeneous homophily patterns, but a robust normal representation should remain consistent across adaptive global and local spectral views. Anomalies receive high scores because they fail to stay close to normal centers in these aligned representations.

The negatives in InfoNCE are other nodes, but they are not labeled as anomalies. They act as contrastive normalization, not as anomaly pseudo-labels.

### Relevance to DualRefGAD

RHO suggests a clean alternative to pseudo-negative training:

> Train DualRefGAD to make normal reference views consistent, and use distance/inconsistency from the learned normal relation as anomaly score.

Transferable components:

| RHO component | Transferability to DualRefGAD | Comment |
|---|---:|---|
| Normal-only hypersphere objective | Very high | Directly protocol-clean. |
| Cross-view normality alignment | Very high | DualRef has natural views: target-normal, target-deviation, local/global references. |
| InfoNCE with in-batch negatives | Medium | Negatives are not anomalies; must be framed as view discrimination, not anomaly labels. |
| No pseudo anomaly | High | Avoids questionable negative construction. |

---

## 4. Cross-method abstraction

| Method | Uses true anomaly labels in training? | Constructs pseudo anomalies? | Negative signal type | Theoretical basis |
|---|---:|---:|---|---|
| GGAD | No | Yes | generated outlier nodes + BCE/margin | anomaly priors: asymmetric local affinity, egocentric closeness |
| VecGAD | No | Yes | residual-guided hard negatives + BCE | normal-manifold structured reconstruction residual |
| RHO | No | No | one-class center + cross-view InfoNCE | heterogeneous normal homophily + normality alignment |

Key distinction:

- GGAD / VecGAD create **pseudo anomaly samples**.
- RHO creates **normal consistency pressure** and scores distance from normality.

For DualRefGAD, the proxy-pair route is weaker than all three because it lacks an independent theoretical assumption. It should not be the primary path.

---

## 5. Candidate DualRefGAD training targets from this audit

### Candidate A: DualRef Normality Alignment (RHO-style, highest scientific cleanliness)

Train only on labeled normal nodes. Define multiple normality views from DualRef outputs, e.g.:

- target-normal relation view: `z_n = f(h_i, r_n(i))`
- deviation-reference relation view: `z_d = g(h_i, r_n(i), r_a(i))`
- local/global reference views if available

Training objective:

1. Pull labeled normal nodes toward a normal relation center.
2. Align local/global or target/reference views for the same normal node.
3. Use final anomaly score as distance or inconsistency from learned normal relation.

This avoids pseudo anomalies entirely and is protocol-clean.

Risk: if the learned normality score collapses or becomes too smooth, it may underperform margin-only.

### Candidate B: DualRef Residual-Guided Hard Negative (VecGAD-style)

Define a DualRef residual for normal nodes, for example:

```text
residual_i = h_i - reconstruct_from_normal_reference(h_i, r_n(i))
```

or a token/relation residual from the GT/VecGAD encoder. Generate pseudo anomaly embeddings by moving normal embeddings along this residual direction, with a boundary constraint:

```text
h_tilde_i = h_i + beta * normalize(P(residual_i))
```

Then train a head to separate labeled normal embeddings from residual-guided pseudo anomalies.

This is more defensible than N1/N2 mismatch because the negative direction is tied to failure under normal reconstruction/reference explanation.

Risk: introduces beta/radius/projection choices; must avoid too many hyperparameters.

### Candidate C: DualRef Affinity/Boundary Pseudo-Outlier (GGAD-style)

Generate pseudo outliers from normal nodes by modifying their relation to references such that:

- normal-reference affinity decreases;
- pseudo node remains near the normal boundary;
- generation follows an explicit anomaly prior, not arbitrary mismatch.

This can use GGAD's idea of local affinity margin, but adapted to reference relation rather than graph-neighbor affinity.

Risk: weaker than VecGAD if the anomaly prior is not clearly defined.

---

## 6. Recommended experimental design

### Stage 1: no training, theory-to-implementation checks only

Do not run another proxy AUC sweep. Instead, implement or inspect whether the following quantities are available under frozen GT embeddings:

1. normal relation center stability across seeds;
2. local/global or normal/deviation view consistency for labeled normal nodes;
3. reconstruction/residual vectors from existing VecGAD/GT pipeline;
4. whether residual magnitude/direction correlates with real anomaly labels only at test time.

These are diagnostic checks for assumptions, not proof of method superiority.

### Stage 2: minimal training probe, only after Stage 1 supports an assumption

Preferred first probe:

```text
RHO-style DualRef Normality Alignment
```

Reason:

- protocol-clean;
- no pseudo anomaly labels;
- directly addresses the theoretical objection;
- fewer arbitrary negative-generation hyperparameters.

Minimal setup:

- Dataset: Elliptic
- seed: 0 only
- fixed GT embeddings
- train labels: normal-only
- no validation cherry-pick
- compare against:
  - margin-only epoch0 baseline
  - current `dual_margin_two_score` final
  - simple one-class center on `h_i` as sanity baseline

If RHO-style alignment fails, second probe:

```text
VecGAD-style residual-guided hard negative
```

but only if residuals are well-defined and not arbitrary.

---

## 7. Current conclusion

The next DualRefGAD head should not be selected based on N1/N2 proxy AUC. The strongest theoretically grounded paths are:

1. **Normality alignment without pseudo anomalies** (RHO-style): safest and cleanest.
2. **Residual-guided boundary hard negatives** (VecGAD-style): potentially stronger but more design-heavy.
3. **Affinity-prior pseudo outliers** (GGAD-style): possible, but requires a clear DualRef-specific anomaly prior.

Immediate next action:

> Design a DualRef Normality Alignment minimal probe before any pseudo-negative training.
