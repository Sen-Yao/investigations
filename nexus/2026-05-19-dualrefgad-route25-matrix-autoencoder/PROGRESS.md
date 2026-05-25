# Progress

## 2026-05-19 — Investigation created

- Created independent investigation for Route2.5 Matrix Autoencoder.
- User approved execution of seed0 probe.
- Planned runner-registered probe using `experiment-runner profile=probe`.

### Probe summary

- Input: flattened response matrix `M ∈ R^(4×16)`.
- Train: labeled-normal nodes only.
- Evaluate: AUC/AP on test split, labels diagnostic-only.
- Baselines: margin / -margin / matrix scalar summaries / degree / rejection.


## 2026-05-19 — Seed0 probe completed

Runner job: `exp_20260519_165831_route25_matrix_ae_elliptic_seed0`  
Remote: HCCS-25 GPU0  
Status: finished, exit code 0  
Runtime: 42.3s

### Key result

- Decision: `INCONCLUSIVE`
- Strongest scalar baseline: `neg_mat_mean` AUC=0.6377, AP=0.1405
- Best AE: `ae_mse_latent8` AUC=0.6028, AP=0.1140
- AE Spearman with margin: -0.4560
- AE top5 Jaccard with margin: 0.0036

### Immediate interpretation

Seed0 does **not** promote Matrix AE yet: best AE underperforms the best scalar baseline by 0.0349 AUC. It is above the DROP threshold but not close enough to beat/replace scalar summaries. Low top-k overlap suggests it may still be a complementary diagnostic, but this single seed is insufficient for a method claim.

## 2026-05-19 — Seed1-4 probes completed and 5-seed aggregation

Runner jobs:
- `exp_20260519_170908_route25_matrix_ae_elliptic_seed1` — finished, HCCS-25 GPU1
- `exp_20260519_170909_route25_matrix_ae_elliptic_seed2` — finished, HCCS-25 GPU2
- `exp_20260519_170909_route25_matrix_ae_elliptic_seed3` — finished, HCCS-25 GPU3
- `exp_20260519_170909_route25_matrix_ae_elliptic_seed4` — finished, HCCS-25 GPU4

All outputs were pulled back into `experiments/outputs/`; logs were pulled into `experiments/logs/`. Aggregate JSON: `experiments/outputs/route25_matrix_ae_elliptic_5seed_aggregate.json`.

### 5-seed aggregate

- Best AE AUC: 0.6115 ± 0.0567
- Best AE AP: 0.1511 ± 0.0696
- Best scalar AUC: 0.6500 ± 0.0111
- Best scalar AP: 0.1897 ± 0.0461
- AE - scalar ΔAUC: -0.0385 ± 0.0647
- AE - scalar ΔAP: -0.0387 ± 0.0951
- Best AE Spearman with margin: -0.4218 ± 0.1051
- Best AE top5 Jaccard with margin: 0.0075 ± 0.0061
- Per-seed decisions: PROMOTE=2, DROP=2, INCONCLUSIVE=1

### Updated interpretation

The multi-seed result does **not** support promoting Matrix AE as a stable method component. Although seed1 and seed3 individually cross the probe's PROMOTE rule, seed2 and seed4 drop below threshold and the 5-seed mean underperforms the strongest scalar matrix summaries by ΔAUC=-0.0385. The signal is seed-unstable and weaker than cheap scalar response-matrix statistics.

Conclusion boundary: this is valid runner-registered exploratory evidence for Route2.5 diagnosis, not a full SOTA experiment. It argues against spending a full method sweep on this exact normal-only Matrix AE head without repairing the representation/regime problem first.

## 2026-05-19 — Phase 1 AE instability audit completed

Runner job: `exp_20260519_175609_route25_matrix_ae_phase1_instability_aud`  
Remote: HCCS-25 GPU0  
Status: finished, exit code 0  
Runtime: 114.6s

### Audit protocol

- Reused the Route2.5 frozen-encoder response-matrix construction.
- For each split seed `{0,1,2,3,4}`, reran AE training with AE init seeds `{0,1,2,3,4}`.
- Audited latent dimensions `{8,16}`; labels remained diagnostic-only.
- Output: `experiments/outputs/route25_matrix_ae_phase1_instability_audit.json`.

### Phase 1 aggregate

- Best repeat AE AUC: 0.6143 ± 0.0499
- Best repeat AE ΔAUC vs scalar: -0.0357 ± 0.0571
- Scalar AUC: 0.6500 ± 0.0100
- Mean within-split AE AUC std: 0.0033 ± 0.0019
- Promote-repeat splits: 2/5
- Drop-repeat splits: 2/5
- Decision: `SPLIT_REFERENCE_INSTABILITY_DOMINATES__DO_NOT_PROMOTE`

### Interpretation

Repeating AE initializations does **not** repair Route2.5 Matrix AE. Within a fixed split/reference construction, AE AUC is very stable (mean within-split std ≈ 0.0033), but across split seeds the best-repeat AE remains unstable and underperforms scalar summaries on average. This localizes the failure more to split/reference/representation regime instability than to random AE initialization noise.

Conclusion boundary: Phase 1 strengthens the negative method decision for this exact Matrix AE head. It does not reject response-matrix features globally; it says the next repair should target reference distribution / orientation regime before adding learnable capacity.

## 2026-05-19 — Phase 2 learning-strategy posthoc diagnostic completed

Runner job: `exp_20260519_184959_route25_matrix_ae_phase2_learning_strate`  
Execution: local zero-new-training posthoc probe over existing Phase-1 repeated-AE output  
Status: finished, exit code 0  
Output: `experiments/outputs/route25_matrix_ae_phase2_learning_strategy_posthoc.json`.

### Audit protocol

- No new model training and no GPU use; this is a low-cost posthoc diagnostic over Phase-1 results.
- Tested whether cheap learning-strategy choices could repair Matrix AE instability:
  - deployable validation-loss selector;
  - fixed latent dimension `{8,16}`;
  - label-oracle best-AUC selector as an upper-bound autopsy only.
- Labels remain diagnostic-only; the label-oracle selector is explicitly **not deployable**.

### Phase 2 aggregate

- Scalar baseline AUC: 0.6500 ± 0.0100
- Oracle best-AUC selector AE AUC: 0.6143 ± 0.0499; ΔAUC vs scalar: -0.0357 ± 0.0571
- Validation-loss selector AE AUC: 0.6099 ± 0.0488; ΔAUC vs scalar: -0.0401 ± 0.0556
- Validation-loss selector promote splits: 1/5; drop splits: 2/5
- Fixed latent=8 ΔAUC: -0.0406 ± 0.0586; promote 1/5; drop 2/5
- Fixed latent=16 ΔAUC: -0.0357 ± 0.0571; promote 2/5; drop 2/5
- Spearman-like rank correlation of validation loss with AUC across all runs: -0.0618
- Decision: `NO_CHEAP_LEARNING_STRATEGY_REPAIR__KEEP_DROP_DECISION`

### Interpretation

Cheap learning-strategy fixes do **not** repair Route2.5 Matrix AE. Even the label-oracle best-AUC selector remains below scalar response-matrix summaries on average, and the deployable validation-loss selector is worse. Fixed latent capacity also fails. The negative decision is therefore not just “we selected the wrong AE checkpoint/latent”; the failure remains localized to reference/orientation/representation regime rather than AE training policy.

## 2026-05-19 — Response matrix information-mining pass completed

This pass did not add new training. It cross-analyzed existing scalar routes and Phase-2 posthoc AE evidence to ask a narrower question: **which pieces of the response matrix remain valuable after repeated negative AE evidence?**

### Cross-analysis summary

- The most stable scalar families across the existing 5-seed Route2 matrix diagnostic are:
  - `quantile` AUC=0.5134 ± 0.0485
  - `trimmed` AUC=0.5052 ± 0.0517
  - `mean` / `weighted` AUC=0.5022 ± 0.0547
- `max_mean` is the clear loser: AUC=0.4434 ± 0.0434.
- The useful signal is not “matrix summary in general”; it is the **sign/orientation family** and a mild **distribution-shape family**.
- In Phase 2, the split-level scalar winners were dominated by `neg_mat_mean` (4/5 splits) with one `mat_std` split.
- The AE posthoc results did **not** change the route decision: cheap repair by checkpoint/latent selection still fails.

### Interpretation

The response matrix still appears to contain information, but the useful part is more likely to live in:

1. sign/orientation flips (`neg_mat_mean` vs `mat_mean`),
2. distribution shape (`quantile`, `trimmed`, `mat_std`),
3. regime partitioning (why some split/reference regimes promote while others drop).

This means the next investigation should be about **what structure the matrix is encoding**, not about forcing the matrix through a larger AE.

## 2026-05-20 — Stage A preflight and next-probe registration plan

Timestamp: 2026-05-20 09:25:52 CST

### Context located

Current Stage A context is this investigation: `2026-05-19-dualrefgad-route25-matrix-autoencoder`.

The completed evidence stack is:

- seed0 Matrix AE probe: `exp_20260519_165831_route25_matrix_ae_elliptic_seed0`, finished on HCCS-25 GPU0;
- seed1-4 Matrix AE probes: finished on HCCS-25 GPU1-4;
- Phase 1 instability audit: `exp_20260519_175609_route25_matrix_ae_phase1_instability_aud`, finished on HCCS-25 GPU0;
- Phase 2 learning-strategy posthoc: `exp_20260519_184959_route25_matrix_ae_phase2_learning_strate`, finished locally with no new training.

### Environment preflight

Live read-only HCCS-25 probe succeeded:

- host: `VM-2-25-ubuntu`;
- GPU: 8 × RTX 2080 Ti, all queried as 0 MiB used / 11011 MiB free / 0% util at probe time;
- `~/DualRefGAD`: present;
- `~/DualRefGAD/dataset/elliptic.mat`: present;
- `~/DualRefGAD/experiments/scripts/route25_matrix_autoencoder_probe.py`: missing.

Nezha WebSocket status probe failed with `InvalidMessage: did not receive a valid HTTP response`, so current machine state is based on direct SSH/nvidia-smi evidence rather than Nezha.

### Stage A decision boundary

Existing results are already enough to keep the **negative decision for normal-only Matrix AE as-is**:

- 5-seed best AE AUC: 0.6115 ± 0.0567;
- strongest scalar AUC: 0.6500 ± 0.0111;
- AE ΔAUC vs scalar: -0.0385 ± 0.0647;
- Phase 1 shows AE init noise is not the culprit;
- Phase 2 shows cheap checkpoint/latent selection does not repair the route.

Therefore Stage A should not be a bigger-AE or checkpoint-selection run. The next valid Stage A probe is a **mechanistic response-matrix decomposition** focused on sign/orientation, distribution shape, and regime/reference sensitivity.

### Proposed Stage A probe before launch

Name: `route25_stage_a_matrix_orientation_regime_probe`

Objective:

> Explain why the response matrix retains scalar/orientation signal while normal-only Matrix AE remains unstable; identify whether useful signal is concentrated in sign/orientation, distribution-shape, or split/reference regimes.

Protocol:

- no new model training unless the implementation has to recompute frozen embeddings/reference matrices;
- labels diagnostic-only for AUC/AP/autopsy;
- use real Elliptic only;
- seeds `{0,1,2,3,4}`;
- compare scalar families: `mat_mean`, `neg_mat_mean`, `mat_std`, median/quantile/trimmed, high-ratio, and margin orientations;
- stratify by at least: split seed, reference anomaly ratio, rejection proxy, degree proxy, and scalar winner family;
- produce one JSON output plus a compact aggregate table suitable for updating `insights.md`.

Runner/operations plan:

1. create or reuse an investigation script under `experiments/scripts/`;
2. create a valid probe config under `experiments/configs/`;
3. validate with `experiment.py validate --profile probe`;
4. register with `experiment.py register --profile probe --kind probe`;
5. because HCCS-25 is currently missing the investigation scripts, sync the required `experiments/scripts/*.py` into `~/DualRefGAD/experiments/scripts/` before execution;
6. run via a Hermes-tracked SSH/background process, then mark the runner job terminal and pull outputs/logs back;
7. update `PROGRESS.md` and `insights.md`.

### Authorization status

User approved execution on 2026-05-20. Stage A was launched as runner-registered probe `exp_20260520_093352_route25_stage_a_matrix_orientation_regim`.

## 2026-05-20 — Stage A orientation/regime probe completed

Timestamp: 2026-05-20 09:38:56 CST

Runner job: `exp_20260520_093352_route25_stage_a_matrix_orientation_regim`  
Remote: HCCS-25 GPU0  
Status: finished, exit code 0  
Runtime: 223.8s  
Output: `experiments/outputs/route25_stage_a_matrix_orientation_regime_probe.json`  
Log: `experiments/logs/route25_stage_a_matrix_orientation_regime_probe.log`

### Protocol

- Reused the exact Route2.5 frozen encoder / reference / response-matrix construction.
- No Matrix AE or new learnable head was trained.
- Seeds: `{0,1,2,3,4}`.
- Labels were used only for AUC/AP and autopsy.
- Compared orientation and shape families: `mat_mean`, `neg_mat_mean`, median/quantile, trimmed/top/bottom summaries, `mat_std`/IQR, margin orientations, rejection and degree proxies.
- Stratified best scalar by degree, rejection, and margin regimes.

### Aggregate results

- Best scalar-family AUC: 0.6636 ± 0.0173.
- `neg_mat_mean` AUC: 0.6490 ± 0.0110.
- Winner counts:
  - `neg_mat_median`: 1/5
  - `mat_iqr`: 1/5
  - `neg_mat_q75`: 1/5
  - `neg_mat_bottom5_mean`: 1/5
  - `neg_mat_top5_mean`: 1/5
- Orientation family counts:
  - negative-orientation family: 4/5
  - other: 1/5
- `mat_mean - neg_mat_mean` AUC: -0.2980 ± 0.0220.
- Anomaly-reference anomaly ratio: 0.0267 ± 0.0072.
- Spearman(best score, margin): -0.7306 ± 0.1670.
- Spearman(best score, degree): -0.3174 ± 0.0670.

### Per-seed winners

- seed0: `neg_mat_median`, AUC/AP 0.6428 / 0.1465.
- seed1: `mat_iqr`, AUC/AP 0.6520 / 0.2362.
- seed2: `neg_mat_q75`, AUC/AP 0.6919 / 0.2537.
- seed3: `neg_mat_bottom5_mean`, AUC/AP 0.6579 / 0.1596.
- seed4: `neg_mat_top5_mean`, AUC/AP 0.6735 / 0.2365.

### Decision

`NEGATIVE_ORIENTATION_SIGNAL_STABLE__PRIORITIZE_SIGN_AWARE_SCALAR_OR_REGIME_SCORE`

Stage A strengthens the information-mining interpretation: the response matrix is not useless, but the useful signal is mostly **orientation/sign-aware scalar or shape structure**, not a normal-only Matrix AE reconstruction pattern. Negative orientation is stable across seeds, and the best scores are not dominated by degree proxy correlation. The next method-relevant direction should be a lightweight sign-aware scalar/regime score or reference-pool/orientation repair, not a bigger AE.



## 2026-05-21 — Evidence chain consolidated while waiting for parallel A/B/C scan

Timestamp: 2026-05-21 16:39 CST

### What is already established

- **Matrix AE as-is is not promoted**: 5-seed mean AUC/AP remains below the strongest scalar baseline; Phase 1 and Phase 2 both rule out initialization and cheap checkpoint/latent repair as the main issue.
- **Response matrix still carries signal**: Stage A shows a stable sign/orientation family and a weaker distribution-shape family.
- **Current reference regime is not the same as the old strong regime**: the old exact / old refs alignment probe recovers `mat_mean≈0.80`, while current refs flip toward `neg_mat_mean`.
- **AMRF is diagnostic, not yet a main method**: current refs yield only a tiny positive delta; old exact/semantic regimes do not beat scalar baselines.
- **The key control-variable lesson is now explicit**: do not change the historically strong sequence generator before proving the response-matrix signal itself under that fixed regime.

### What is still pending

- The long-running `exp_20260521_144041_route25_parallel_mat_mean_reference_scan` job is still running.
- Its A-CRA partial result already suggests negative orientation under the current setup, but B-LSS and C-LEG3 are needed before drawing the final regime comparison.

### What this means for the investigation boundary

This investigation has now evolved from a “Matrix AE probe” into a broader Route2.5 response-matrix mechanism record. The evidence chain should be read as:

1. fixed-route matrix AE instability;
2. sign/orientation and shape information still present;
3. current reference regime differs sharply from the historical strong regime;
4. therefore the next scientific step, after the pending scan lands, should fix C and study readout rather than keep changing the generator.
