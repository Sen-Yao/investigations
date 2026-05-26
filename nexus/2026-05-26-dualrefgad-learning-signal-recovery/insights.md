# Insights — DualRefGAD Learning Signal Recovery

## Initial framing

This investigation starts from a route-level decision: the previous Reliability / Heterogeneous Proxy Map investigation should close because it answered the proxy-map / shallow-gate review question. The new scientific unit is learning-signal recovery.

The central distinction is:

- Old question: can C-LEG3 response-matrix oracle categories be mapped to label-free proxy signals?
- New question: can those signals be recovered into a reliable, auditable, label-free target `T(v)` that is worth training later?

## Initial gate

Do not promote a learnable head unless fixed-formula or low-capacity diagnostics show at least one stable continuation signal:

- AP improvement;
- reduced false-positive reintroduction;
- preserved anomaly retention;
- non-shortcut behavior vs `margin`, `mat_mean`, degree, and rejection;
- seed-consistent direction.

<!-- ABCD_PROBE_RESULT_BEGIN -->
# ABCD probe terminal summary

- Job: `exp_20260526_152642_dualrefgad_learning_signal_abcd_probe`
- Status: `finished`
- Dataset: `elliptic`
- Seeds: `[0, 1, 2, 3, 4]`
- Generated: 2026-05-26T19:23:31

## Variant `old_exact_080_regime`

- margin AUC: 0.7964; AP: 0.5185
- mat_mean AUC: 0.8104; AP: 0.5593
- Layer1 decision: `LAYER1_LABEL_FREE_GATE_NOT_PROMOTED_USE_AS_DIAGNOSTIC`
- D/readiness: `FIXED_FORMULA_OR_DIAGNOSTIC_ONLY_REVIEW_REQUIRED`
- Best Layer1: `L1_lfgate_q0.1_qf0.1_aa0.5_am0.1_l20.01` AUC=0.8078 AP=0.5572 ΔAUC=-0.0026 ΔAP=-0.0022
- Best fixed-formula by AUC: `L0_sigmoid_gate_a2_b1_l0.25_m0.5` AUC=0.8111 AP=0.5581
- A/top-K categories: `{"introduced_false_positives_mat_only_normal": {"mean": 1079.2, "std": 45.43302763409016, "min": 1011.0, "max": 1148.0}, "lost_anomalies_margin_only_true_positive": {"mean": 143.2, "std": 47.788701593577535, "min": 82.0, "max": 217.0}, "removed_false_positives_margin_only_normal": {"mean": 1268.6, "std": 83.71523158900058, "min": 1161.0, "max": 1355.0}, "rescued_anomalies_mat_only_true_positive": {"mean": 332.6, "std": 65.34707338511802, "min": 232.0, "max": 418.0}}`

## Artifact paths

- Aggregate JSON: `/home/openclawvm/investigations/nexus/2026-05-26-dualrefgad-learning-signal-recovery/experiments/outputs/dualrefgad_learning_signal_abcd_probe.json`
- Progress JSON: `/home/openclawvm/investigations/nexus/2026-05-26-dualrefgad-learning-signal-recovery/experiments/outputs/dualrefgad_learning_signal_abcd_probe.progress.json`
- Log: `/home/openclawvm/investigations/nexus/2026-05-26-dualrefgad-learning-signal-recovery/experiments/logs/dualrefgad_learning_signal_abcd_probe.log`
<!-- ABCD_PROBE_RESULT_END -->

## ABCD terminal interpretation — 2026-05-26

ABCD probe has finished as a runner-registered pure probe (`exp_20260526_152642_dualrefgad_learning_signal_abcd_probe`) on Elliptic, seeds `[0,1,2,3,4]`, using `old_exact_080_regime / C-LEG3`. The result should be interpreted as a **machine-learning-method step** rather than merely a failed promotion gate:

- Baselines: `margin` AUC/AP = 0.7964 ± 0.0043 / 0.5185 ± 0.0180; `mat_mean` AUC/AP = 0.8104 ± 0.0068 / 0.5593 ± 0.0279.
- Phase A confirmed that `mat_mean` recovers many anomalies that scalar margin misses: rescued anomalies mean `332.6`, while also introducing false positives mean `1079.2`. This makes the problem a target-shaping problem, not a simple score-selection problem.
- Phase B fixed-formula decomposition produced usable continuation hints. Best AUC formula `L0_sigmoid_gate_a2_b1_l0.25_m0.5` reached AUC `0.8111` and AP `0.5581`; best AP formula `L0_mix_only` reached AP `0.5624` with ΔAP `+0.0031`.
- Phase C showed reliability-like strategies are highly correlated with `mat_mean` / top-K overlap; this weakens method-promotion but proves the response-matrix relation can be parameterized and audited.
- Phase D best shallow label-free gate `L1_lfgate_q0.1_qf0.1_aa0.5_am0.1_l20.01` reached AUC `0.8078` and AP `0.5572`, with ΔAUC `-0.0026` and ΔAP `-0.0022` vs `mat_mean`.

**Updated interpretation.** The earlier phrase “not promoted” should not be read as “no machine-learning method”. The leap is that ABCD converts hand-written response-matrix diagnostics into a low-capacity, label-free, optimizable gate with explicit anchor construction, loss, regularization, and readiness criteria. Current evidence says: method form is established; current target is not yet strong enough to become the main trainable objective.

**Continuation.** Keep ABCD as diagnostic + target-shaping framework. Next work should improve the target source rather than increase head capacity: decompose reference relation by normal/deviation side, penalize false-positive reintroduction directly at fixed-formula level, and test low-capacity differentiable gates only after fixed targets pass AP/FP gates.
