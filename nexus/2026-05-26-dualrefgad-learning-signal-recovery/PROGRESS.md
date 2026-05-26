# Progress — DualRefGAD Learning Signal Recovery

## 2026-05-26 15:00 CST — Investigation created

Created new Nexus investigation as requested by SenYao for the long task with three fixed goals:

1. Create a new investigation skeleton, clean/sync current code git, remove abandoned temporary code, and prepare for new experimental code.
2. Based on the report `dualrefgad-learning-signal-recovery-discussion-2026-05-26.html`, prepare experimental code for all ABCD stages and run them one by one through the experiment-runner workflow while fully utilizing available GPUs.
3. Set watchdogs for ABCD; after D finishes, supplement the report with ABCD results, explain result meaning/definitions/potential/future directions, then organize git and the investigation.

Source report:

- `https://report.senyao.org/reports/2026/05/26/dualrefgad-learning-signal-recovery-discussion-2026-05-26.html`
- Local source: `/home/openclawvm/workspace/senyao-reports/site/reports/2026/05/26/dualrefgad-learning-signal-recovery-discussion-2026-05-26.html`

ABCD scope frozen from report:

- Phase A — Top-K failure autopsy extension.
- Phase B — Fragmentation decomposition probe.
- Phase C — Reference relation reliability probe.
- Phase D — Trainable target readiness.

Execution rules:

- Any experiment/probe must be runner-registered or explicitly recorded as protocol deviation.
- Anomaly labels are diagnostic-only.
- No manual WandB agent startup.
- Prefer a single bundled multi-variant pure probe with internal GPU task queue if runner has no native launch-probe path.
- Set split watchdog + terminal publisher cron: watchdog operationally monitors/pulls artifacts; publisher only acts after terminal aggregate JSON.

Immediate next steps:

1. Inspect current DualRefGAD code git state and identify authoritative remote checkout.
2. Clean abandoned temporary code safely, without deleting evidence artifacts.
3. Prepare ABCD probe script/config and validate locally with `experiment.py validate --profile probe`.
4. Preflight HCCS host/project/data/env/GPU state before registering/launching.

## 2026-05-26 15:18 CST — Remote code checkout preflight and cache cleanup

HCCS live preflight found the current usable DualRefGAD checkout on HCCS-25:

- Host: `linziyao@81.70.88.243` (`VM-2-25-ubuntu`)
- Project: `/home/linziyao/DualRefGAD`
- Git: `main...origin/main`, local HEAD `40d3ab0 Optimize reference selection for large datasets (Elliptic)`
- Upstream GitHub `refs/heads/main`: `40d3ab0e5affaee745c30e5d702b23f1a5ac46d7`, so the project checkout is at current upstream HEAD.
- Dataset: `dataset/elliptic.mat` exists.
- GPU: 8 × RTX 2080 Ti, all idle at preflight time.
- Other checked hosts: HCCS-80/HCCS-86/HCCS-88 routes unavailable; HCCS-85 reachable but no DualRefGAD checkout.

Remote `git fetch --dry-run` over HTTPS failed because HCCS-25 lacks GitHub credential context (`could not read Username for https://github.com`). Per hccs-cluster caveat, if future sync is needed, use agent-host `git ls-remote` / clone-transfer rather than expecting direct HCCS-25 fetch.

Cleanup performed:

- Removed generated Python bytecode caches (`__pycache__` / `*.pyc`) under remote `experiments/` and `investigations/`.
- No scientific scripts, configs, logs, JSON outputs, or evidence artifacts were deleted.
- Remote status after cleanup still shows untracked `experiments/` and `investigations/`; these are historical runner/probe artifacts and must be treated as evidence unless deliberately archived.
