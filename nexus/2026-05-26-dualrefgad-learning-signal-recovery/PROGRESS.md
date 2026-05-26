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
