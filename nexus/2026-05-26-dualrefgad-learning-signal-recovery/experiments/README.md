# Experiments

This directory stores runner-registered pure probes for DualRefGAD Learning Signal Recovery.

Planned first bundle:

- `dualrefgad_learning_signal_abcd_probe`: one aggregate diagnostic over ABCD phases.

Rules:

- All runs must be registered through experiment-runner or documented as protocol deviation.
- Labels are diagnostic-only.
- Output JSON must include `status`, per-phase results, progress-compatible metadata, and continuation/readiness decisions.
